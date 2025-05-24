import csv
import os

import numpy as np
import pandas as pd
import torch
from himalaya.kernel_ridge import (ColumnKernelizer, Kernelizer, KernelRidgeCV,
                                   MultipleKernelRidgeCV)
from himalaya.ridge import ColumnTransformerNoStack, GroupRidgeCV, RidgeCV
from himalaya.lasso import SparseGroupLassoCV
from himalaya.scoring import correlation_score, correlation_score_split
from numba import jit, prange
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@jit(nopython=True)
def build_Y(brain_signal, onsets, lags, window_size):
    """[summary]

    Args:
        onsets ([type]): [description]
        brain_signal ([type]): [description]
        lags ([type]): [description]
        window_size ([type]): [description]

    Returns:
        [type]: [description]
    """
    FS = 512  # frame rate
    MS = 1000  # convert from seconds to milliseconds
    half_window = round((window_size / MS) * FS / 2)  # convert to signal fs
    Y1 = np.zeros((len(onsets), len(lags)))

    for lag in prange(len(lags)):
        lag_amount = int(lags[lag] / MS * FS)  # convert fo signal fs
        index_onsets = np.round(onsets, 0, onsets) + lag_amount  # lag onsets
        starts = index_onsets - half_window - 1  # lag window onset
        stops = index_onsets + half_window  # lag window offset
        for i, (start, stop) in enumerate(zip(starts, stops)):
            Y1[i, lag] = np.mean(
                brain_signal[start:stop].reshape(-1)
            )  # average lag window

    return Y1


def get_groupkfolds(datum, X, Y, fold_num=10):
    fold_cat = np.zeros(datum.shape[0])
    grpkfold = GroupKFold(n_splits=fold_num)
    folds = [t[1] for t in grpkfold.split(X, Y, groups=datum["conversation_id"])]

    for i in range(0, len(folds)):
        for row in folds[i]:
            fold_cat[row] = i  # turns into fold category

    fold_cat_comp = fold_cat[datum.speaker != "Speaker1"]
    fold_cat_prod = fold_cat[datum.speaker == "Speaker1"]

    return (fold_cat_comp, fold_cat_prod)


def get_kfolds(X, fold_num=10):
    print("Using kfolds", flush=True)
    skf = KFold(n_splits=fold_num, shuffle=False)
    folds = [t[1] for t in skf.split(np.arange(X.shape[0]))]
    fold_cat = np.zeros(X.shape[0])
    for i in range(0, len(folds)):
        for row in folds[i]:
            fold_cat[row] = i  # turns into fold category
    return fold_cat


def encoding_setup(args, elec_name, elec_datum, elec_signal):

    # Build x and y matrices
    X = np.stack(elec_datum.embeddings).astype("float64") # embeddings
    Y = build_Y(
        elec_signal.reshape(-1, 1),
        elec_datum.adjusted_onset.values,
        np.array(args.lags),
        args.window_size,
    ) # Signal (cut by word onset)
    X = X.astype("float32")
    Y = Y.astype("float32")

    # Split into production and comprehension
    prod_X = X[elec_datum.speaker == "Speaker1", :]
    comp_X = X[elec_datum.speaker != "Speaker1", :]
    prod_Y = Y[elec_datum.speaker == "Speaker1", :]
    comp_Y = Y[elec_datum.speaker != "Speaker1", :]

    # get folds
    if args.project_id == "podcast":  # podcast
        fold_cat_comp = get_kfolds(comp_X, args.cv_fold_num)
        fold_cat_prod = []
    elif len(args.conv_ids) < args.cv_fold_num:  # small num of convos (< 10)
        print(
            f"{args.sid} {elec_name} has less convos than fold nums, doing kfold instead", flush=True
        )
        fold_cat_comp = get_kfolds(comp_X, args.cv_fold_num)
        fold_cat_prod = get_kfolds(prod_X, args.cv_fold_num)
    else:  # Get groupkfolds
        fold_cat_comp, fold_cat_prod = get_groupkfolds(
            elec_datum, X, Y, args.cv_fold_num
        )

    # Oragnize
    comp_data = comp_X, comp_Y, fold_cat_comp
    prod_data = prod_X, prod_Y, fold_cat_prod

    return comp_data, prod_data


def encoding_correlation(CA, CB):
    """[summary]

    Args:
        CA ([type]): [description]
        CB ([type]): [description]

    Returns:
        [type]: [description]
    """
    df = np.shape(CA)[0] - 2

    CA -= np.mean(CA, axis=0)
    CB -= np.mean(CB, axis=0)

    r = np.sum(CA * CB, 0) / np.sqrt(np.sum(CA * CA, 0) * np.sum(CB * CB, 0))

    t = r / np.sqrt((1 - np.square(r)) / df)
    p = stats.t.sf(t, df)

    r = r.squeeze()

    if r.size > 1:
        r = r.tolist()
    else:
        r = float(r)

    return r, p, t


def encoding_regression(args, X, Y, folds):

    amount_alphas_to_check = 30

    nwords = X.shape[0] # Num of words
    nlags = Y.shape[1] if Y.shape[1:] else 1 # Num of lags
    linear_dim = args.pca_to if args.regularization == "none" else X.shape[1]

    YHAT = np.zeros((nwords, nlags)).astype("float32")
    Ynew = np.zeros((nwords, nlags)).astype("float32")
    corrs = []
    coeffs = np.zeros((args.cv_fold_num, linear_dim, nlags))
    best_l1_regs = np.zeros((args.cv_fold_num, nlags))  # best l1 reg (lasso) or alpha (ridge)
    cv_scores = np.zeros((args.cv_fold_num, amount_alphas_to_check, nlags))  # cross-validation scores

    # intercepts = np.zeros((args.cv_fold_num, nlags)) # intercept (linear reg) or alpha (ridge)

    for i in range(0, args.cv_fold_num):

        Xtrain, Xtest = X[folds != i], X[folds == i]
        Ytrain, Ytest = Y[folds != i], Y[folds == i]
        Ytest -= np.mean(Ytrain, axis=0)
        Ytrain -= np.mean(Ytrain, axis=0)

        if args.regularization == "ridge":
            alphas = np.logspace(0, 20, amount_alphas_to_check)
            if Xtrain.shape[0] < Xtrain.shape[1]:
                if i == 0:
                    print(f"Running KernelRidgeCV, emb_dim = {Xtrain.shape[1]}", flush=True)
                model = make_pipeline(StandardScaler(), KernelRidgeCV(alphas=alphas))
            else:
                if i == 0:
                    print(f"Running RidgeCV, emb_dim = {Xtrain.shape[1]}", flush=True)
                model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))

        elif args.regularization == "lasso":
            alphas = np.logspace(-1, 20, amount_alphas_to_check)
            if i == 0:
                print(f"Running LassoCV, emb_dim = {Xtrain.shape[1]}", flush=True)
            # TODO: maybe try also sklearn.linear_model.MultiTaskLassoCV or MultiTaskElasticNetCV as well
            model = make_pipeline(StandardScaler(), SparseGroupLassoCV(l1_regs=alphas, l21_regs=[0], groups=None, solver_params=dict(max_iter=1000),))

        else:  # ols, no regularization
            if args.pca_to == 0: # No pca
                if i == 0:
                    print(f"Running OLS, emb_dim = {Xtrain.shape[1]}", flush=True)
                if args.himalaya:
                    model = make_pipeline(StandardScaler(), RidgeCV(alphas=[1e-9]))
                else:
                    model = make_pipeline(StandardScaler(), LinearRegression())
            else:  # pca + ols
                if i == 0:
                    print(f"Running PCA (from {Xtest.shape[1]} to {args.pca_to}) + OLS", flush=True)
                model = make_pipeline(
                    StandardScaler(), PCA(args.pca_to, whiten=True), LinearRegression()
                )

        torch.cuda.empty_cache()
        model.fit(Xtrain, Ytrain)

        # Prediction & Correlation
        foldYhat = model.predict(Xtest)
        fold_cors = correlation_score(Ytest, foldYhat)
        corrs.append(fold_cors)

        # Coeff & bias
        linmodel = model[-1]

        coeffs[i, :, :] = linmodel.coef_.reshape(-1, nlags)  # NMTODO himalaya (n_features, n_targets)  e.g 50 x n
        best_l1_regs[i,:] = linmodel.best_l1_reg_.cpu().numpy()
        cv_scores[i, :, :] = linmodel.cv_scores_.cpu().numpy()

        Ynew[folds == i, :] = Ytest.reshape(-1, nlags)
        YHAT[folds == i, :] = foldYhat.cpu().reshape(-1, nlags)

    model_fittind_params = {"coeffs": coeffs,
                            "best_l1_regs": best_l1_regs,
                            # "cv_scores": cv_scores,
                            }
    return (YHAT, Ynew, corrs, model_fittind_params)


def run_encoding(args, X, Y, folds):

    # train lm and predict
    Y_hat, Y_new, corrs, model_fittind_params = encoding_regression(args, X, Y, folds)

    # Correlation over all folds
    # corr_datum = correlation_score(Y_new, Y_hat)
    # corrs.append(corr_datum)

    if torch.is_tensor(corrs[-1]):  # torch tensor
        corrs = torch.stack(corrs)
    else:
        corrs = np.stack(corrs)

    return corrs, model_fittind_params


def write_encoding_results(args, results, filename):
    """Write output into csv files

    Args:
        args (namespace): commandline arguments
        results: (correlation results, linear coefficients, intercept)
        filename: usually electrode name plus 'prod' or 'comp'. No need to add filetype ending.

    Returns:
        None
    """
    corrs, model_fittind_params = results
    filename = os.path.join(args.output_dir, filename)

    # Save correlations
    corr_filename = filename+".csv"
    if torch.is_tensor(corrs):
        corrs = corrs.cpu().numpy()
    corrs_df = pd.DataFrame(corrs)
    corrs_df.to_csv(corr_filename, index=False, header=False)

    # Save coefficients
    for param in model_fittind_params.keys():
        param_filename = filename+"_"+param
        if torch.is_tensor(model_fittind_params[param]):
            model_fittind_params[param] = model_fittind_params[param].cpu().numpy()
        np.save(param_filename,model_fittind_params[param])

    # Save intercepts
    # intercepts_filename = filename+"_intercepts.pkl"
    # np.save(intercepts_filename, intercepts)

    return
