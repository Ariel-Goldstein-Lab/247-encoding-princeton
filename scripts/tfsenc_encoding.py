import csv
import os

import numpy as np
import pandas as pd
import pickle
import torch
import statsmodels.api as sm
from scipy.stats import pearsonr
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
    # print("Using kfolds", flush=True)
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
    # if onsets.dtype == 'object':
    onsets = elec_datum.adjusted_onset.values.astype(np.float64)
    Y = build_Y(
        elec_signal.reshape(-1, 1),
        onsets,
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


def encoding_regression_sig_coeffs(args, X, Y, folds):
    nwords = X.shape[0]  # Num of words
    nlags = Y.shape[1] if Y.shape[1:] else 1  # Num of lags
    linear_dim = args.pca_to if args.regularization == "none" else X.shape[1]
    # results = pd.DataFrame(
    # index=pd.MultiIndex.from_product(
    #     [range(nlags), range(linear_dim)],
    #     names=['nlags', 'linear_dim']
    # ),
    # columns=['lasso_coeff', 'ols_coeff', 'pvalue']
    # )

    Y -= np.mean(Y, axis=0)  # Center Y

    # Run LassoCV:
    alphas = np.logspace(args.min_alpha, args.max_alpha, args.amount_of_alphas)
    print(f"Running LassoCV, emb_dim = {X.shape[1]}", flush=True)

    model = make_pipeline(StandardScaler(), SparseGroupLassoCV(l1_regs=alphas, l21_regs=[0], groups=None,
                                                               solver_params=dict(max_iter=5000)))
    torch.cuda.empty_cache()
    model.fit(X, Y)

    Yhat = model.predict(X)
    lasso_corrs = correlation_score(Y, Yhat).T

    linmodel = model[-1]
    lasso_coeffs = linmodel.coef_.reshape(-1, nlags)  # NMTODO himalaya (linear_dim, nlags)  e.g 50 x n
    best_l1_regs = linmodel.best_l1_reg_.cpu().numpy()
    cv_scores = linmodel.cv_scores_.cpu().numpy()

    # YHAT = Yhat.cpu().reshape(-1, nlags)

    lasso_model_fitting_params = {"coeffs": lasso_coeffs,
                            "best_l1_regs": best_l1_regs,
                            "cv_scores": cv_scores,
                            }
    # results['lasso_coeff'] = lasso_coeffs.T.flatten()

    # Run OLS on Lasso coefficients
    coeffs_conversion_dict = np.empty((nlags), dtype=object)  # Each element is a dict converting index of new coeff into original coeff index
    coeffs_conversion_dict.fill(None)

    for lag in range(nlags):
        non_zero_coeffs_col = np.nonzero(lasso_coeffs[:, lag])
        col_dict = dict()
        if len(non_zero_coeffs_col) > 0:
            for new_idx, orig_idx in enumerate(non_zero_coeffs_col):
                col_dict[new_idx] = orig_idx
            coeffs_conversion_dict[lag] = col_dict

    # Using non zero coefficients of lasso to do OLS and get coeff significance (pvalues, conf intervals, etc.)
    # Note that since the number of non-zero coefficients can vary for each lag, we will do OLS for each lag separately

    ols_model_fitting_params = np.empty((nlags), dtype=object)  # Each element is a dict converting index of new coeff into original coeff index
    ols_model_fitting_params.fill(None)
    ols_r2 = np.zeros((nlags))  # R2 for each lag, size (nlags,)
    ols_r2.fill(np.nan)

    for lag in range(nlags):
        ind_lag_non_zero_coeffs = lasso_coeffs[:, lag] != 0
        if ind_lag_non_zero_coeffs.sum() > 0:
            OLS_X = X[:, ind_lag_non_zero_coeffs] # X indecies where coeffs of lasso != 0, shape (nwords, n_non_zero_coeffs)
            OLS_Y = Y[:, lag] # Y for the specific lag, shape (nwords,)

            OLS_X_with_intercept = sm.add_constant(OLS_X) # Intercept for OLS
            model = sm.OLS(OLS_Y, OLS_X_with_intercept).fit()

            # Get all statistics
            print(model.summary())

            # Access specific values
            r2 = model.rsquared # size 1
            coeffs = model.params  # size linear_dim + 1 (intercept)
            p_values = model.pvalues # size linear_dim + 1 (intercept)
            conf_intervals = model.conf_int() # size (linear_dim + 1, 2) - lower and upper bounds for each coeff

            ols_r2[lag] = r2

            ols_model_fitting_params[lag] = {
                "coeffs": coeffs,
                "p_values": p_values,
                "conf_intervals": conf_intervals,
            }

        #else: len(non_zero_vals) == 0: # No coeffs to do OLS on, all were 0, supposed to be same as - coeffs_conversion_dict[lag] == np.nan

    return (lasso_corrs, lasso_model_fitting_params, ols_r2, ols_model_fitting_params, coeffs_conversion_dict)


def encoding_regression(args, X, Y, folds):
    nwords = X.shape[0] # Num of words
    nlags = Y.shape[1] if Y.shape[1:] else 1 # Num of lags
    linear_dim = args.pca_to if args.regularization == "none" else X.shape[1]

    YHAT = np.zeros((nwords, nlags)).astype("float32")
    Ynew = np.zeros((nwords, nlags)).astype("float32")
    corrs = []
    coeffs = np.empty((args.cv_fold_num, linear_dim, nlags))
    coeffs.fill(np.nan)
    best_l1_regs = np.empty((args.cv_fold_num, nlags)) # best l1 reg (lasso) or alpha (ridge)
    best_l1_regs.fill(np.nan)
    cv_scores = np.empty((args.cv_fold_num, args.amount_of_alphas if hasattr(args, "amount_of_alphas") else 0, nlags)) # cross-validation scores of choosing alpha
    cv_scores.fill(np.nan)
    # intercepts = np.zeros((args.cv_fold_num, nlags)) # intercept (linear reg) or alpha (ridge)

    for i in range(0, args.cv_fold_num):

        Xtrain, Xtest = X[folds != i], X[folds == i]
        Ytrain, Ytest = Y[folds != i], Y[folds == i]
        Ytest -= np.mean(Ytrain, axis=0)
        Ytrain -= np.mean(Ytrain, axis=0)

        if args.regularization == "ridge":
            alphas = np.logspace(args.min_alpha, args.max_alpha, args.amount_of_alphas)
            if Xtrain.shape[0] < Xtrain.shape[1]:
                if i == 0:
                    print(f"Running KernelRidgeCV, emb_dim = {Xtrain.shape[1]}", flush=True)
                model = make_pipeline(StandardScaler(), KernelRidgeCV(alphas=alphas))
            else:
                if i == 0:
                    print(f"Running RidgeCV, emb_dim = {Xtrain.shape[1]}", flush=True)
                model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))

        elif args.regularization == "lasso":
            alphas = np.logspace(args.min_alpha, args.max_alpha, args.amount_of_alphas)
            if i == 0:
                print(f"Running LassoCV, emb_dim = {Xtrain.shape[1]}", flush=True)
            # TODO: maybe try also sklearn.linear_model.MultiTaskLassoCV or MultiTaskElasticNetCV as well
            model = make_pipeline(StandardScaler(), SparseGroupLassoCV(l1_regs=alphas, l21_regs=[0], groups=None, solver_params=dict(max_iter=5000),))

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
                    print(f"Running PCA (from {Xtrain.shape[1]} to {args.pca_to}) + OLS", flush=True)
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

        if hasattr(linmodel, "coef_"):
            coeffs[i, :, :] = linmodel.coef_.reshape(-1, nlags)  # NMTODO himalaya (n_features, n_targets)  e.g 50 x n

        if hasattr(linmodel, "best_l1_reg_"):
            if hasattr(linmodel.best_l1_reg_, 'cpu'):
                best_l1_regs[i,:] = linmodel.best_l1_reg_.cpu().numpy()
            else:
                best_l1_regs[i, :] = linmodel.best_l1_reg_.numpy()

        if hasattr(linmodel, "cv_scores_"):
            if hasattr(linmodel.cv_scores_, 'cpu'): # Tensor and not Numpy
                cv_scores[i, :, :] = linmodel.cv_scores_.cpu().numpy()
            else:
                cv_scores[i, :, :] = linmodel.cv_scores_

        # if pca:
        #     results["pca_xv"].append(model[0].explained_variance_ratio_.astype(np.float32))

        Ynew[folds == i, :] = Ytest.reshape(-1, nlags)
        if hasattr(foldYhat, 'cpu'):  # If foldYhat is a torch tensor
            foldYhat = foldYhat.cpu()  # Convert to numpy
        YHAT[folds == i, :] = foldYhat.reshape(-1, nlags)

    model_fittind_params = {"coeffs": coeffs,
                            "best_l1_regs": best_l1_regs,
                            "cv_scores": cv_scores,
                            }
    return (YHAT, Ynew, corrs, model_fittind_params)

def run_encoding_sig_coeffs(args, X, Y, folds):
    # train lm and predict
    (lasso_corrs, lasso_model_fitting_params, ols_r2, ols_model_fitting_params,
     coeffs_conversion_dict) = encoding_regression_sig_coeffs(args, X, Y, folds)

    return lasso_corrs, lasso_model_fitting_params, ols_r2, ols_model_fitting_params, coeffs_conversion_dict


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

def run_rafi_encoding(args, conversation_ids, X, Y, _, filename):
    nwords = X.shape[0] # Num of words
    nlags = Y.shape[1] if Y.shape[1:] else 1 # Num of lags
    linear_dim = args.pca_to if args.regularization == "none" else X.shape[1]

    corrs_per_diff = {i:[] for i in range(0, conversation_ids.max())}
    models = {i:None for i in conversation_ids.unique()}

    for train_conv in conversation_ids.unique():
        Xtrain = X[conversation_ids == train_conv]
        Ytrain = Y[conversation_ids == train_conv]
        Ytrain -= np.mean(Ytrain, axis=0)

        if args.regularization == "ridge":
            alphas = np.logspace(args.min_alpha, args.max_alpha, args.amount_of_alphas)
            if Xtrain.shape[0] < Xtrain.shape[1]:
                model = make_pipeline(StandardScaler(), KernelRidgeCV(alphas=alphas))
            else:
                model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
        elif args.regularization == "lasso":
            alphas = np.logspace(args.min_alpha, args.max_alpha, args.amount_of_alphas)
            # TODO: maybe try also sklearn.linear_model.MultiTaskLassoCV or MultiTaskElasticNetCV as well
            model = make_pipeline(StandardScaler(), SparseGroupLassoCV(l1_regs=alphas, l21_regs=[0], groups=None, solver_params=dict(max_iter=5000), ))
        else:  # ols, no regularization
            if args.pca_to == 0:  # No pca
                if args.himalaya:
                    model = make_pipeline(StandardScaler(), RidgeCV(alphas=[1e-9]))
                else:
                    model = make_pipeline(StandardScaler(), LinearRegression())
            else:  # pca + ols
                model = make_pipeline(
                    StandardScaler(), PCA(args.pca_to, whiten=True), LinearRegression()
                )

        torch.cuda.empty_cache()
        model.fit(Xtrain, Ytrain)

        models[train_conv] = model

        diff = 0
        while train_conv + diff <= conversation_ids.max():
            if train_conv + diff not in conversation_ids.unique():
                print(f"Conversation {train_conv + diff} not found in conversation_ids, when having train_conv of {train_conv} and diff of {diff}, skipping", flush=True)
                diff += 1
                continue
            test_conv = train_conv + diff
            Xtest = X[conversation_ids == test_conv]
            Ytest = Y[conversation_ids == test_conv]
            Ytest -= np.mean(Ytrain, axis=0)

            # Prediction & Correlation
            testYhat = model.predict(Xtest)
            fold_cors = correlation_score(Ytest, testYhat)
            corrs_per_diff[diff].append(fold_cors)

            diff += 1

    # Save
    filename = os.path.join(args.output_dir, filename)

    # Save correlations
    for diff in range(0, conversation_ids.max()):
        corr_filename = filename +f"_diff_{diff}.csv"
        corrs = corrs_per_diff[diff]

        if torch.is_tensor(corrs[-1]):  # torch tensor
            corrs = torch.stack(corrs)
        else:
            corrs = np.stack(corrs)
        if torch.is_tensor(corrs):
            corrs = corrs.cpu().numpy()

        corrs_df = pd.DataFrame(corrs)
        corrs_df.to_csv(corr_filename, index=False, header=False)

    # Save models
    models_filename = filename + "_models.pkl"
    with open(models_filename, 'wb') as f:
        pickle.dump(models, f)
    return

def run_correlations(args, X, Y, folds, filename):
    # Remove constant vectors
    X_std = np.std(X, axis=0)
    non_constant_mask = X_std > 1e-10
    X_filtered = X[:, non_constant_mask]

    # Run correlations
    X_reshaped = X_filtered[:, :, np.newaxis]
    Y_reshaped = Y[:, np.newaxis, :]
    res = pearsonr(X_reshaped, Y_reshaped, axis=0)
    cis = res.confidence_interval(confidence_level=0.955)

    # Create full matrices with NaNs for constant vectors
    full_correlations = np.full((X.shape[1], Y.shape[1]), np.nan)
    full_correlations[non_constant_mask, :] = res.statistic

    full_p_values = np.full((X.shape[1], Y.shape[1]), np.nan)
    full_p_values[non_constant_mask, :] = res.pvalue

    full_cis = np.full((X.shape[1], Y.shape[1], 2), np.nan)
    full_cis[non_constant_mask, :, 0] = cis.low
    full_cis[non_constant_mask, :, 1] = cis.high

    # Save
    filename = os.path.join(args.output_dir, filename)
    corr_filename = filename + "_corr.npy"
    np.save(corr_filename, full_correlations)
    pval_filename = filename + "_pval.npy"
    np.save(pval_filename, full_p_values)
    ci_filename = filename + "_ci.npy"
    np.save(ci_filename, full_cis)

    return

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

    return

def write_encoding_sig_coeffs_results(args, results, filename):
    """Write output into csv files

    Args:
        args (namespace): commandline arguments
        results: (correlation results, linear coefficients, intercept)
        filename: usually electrode name plus 'prod' or 'comp'. No need to add filetype ending.

    Returns:
        None
    """
    # corrs, model_fittind_params = results
    lasso_corrs, lasso_model_fitting_params, ols_r2, ols_model_fitting_params, coeffs_conversion_dict = results
    filename = os.path.join(args.output_dir, filename)

    # Save correlations
    lasso_corr_filename = filename+"_lasso.csv"
    if torch.is_tensor(lasso_corrs):
        lasso_corrs = lasso_corrs.cpu().numpy()
    lasso_corrs_df = pd.DataFrame(lasso_corrs)
    lasso_corrs_df.to_csv(lasso_corr_filename, index=False, header=False)

    ols_r2_filename = filename+"_ols.csv"
    if torch.is_tensor(ols_r2):
        ols_r2 = ols_r2.cpu().numpy()
    ols_r2_df = pd.DataFrame(ols_r2)
    ols_r2_df.to_csv(ols_r2_filename, index=False, header=False)

    # Save coefficients
    for param in lasso_model_fitting_params.keys():
        param_filename = filename+"_"+param+"_lasso"
        if torch.is_tensor(lasso_model_fitting_params[param]):
            lasso_model_fitting_params[param] = lasso_model_fitting_params[param].cpu().numpy()
        np.save(param_filename,lasso_model_fitting_params[param])

    with open(filename+"_ols.pkl", 'wb') as f:
        pickle.dump(ols_model_fitting_params, f)

    # Save coeffs conversion dict
    coeffs_conversion_filename = filename+"_coeffs_conversion_dict.npy"
    np.save(coeffs_conversion_filename, coeffs_conversion_dict)

    return
