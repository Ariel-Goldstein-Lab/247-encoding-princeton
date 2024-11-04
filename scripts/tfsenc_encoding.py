import csv
import os
import numpy as np
import pandas as pd
import torch
from numba import jit, prange
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from himalaya.ridge import GroupRidgeCV, RidgeCV, ColumnTransformerNoStack
from himalaya.kernel_ridge import (
    MultipleKernelRidgeCV,
    KernelRidgeCV,
    ColumnKernelizer,
    Kernelizer,
)
from himalaya.scoring import correlation_score, correlation_score_split


# @jit(nopython=True)
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
    half_window = round((window_size / 1000) * 512 / 2)  # convert to signal fs
    Y1 = np.zeros((len(onsets), len(lags)))

    for lag in prange(len(lags)):
        lag_amount = int(lags[lag] / 1000 * 512)  # convert fo signal fs
        index_onsets = np.round_(onsets, 0, onsets) + lag_amount  # lag onsets
        starts = index_onsets - half_window - 1  # lag window onset
        stops = index_onsets + half_window  # lag window offset
        starts = np.array(starts, dtype=int)
        stops = np.array(stops, dtype=int)
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
    print("Using kfolds")
    skf = KFold(n_splits=fold_num, shuffle=False)
    folds = [t[1] for t in skf.split(np.arange(X.shape[0]))]
    fold_cat = np.zeros(X.shape[0])
    for i in range(0, len(folds)):
        for row in folds[i]:
            fold_cat[row] = i  # turns into fold category
    return fold_cat


def encoding_setup(args, elec_name, elec_datum, elec_signal):

    # Build x and y matrices
    X = np.stack(elec_datum.embeddings).astype("float64")
    Y = build_Y(
        elec_signal.reshape(-1, 1),
        elec_datum.adjusted_onset.values,
        np.array(args.lags),
        args.window_size,
    )
    X = X.astype("float32")
    Y = Y.astype("float32")

    # TODO: HERE SUBSET BASED ON ANNOTS
    if "data_subset_type" in args:
        if args.data_subset_type == "ref_recap_only":
            X = X[elec_datum.annot_type.isin(["ref", "recap", "mapping"]), :]
            Y = Y[elec_datum.annot_type.isin(["ref", "recap", "mapping"]), :]
            elec_datum = elec_datum[
                elec_datum.annot_type.isin(["ref", "recap", "mapping"])
            ]

    # Split into production and comprehension
    prod_X = X[elec_datum.speaker == "Speaker1", :]
    comp_X = X[elec_datum.speaker != "Speaker1", :]
    prod_Y = Y[elec_datum.speaker == "Speaker1", :]
    comp_Y = Y[elec_datum.speaker != "Speaker1", :]

    # Print shapes
    print(f"Prod X: {prod_X.shape}, Prod Y: {prod_Y.shape}")
    print(f"Comp X: {comp_X.shape}, Comp Y: {comp_Y.shape}")

    # get folds
    if args.project_id == "podcast":  # podcast
        fold_cat_comp = get_kfolds(comp_X, args.cv_fold_num)
        fold_cat_prod = []
    elif len(args.conv_ids) < args.cv_fold_num * 2:  # small num of convos, also 798
        print(
            f"{args.sid} {elec_name} has less convos than fold nums, doing kfold instead"
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

    nSamps = X.shape[0]
    nChans = Y.shape[1] if Y.shape[1:] else 1

    YHAT = np.zeros((nSamps, nChans)).astype("float32")
    Ynew = np.zeros((nSamps, nChans)).astype("float32")
    corrs = []

    # TODO: TAKE OUT FIGURE OUT WHAT IS HAPPENING.
    Y = np.nan_to_num(Y)

    for i in range(0, args.cv_fold_num):

        Xtrain, Xtest = X[folds != i], X[folds == i]
        Ytrain, Ytest = Y[folds != i], Y[folds == i]
        Ytest -= np.mean(Ytrain, axis=0)
        Ytrain -= np.mean(Ytrain, axis=0)

        if not args.ridge:  # ols
            if args.pca_to == 0:
                print(f"Running OLS, emb_dim = {Xtrain.shape[1]}")
                if args.himalaya:
                    model = make_pipeline(StandardScaler(), RidgeCV(alphas=[1e-9]))
                else:
                    model = make_pipeline(StandardScaler(), LinearRegression())
            else:  # pca + ols
                print(f"Running PCA (from {Xtest.shape[1]} to {args.pca_to}) + OLS")
                model = make_pipeline(
                    StandardScaler(), PCA(args.pca_to, whiten=True), LinearRegression()
                )
        else:  # ridge cv
            alphas = np.logspace(0, 20, 10)
            solver_params = {"n_alphas_batch": 5}
            if Xtrain.shape[0] < Xtrain.shape[1]:
                print(f"Running KernelRidgeCV, emb_dim = {Xtrain.shape[1]}")
                model = make_pipeline(StandardScaler(), KernelRidgeCV(alphas=alphas))
            else:
                print(f"Running RidgeCV, emb_dim = {Xtrain.shape[1]}")
                model = make_pipeline(
                    StandardScaler(),
                    RidgeCV(alphas=alphas, solver_params=solver_params),
                )
        torch.cuda.empty_cache()
        model.fit(Xtrain, Ytrain)

        # Prediction & Correlation
        foldYhat = model.predict(Xtest)
        fold_cors = correlation_score(Ytest, foldYhat)
        corrs.append(fold_cors)

        Ynew[folds == i, :] = Ytest.reshape(-1, nChans)
        YHAT[folds == i, :] = foldYhat.reshape(-1, nChans)

    return (YHAT, Ynew, corrs)


def run_encoding(args, X, Y, folds):

    # train lm and predict
    Y_hat, Y_new, corrs = encoding_regression(args, X, Y, folds)

    # # Old correlation
    # rps = []
    # # correlation for whole datum
    # rp, _, _ = encoding_correlation(Y_new, Y_hat)
    # rps.append(rp)

    # # correlation per folds
    # for i in np.unique(folds).astype(int):
    #     rp_fold, _, _ = encoding_correlation(Y_new[folds == i], Y_hat[folds == i])
    #     rps.append(rp_fold)

    # New correlation
    corr_datum = correlation_score(Y_new, Y_hat)
    corrs.append(corr_datum)
    if torch.is_tensor(corr_datum):  # torch tensor
        corrs = torch.stack(corrs)
    else:
        corrs = np.stack(corrs)

    return corrs, Y_hat, Y_new


def write_encoding_results(args, results, Y_hat, Y_new, filename, folds=None):
    """Write output into csv files

    Args:
        args (namespace): commandline arguments
        results: correlation results
        filename: usually electrode name plus 'prod' or 'comp'

    Returns:
        None
    """
    filename = os.path.join(args.output_dir, filename)
    if torch.is_tensor(results):
        results = results.cpu().numpy()
    results_df = pd.DataFrame(results)
    results_df.to_csv(filename, index=False, header=False)
    if args.save_preds:
        np.savez(
            filename.replace(".csv", ".npz"), Y_hat=Y_hat, Y_new=Y_new, folds=folds
        )

    return
