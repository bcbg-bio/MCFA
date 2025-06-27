# pylint: disable=invalid-name
"""Multiset Correlation and Factor Analysis

Contains functions for calculating correlated and private factors for
multiple high-dimensional datasets.

Usage:
  TODO(brielin): Add usage example
"""

import numpy as np
import pandas as pd
import torch
from torch import multiprocessing
from dataclasses import dataclass
from typing import Iterable, List, Union
from sklearn import model_selection
from sklearn import preprocessing
from multicor_fa import _em, _pca, _initializers, _cv


# This is required or the pool code below hands on .join().
# force=True is required or I get an error that the context
# is already set. I have absolutely no idea why this works.
# https://pythonspeed.com/articles/python-multiprocessing/
multiprocessing.set_start_method('spawn', force=True)


@dataclass
class MCFARes:
    """Simple dataclass for storing MCFA results."""
    data_pcs: List[_pca.PCARes]
    Z: pd.DataFrame
    X: Iterable[pd.DataFrame]
    W: Iterable[pd.DataFrame]
    L: Iterable[pd.DataFrame]
    Phi: Iterable[pd.DataFrame]
    rho: pd.Series
    lam: Iterable[pd.Series]
    var_exp_Z: Iterable[pd.Series]
    var_exp_X: Iterable[pd.Series]
    l: List[float]
    cd: List[float]
    n_pcs: List[int]
    d: int
    k: List[int]
    center: bool
    scale: bool
    init: str
    maxit: int
    delta: float
    device: str
    rcond: float


def score(data, Z, transform = True):
    """Calculates (transformed) correlations between model factors and data.

    Args:
      data: A pandas DataFrame with features as columns. The original data.
      Z: The factor set used to calculate gene statistics. Usually mcfa_res.Z.
      transform: bool. True to transform correlations to Z-scores.
    Returns:
      A pd.DataFrame with rows as data features and columns as factors, entires
      are (optionally Z-transformed) correlations.
    """
    n = Z.shape[0]
    # Note: this cannot be done in pytorch because it does not get
    #  along with concurrent.futures.
    cors = preprocessing.scale(data).T.dot(preprocessing.scale(Z)) / n
    if transform:
        cors = np.sqrt(n-3) * np.arctanh(cors)
    cors = pd.DataFrame(cors, index=data.columns)
    return cors


def cv(Y: Iterable[pd.DataFrame], mcfa_res: MCFARes,
       folds: Union[str, int] = 10, threads: [int] = 1,
       verbose: bool = False):
    """Checks for over-fitting using k-fold cross validation.

    Args:
      Y: Iterable of N (samples) by p_m (features) pandas DataFrames, the
        M=len(Y) datasets to analyze.
      mcfa_res: An MCFAres dataclass fit from Y.
      folds: Integer, number of folds. Or 'loo' for leave-one-out.
      threads: Integer, number of paralell folds to run.
    Returns:
      A tuple Z, X with each entry formed by removing that row, fitting the
      MCFA model, and projecting that sample's Y into the space fit without
      it.
    """
    if folds == 'loo':
        folds = mcfa_res.Z.shape[0]
    elif isinstance(folds, str):
        raise NotImplementedError

    cv_iter = model_selection.KFold(n_splits=folds)
    results = []
    X_hat = []
    Z_hat = []
    nrmse_tr = []
    nrmse_te = []
    if threads > 0:
        raise NotImplementedError
        # with multiprocessing.get_context('spawn').Pool(threads) as pool:
        #     for indices in enumerate(cv_iter.split(mcfa_res.Z)):
        #         results.append(pool.apply_async(_cv_one_iter, (
        #             indices, Y, mcfa_res.Z, mcfa_res.X, mcfa_res.n_pcs,
        #             mcfa_res.d, mcfa_res.k, mcfa_res.center,
        #             mcfa_res.scale, mcfa_res.init, mcfa_res.maxit, mcfa_res.delta,
        #             mcfa_res.device, mcfa_res.rcond, verbose)))
        #     pool.close()
        #     pool.join()
    else:
        for indices in enumerate(cv_iter.split(mcfa_res.Z)):
            results.append(_cv_one_iter(
                indices, Y, mcfa_res.Z, mcfa_res.X, mcfa_res.n_pcs,
                mcfa_res.d, mcfa_res.k, mcfa_res.center,
                mcfa_res.scale, mcfa_res.init, mcfa_res.maxit, mcfa_res.delta,
                mcfa_res.device, mcfa_res.rcond, verbose))

    for res in results:
        Z_res, X_res, nrmse_tr_res, nrmse_te_res = res.get() if threads > 0 else res
        X_hat.append(X_res)
        Z_hat.append(Z_res)
        nrmse_tr.append(nrmse_tr_res)
        nrmse_te.append(nrmse_te_res)

    X_hat = [np.concatenate(X_m, 0) for X_m in map(list, zip(*X_hat))]
    Z_hat = np.concatenate(Z_hat, 0)

    ds_names = None
    if isinstance(Y, dict):
        ds_names = Y.keys()

    Z_names = ['Z' + str(i+1) for i in range(mcfa_res.d)]
    if ds_names is not None:
        X_names = [['X' + str(i+1) + '_' + name for i in range(k_m)]
                   for name, k_m in zip(ds_names, mcfa_res.k)]
    else:
        X_names = [['X' + str(i+1) + '_' + str(m+1) for i in range(k_m)]
                   for m, k_m in enumerate(mcfa_res.k)]
    Z_hat = pd.DataFrame(Z_hat, index=mcfa_res.Z.index, columns=Z_names)
    X_hat = [pd.DataFrame(X_m, index=mcfa_res.Z.index, columns=names)
         for X_m, names in zip(X_hat, X_names)]
    nrmse_tr = pd.DataFrame(
        np.array(nrmse_tr), index=['fold_' + str(i) for i in range(folds)])
    nrmse_te = pd.DataFrame(
        np.array(nrmse_te), index=['fold_' + str(i) for i in range(folds)])

    if ds_names is not None:
        X_hat = dict(zip(ds_names, X_hat))
        nrmse_tr.columns = ds_names
        nrmse_te.columns = ds_names

    return Z_hat, X_hat, nrmse_tr, nrmse_te


# TODO(brielin): EM is broken if you don't center.
def fit(Y: Iterable[pd.DataFrame], n_pcs: Union[str, List[int]] = 'infer',
        d: Union[str, int] = 'infer', k: Union[str, List[int]] = 'infer',
        missing_entries: str = 'raise', missing_modes: str = 'raise',
        center: bool = True, scale: bool = True, init: str = 'avgvar',
        result_space: str = 'full', maxit: int = 1000, delta: float = 1e-6,
        device = 'cpu', rcond: float = 1e-8, verbose: bool = True):
    """Interface function to the MCFA estimators.

    Args:
        Y: Iterable of N_m (samples) by p_m (features) pandas DataFrames, the
          M=len(Y) datasets to analyze. If a dictionary, keys will be used
          as names in the results.
        center: Bool. True to mean-center columns of Y.
        scale: Bool. True to variance-scale columns of Y to 1.0.
        n_pcs: 'infer', 'all' or a list of length M of integers. The number
          of PCs of each dataset to keep. 'all' does not whiten/PCA data
          prior to modeling, 'infer' uses the Marchenko-Pasteur
          cutoff to choose PCs. A list of integers
          specifies the number of PCs to keep from each dataset.
        d: 'infer', 'all' or integer. Dimensionality of the hidden space.
          If 'infer' a simulation will be done to determine the number
          of correlated components to keep.
        k: 'infer', None, or list of integers. Number of private components
          to model per dataset. If 'infer', k is set to n_pcs - d. If None,
          no private components will be modeled.
        missing_entries: Either 'raise', 'drop', 'skip', 'impute_model', or 'impute_mean'.
          Instructions for what to do with missing entries of observed samples
          in each mode. 'raise' will raise an error, 'skip' will ignore these
          terms in the likelihood, 'impute_model' will impute them using
          ppca, 'impute_mean' will impute them with the mean of other samples.
        missing_modes: Either 'raise', 'drop', 'skip', 'impute_model', or 'impute_mean'.
          Instructions for what to do with unobserved data modes in some samples.
          'raise' will raise an error if any are detected. 'skip' will ignore
          these samples in the likelihood. 'impute_model' will impute them using
          the model. 'impute_mean' will impute them using the mean of other
          observations.
        init: Either 'avgvar', 'avgnorm' or 'random'. Initialization
          strategy. 'avgvar' (default) maximizes the sum of correlations
          with a global mahalalanobis constraint (eg Parra 2019), 'avgnorm'
          does the same with a global euclidean constraint (eg Seurat),
          and random samples W from a std normal while setting Phi to I. Note
          that if n_pcs is not 'all', the model is fit explicitly to the PCs
          of each dataset and therefore the 'avgvar' and 'avgnorm'
          initializations are equivalent.
        result_space: Either 'full' or 'pc' (for informative analyses only).
          If 'full', weight matrices (W, L) will be transformed back to
          the observed space (gene features). If 'pc', weight matrices will
          remain in pc space (pc features). Note that noise matrices (phi)
          are left untransformed because the full pxp noise matrix can be
          enormous.
        maxit: Maximum number of iterations of the pgm to run. Set to 0
          to run none and return only the initial solution.
        delta: Float, convergance tolerance for EM. Set to None to run for
          maxit iterations.
        rcond: Float, zero tolerance for least squares routines.
        verbose: Bool. True to print progress.
    Returns:
        an MCFARes instance.
    Raises:
        NotImplementedError: if a TODO feature is called.
        ValueError: if the input data matrices have a different number
          of rows.
    """
    if isinstance(n_pcs, str) & (n_pcs not in ['infer', 'all']):
        raise NotImplementedError(
            'n_pcs must be "infer", "all" or a list of integers.')

    if isinstance(d, str) & (d not in ['infer', 'all']):
        raise NotImplementedError(
            'd must be "infer", "all" or an integer.')

    if isinstance(k, str) & (k not in ['infer', 'all']):
        raise NotImplementedError(
            'k must be "infer", "all" or a list of integers.')

    if init not in ['avgvar', 'avgnorm', 'random']:
        raise NotImplementedError(
            'Implemented initializers are avgnorm, avgvar, and random.')

    if isinstance(result_space, str) & (result_space not in ['pc', 'full']):
        raise NotImplementedError(
            'n_pcs must be "infer", "all" or a list of integers.')

    if isinstance(missing_entries, str) & \
    (missing_entries not in ['raise', 'drop', 'skip', 'impute_model', 'impute_mean']):
        raise NotImplementedError(
            'missing_entries must be "raise", "drop", "skip",'\
            ' "impute_model" or "impute_mean".')

    if isinstance(missing_modes, str) & \
    (missing_modes not in ['raise', 'drop', 'skip', 'impute_model', 'impute_mean']):
        raise NotImplementedError(
            'missing_modes must be "raise", "drop",'\
            '"skip", "impute_model" or "impute_mean".')
    
    ds_names = None
    if isinstance(Y, dict):
        ds_names = Y.keys()
        Y = list(Y.values())
    # Drop rows which are entirely NA to conform to expectations below
    Y = [Y_m.drop(Y_m.index[pd.isna(Y_m).all(axis=1)]) for Y_m in Y]
    sample_names = [Y_m.index for Y_m in Y]
    feature_names = [Y_m.columns for Y_m in Y]

    if any(any(pd.isna(Y_m)) for Y_m in Y):
        if missing_entries == 'raise':
            raise ValueError('Missing entries detected in some datasets')
        elif missing_entries == 'drop':
            print('Missing values detected in input, dropping samples with missing data.')
            [Y_m.dropna(inplace=True) for Y_m in Y]
        elif missing_entries == 'impute_mean':
            print('Missing values detected in input, imputing with mean.')
            [Y_m.fillna(Y_m.mean(), inplace=True) for Y_m in Y]
        elif missing_entries == 'skip':
            print('Missing values detected in input, they will be skipped.')
        elif missing_entries == 'impute_model':
            print('Missing values detected in input, they will be imputed during'
                  'model fitting.')

    common_samples = sample_names[0]
    all_samples = sample_names[0]
    use_samples = None
    for names in sample_names[1:]:
        common_samples = common_samples.intersection(names)
        all_samples = all_samples.union(names)
    N_common = len(common_samples)
    if any(Y_m.shape[0] > N_common for Y_m in Y):
        if missing_modes == 'raise':
            raise ValueError('Missing modes detected for some samples.')
        elif missing_modes == 'drop':
            print('Missing modes detected for some samples, dropping samples'
                  'with missing modes. There are {0:d} samples remaining.'.format(N_common))
            Y = [Y_m.loc[common_samples] for Y_m in Y]
            if N_common <= 1:
                raise ValueError('One or fewer samples remain.')
            use_samples = common_samples
        elif missing_modes == 'impute_mean':
            print('Missing modes detected for some samples, Imputing with the mean')
            Y = [pd.concat([Y_m, pd.DataFrame(index=all_samples.difference(Y_m.index),
                                              columns=Y_m.columns)]).fillna(Y_m.mean()) for Y_m in Y]
        elif missing_modes == 'skip':
            print('Missing modes detected in input, they will be skipped.')
            Y = [pd.concat([Y_m, pd.DataFrame(index=all_samples.difference(Y_m.index),
                                              columns=Y_m.columns)]) for Y_m in Y]
        elif missing_modes == 'impute_model':
            print('Missing modes detected in input, they will be imputed during'
                  'model fitting.')
            Y = [pd.concat([Y_m, pd.DataFrame(index=all_samples.difference(Y_m.index),
                                              columns=Y_m.columns)]) for Y_m in Y]

    # Rearrange to match index
    Y = [Y_m.loc[use_samples] for Y_m in Y]

    if isinstance(n_pcs, List):
        if len(n_pcs) != len(Y):
            raise ValueError(
                'Length of PC list does not match number of datasets.')

    if isinstance(k, List):
        if len(k) != len(Y):
            raise ValueError(
                'Length of private list does not match number of datasets.')

    # TODO(brielin): this needs to be a list if n_pcs is allowed to have
    #   individual entries be 'all' so only some datasets are processed
    #   with informative PCs.
    informative = (n_pcs == 'infer')  | isinstance(n_pcs, List)

    M = len(Y)
    if n_pcs == 'all':
        n_pcs = ['all']*M
    elif n_pcs == 'infer':
        n_pcs = ['infer']*M

    if verbose: print('Calculating data PCs.')

    Y_pcs = [_pca.pca(Y_m, n_pc_m, center, scale)
             for Y_m, n_pc_m in zip(Y, n_pcs)]
    if informative and result_space == 'pc':
        feature_names = [['pc_' + str(k+1) for k in range(Y_pc.k)]
                         for Y_pc in Y_pcs]

    if informative:
        p = [pc.k for pc in Y_pcs]
        n_pcs = p
        Y_all = torch.cat([torch.from_numpy(pc.pcs.values)
                           for pc in Y_pcs], axis=1)
    else:
        p = [Y_m.shape[1] for Y_m in Y]
        n_pcs = None
        if center | scale:
            Y_all = torch.cat(
                [torch.from_numpy(preprocessing.scale(
                    Y_m, with_mean=center, with_std=scale)) for Y_m in Y],
                axis = 1)
        else:
            Y_all = torch.cat([torch.from_numpy(Y_m.values) for Y_m in Y],
                              axis=1)
    if verbose: print('Calculating empirical covariance.')
    mask = ~torch.isnan(Y_all)
    N = mask.type(torch.int64).T @ mask.type(torch.int64)
    Sigma_hat = Y_all.nan_to_num().T @ Y_all.nan_to_num() / N
    psum = np.concatenate([[0], np.cumsum(p, 0)])
    p_all = sum(p)

    # TODO(brielin): This is doing a little extra work/memory if d == 'infer'
    #   and init = 'avgvar' (the default). Also note that _init_ methods may
    #   no longer need to return rho and ppca may no longer need to return vals.
    if verbose: print('Initialzing model.')
    if d == 'all':
        d = p_all
    elif d == 'infer':
        if verbose: print('Inferring the shared dimensionality.')
        rho_min, _ = _rho_mp_sim(N, p)
        U_all = torch.cat([pc.U for pc in Y_pcs], dim = 1)
        UTU = U_all.T @ U_all
        rho0 = torch.linalg.eigvalsh(UTU)
        d = sum(rho0 > rho_min)
        if verbose:
            print(('There are {} components above rho' +
                   ' inclusion threshold {}.').format(d, rho_min))

    if init == 'random':
        W0 = [torch.randn((p_m, d)).double() for p_m in p]
    elif init == 'avgnorm':
        W0, _ = _init_norm_W(Sigma_hat, psum, d, M)
    elif init == 'avgvar':
        W0, _  = _init_var_W(Y_pcs, psum, d, informative)

    # TODO(brielin): Edge case of some mp_dim < d
    if k == 'infer':
        k = [pca_m.mp_dim - d for pca_m in Y_pcs]

    if init == 'random':
        L0 = None if k is None else [
            torch.randn((p_m, k_m)).double() for k_m, p_m in zip(k, p)]
        Phi0 = [torch.eye(p_m) for p_m in p]
    else:
        L0, Phi0 = _init_L_Phi(Sigma_hat, W0, psum, p, k)

    if verbose: print('Fitting the model.')
    W, L, Phi, l, cd = _em.fit_EM_iter(
        Y_all, Sigma_hat, W0, L0, Phi0, maxit, device, rcond, delta, verbose, impute = (missing_modes == 'impute_model'))
    rho = _em.calculate_rho(W, L, Phi, Y_all, device, rcond, 'genvar')
    rho, order = torch.sort(rho, descending=True)

    W = [W_m[:, order] for W_m in W]
    Z, X = _em.get_latent(W, L, Phi, Y_all, device, rcond)

    if verbose: print('Calculating feature importance.')
    if informative and (result_space == 'full'):
        W = [pc_m.V @ W_m for W_m, pc_m in zip(W, Y_pcs)]
        L = None if L is None else [pc_m.V @ L_m for L_m, pc_m in zip(L, Y_pcs)]

    Z_names = ['Z' + str(i+1) for i in range(d)]
    if ds_names is not None:
        X_names = [['X' + str(i+1) + '_' + name for i in range(k_m)]
                   for name, k_m in zip(ds_names, k)]
    else:
        X_names = [['X' + str(i+1) + '_' + str(m+1) for i in range(k_m)]
                   for m, k_m in enumerate(k)]
    Z = pd.DataFrame(Z.numpy(), index=common_samples, columns=Z_names)
    X = [pd.DataFrame(X_m.numpy(), index=common_samples, columns=names)
         for X_m, names in zip(X, X_names)]
    W = [pd.DataFrame(W_m.numpy(), index=names, columns=Z_names)
         for W_m, names in zip(W, feature_names)]
    rho = pd.Series(rho, index=Z_names)
    Phi = [pd.DataFrame(phi.numpy()) for phi in Phi]

    if scale:
        var_exp_Z = [(W_m**2).sum(0)/W_m.shape[0] for W_m in W]
    else:
        var_exp_Z = [(W_m**2).sum(0)/sum(Y_m.var(0))
                     for W_m, Y_m in zip(W, Y)]
    var_exp_Z = pd.concat(var_exp_Z, axis=1)

    var_exp_X = None
    lam = None
    if L is not None:
        L = [pd.DataFrame(L_m.numpy(), index=ind_names, columns=col_names)
             for L_m, ind_names, col_names in zip(L, feature_names, X_names)]
        lam = [(L_m**2).sum(0) for L_m in L]
        if scale:
            var_exp_X = [l/L_m.shape[0] for l, L_m in zip(lam, L)]
        else:
            var_exp_X = [l/sum(Y_m.var(0)) for l, Y_m in zip(lam, Y)]

    if ds_names is not None:
        X = dict(zip(ds_names, X))
        Y_pcs = dict(zip(ds_names, Y_pcs))
        W = dict(zip(ds_names, W))
        L = dict(zip(ds_names, L))
        Phi = dict(zip(ds_names, Phi))
        lam = dict(zip(ds_names, lam))
        var_exp_Z.columns = ds_names
        var_exp_X = dict(zip(ds_names, var_exp_X))

    return MCFARes(Y_pcs, Z, X, W, L, Phi, rho, lam, var_exp_Z, var_exp_X, l,
                   cd, n_pcs, d, k, center, scale, init, maxit, delta,
                   device, rcond)
