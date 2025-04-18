# pylint: disable=invalid-name
"""CV routines for MCFA model.

Do not call directly, see mcfa.py for usage.
"""

import pandas as pd
import torch
from typing import Iterable, List, Union


def _cv_one_iter(
        indices, Y: Iterable[pd.DataFrame],
        Z: pd.DataFrame, X: Iterable[pd.DataFrame],
        n_pcs: Union[str, List[int]] = 'infer',
        d: Union[str, int] = 'infer', k: Union[str, List[int]] = 'infer',
        center: bool = True, scale: bool = True, init: str = 'avgvar',
        maxit: int = 1000, delta: float = 1e-6,
        device = 'cpu', rcond: float = 1e-8, verbose: bool = False):
    # TODO(brielin): At the moment, this is assuming that we're doing
    #   pc-based analysis. Some modification is required if we aren't.
    (it, (train_idx, test_idx)) = indices
    n_train = len(train_idx)
    n_test = len(test_idx)
    if verbose: print(it, n_train, n_test)

    if isinstance(Y, dict):
        ds_names = Y.keys()
        Y = list(Y.values())
        X = list(X.values())

    Y_train = [Y_m.iloc[train_idx] for Y_m in Y]
    Y_test = [Y_m.iloc[test_idx] for Y_m in Y]
    if center | scale:
        scalers = [preprocessing.StandardScaler(with_mean=center, with_std=scale)
                   for _ in Y_train]
        scalers = [scaler.fit(Y_train_m)
                   for scaler, Y_train_m in zip(scalers, Y_train)]
        Y_train = [pd.DataFrame(scaler.transform(Y_tr_m),
                                index=Y_tr_m.index, columns=Y_tr_m.columns)
                   for scaler, Y_tr_m in zip(scalers, Y_train)]
        Y_test = [pd.DataFrame(scaler.transform(Y_te_m),
                                index=Y_te_m.index, columns=Y_te_m.columns)
                   for scaler, Y_te_m in zip(scalers, Y_test)]

    cv_res = fit(Y_train, n_pcs=n_pcs, d=d, k=k,
                 center=True, scale=True,
                 init=init, maxit=maxit,
                 delta=delta, device=device,
                 rcond=rcond, result_space = 'pc', verbose=verbose)
    Z_tr = torch.from_numpy(cv_res.Z.values)
    X_tr = [torch.from_numpy(X_m_tr.values) for X_m_tr in cv_res.X]

    # The held out data needs to be projected into the PC space learned
    #   from the training data.
    # TODO(brielin): does this work with non-informative analysis?
    Y_test_pcs = [torch.from_numpy(Y_m.values) @ pca_m.V
                  for Y_m, pca_m in zip(Y_test, cv_res.data_pcs)]
    W_tens = [torch.from_numpy(W_m.values) for W_m in cv_res.W]
    L_tens = [torch.from_numpy(L_m.values) for L_m in cv_res.L]
    Phi_tens = [torch.from_numpy(Phi_m.values) for Phi_m in cv_res.Phi]
    Z_te, X_te = _em.get_latent(W_tens, L_tens, Phi_tens, torch.cat(Y_test_pcs, axis=1),
                               device, rcond)
    Y_test_hat = [(Z_te @ W_m.T @ pca_m.V.T + X_m_te @ L_m.T @ pca_m.V.T).numpy()
                  for X_m_te, L_m, W_m, pca_m in zip(X_te, L_tens, W_tens, cv_res.data_pcs)]

    # Y_train = [torch.from_numpy(pca_m.pcs.values) for pca_m in cv_res.data_pcs]
    Y_train_pcs = [torch.from_numpy(Y_m.values) @ pca_m.V
                   for Y_m, pca_m in zip(Y_train, cv_res.data_pcs)]
    Y_train_hat = [(Z_tr @ W_m.T @ pca_m.V.T + X_m_tr @ L_m.T  @ pca_m.V.T).numpy()
                   for X_m_tr, L_m, W_m, pca_m in zip(X_tr, L_tens, W_tens, cv_res.data_pcs)]

    nrmse_tr = [np.sqrt(((Y_m - Y_m_hat)**2/Y_m.var(0)).values.mean())
                for Y_m, Y_m_hat in zip(Y_train, Y_train_hat)]
    nrmse_te = [np.sqrt(((Y_m - Y_m_hat)**2/Y_m_tr.var(0)).values.mean())
                for Y_m, Y_m_hat, Y_m_tr in zip(Y_test, Y_test_hat, Y_train)]

    if verbose:
        print(nrmse_tr, nrmse_te)

    # The CV res Z and X signs might not be aligned with the original.
    Z_signs = np.sign(np.corrcoef(Z.iloc[train_idx].T, cv_res.Z.T).diagonal(Z.shape[1]))
    X_signs = [np.sign(np.corrcoef(X_m.iloc[train_idx].T, X_m_cv.T).diagonal(X_m.shape[1]))
               for X_m, X_m_cv in zip(X, cv_res.X)]
    Z_te = Z_te * Z_signs
    X_te = [X_m_te * X_m_signs for X_m_te, X_m_signs in zip(X_te, X_signs)]
    return Z_te, X_te, nrmse_tr, nrmse_te
