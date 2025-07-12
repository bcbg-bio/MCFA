# pylint: disable=invalid-name
"""PCA routines for MCFA model.

Do not call directly, see mcfa.py for usage.
"""

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from sklearn import preprocessing


@dataclass
class PCARes:
    """Simple dataclass for storing PCA results."""
    pcs: pd.DataFrame
    var_exp: pd.Series
    U: torch.Tensor
    S: torch.Tensor
    V: torch.Tensor
    k: int
    n: int
    d: int
    mp_dim: int

def _ppca_missing(X: torch.Tensor, k: int = 'infer') -> PCARes:
    """Implementation of constrained factor analysis with missing data.

    Args:
        X: n (samples) by d (features) torch.Tensor.
        k: Integer, 'infer' or 'all'. Number of pcs to keep. Default is to
          infer using the marchenko pasteur cutoff.
    Returns:
        A PCARes instance.
    """
    raise NotImplementedError('TBD.')
    

def _pca(X: torch.Tensor, k: int = 'infer', calc_V = True) -> PCARes:
    """Basic PCA implementation.

    This is a basic PCA implementation which is particularly efficient for
    top-k PCA by doing the eigendecomposition of either X X.T / N or
    X.T X / N.

    Args:
        X: n (samples) by d (features) torch.Tensor.
        k: Integer, 'infer' or 'all'. Number of pcs to keep. Default is to
          infer using the marchenko pasteur cutoff.
        calc_V: True to track the PC loadings (right singular vectors)
          of X. Setting to False can save substantial memory if X is very
          wide.
    Returns:
        A PCARes instance.
    """
    N, D = X.shape
    mp_lower_bound = 1 + np.sqrt(D / N)

    if D > N:
        adjustment = D/((~torch.isnan(X)).type(torch.int64) @ \
            (~torch.isnan(X)).type(torch.int64).T)
        gram_mat = adjustment*(torch.nan_to_num(X) @ torch.nan_to_num(X).T)/N
    else:
        adjustment = N/((~torch.isnan(X)).type(torch.int64).T @ \
            (~torch.isnan(X)).type(torch.int64))
        gram_mat = adjustment*(torch.nan_to_num(X).T @ torch.nan_to_num(X))/N

    vals, vecs = torch.linalg.eigh(gram_mat)
    S = torch.sqrt(torch.abs(torch.flip(vals, dims = [0])))
    A = torch.flip(vecs, dims=[1])
    mp_dim = sum(S > mp_lower_bound)

    # TODO(brielin): Throw error if infer and not center/scale.
    if k in ('infer', 'all'):
        k = mp_dim if k == 'infer' else min(N, D)
    S_k = S[0:k]

    if D > N:
        U = A[:, 0:k]
        V = torch.eye(k, dtype=torch.double)
        if calc_V:
            V = X.T @ U / (S_k * np.sqrt(N))
    else:
        V = A[:, 0:k]
        U = X @ V / (S_k * np.sqrt(N))
        V = V if calc_V else torch.eye(k, dtype=torch.double)
    var_exp = S_k**2 / torch.sum(S**2)

    return U, V, S_k, var_exp, k, N, D, mp_dim


def pca(X: pd.DataFrame, k: int = 'infer', center: bool = True,
        scale: bool = True, calc_V: bool = True, missing: str = 'raise') -> PCARes:
    """Handler for PCA functions with potentially missing data

    This is a dispatch function for the above routines depending on the presence
    or absence of missing data and instructions for handling it.

    Args:
        X: n (samples) by d (features) pandas DataFrame.
        k: Integer, 'infer' or 'all'. Number of pcs to keep. Default is to
          infer using the marchenko pasteur cutoff.
        center: bool. True to mean-center columns of X.
        scale: bool. True to variance-scale the columns of X to 1.0.
        calc_V: True to track the PC loadings (right singular vectors)
          of X. Setting to False can save substantial memory if X is very
          wide. Ignored if missing=='impute'.
        missing: String, one of 'raise', 'ignore' or 'impute'
    Returns:
        A PCARes instance.
    """
    # Some entire rows may be 0 if a modality is missing. Need to drop
    # these and add them back later.
    all_samples = X.index
    drop_index = np.asarray(pd.isna(X).all(axis=1)).nonzero()[0]
    if len(drop_index) > 0:
        X.drop(X.index[drop_index], inplace=True)

    if (missing == 'raise') & pd.isna(X).any(axis = None):
        raise ValueError('Missing data not expected in PCA input.')

    if center | scale:
        X = torch.from_numpy(preprocessing.scale(
            X, with_mean=center, with_std=scale))
    else:
        X = torch.from_numpy(X.values)

    pcares = None
    if missing == 'raise' or missing == 'ignore':
        U, V, S_k, var_exp, k, N, D, mp_dim = _pca(X, k, calc_V)
    elif missing == 'impute':
        U, V, S_k, var_exp, k, N, D, mp_dim = _ppca_missing(X, k)

    if len(drop_index) > 0:
        for i in drop_index:
            new_row = torch.empty((1, k))
            new_row[:,:] = float('nan')
            if drop_index == 0:
                U = torch.cat((new_row, U), 0)
            elif drop_index < len(pcares.U):
                U = torch.cat((U[0:i,:], new_row, U[i:,:]), 0)
            else:
                U = torch.cat((U, new_row), 0)

    pc_names = ['PC' + str(i+1) for i in range(k)]
    pcs = pd.DataFrame((U * S_k * np.sqrt(N)).numpy(), index=all_samples,
                       columns=pc_names)
    var_exp = pd.Series(var_exp, index=pc_names)

    return PCARes(pcs, var_exp, U, S_k, V, k, N, D, mp_dim)
