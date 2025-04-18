# pylint: disable=invalid-name
"""Initializers for MCFA model.

Do not call directly, see mcfa.py for usage.
"""

import torch
from typing import List

def _ppca(X, d):
    """Produces a facorization X = W @ W.T + s2*I for symmetric X.

    Args:
      X: symmetric p x p matrix.
      d: number of dimensions to keep.
    """
    vals, vecs = torch.linalg.eigh(X)
    vals = torch.abs(torch.flip(vals, dims = [0]))
    vecs = torch.flip(vecs, dims=[1])
    W = vecs[:, 0:d] * torch.sqrt(vals[0:d])
    s2 = torch.mean(vals[d:])
    return W, s2, vals


# TODO(brielin): Double check that
def _init_var_W(Y_pcs, psum, d, informative):
    """Initializes W using sumcor with avgvar constraint.

    Args:
      Y_pcs: List of PCARes objects.
      psum: List of break indices for inidivudal datasets.
      d: Number of components to keep.
      informative: True to keep W in PC spaces, False to return
        to original data space.
    """
    U_all = torch.cat([pc.U for pc in Y_pcs], dim = 1)
    UTU = U_all.T @ U_all

    W, _, vals = _ppca(UTU, d)
    W = [(W[i:j, :].T * pc.S).T for i, j, pc in zip(psum[:-1], psum[1:], Y_pcs)]
    if informative is False: W = [pc.V @ W for pc, W in zip(Y_pcs, W)]
    return W, vals


def _init_norm_W(Sigma_hat, psum, d, M):
    """Initializes W using sumcor with avgnorm constraint.

    Args:
      Sigma_hat: Cross correlation matrix to model.
      d: Number of components to keep.
      M: Number of datasets.
    """
    W, _, _ = _ppca(Sigma_hat, d)
    W = [W[i:j, :] for i, j in zip(psum[:-1], psum[1:])]
    rho = sum([(W_m**2).sum(0) for W_m in W])
    return W, rho


def _init_L_Phi(Sigma_hat, W, psum, p, k):
    """Initializes L and Phi for a given W, Sigma_hat.

    Args:
      Sigma_hat: Cross correlation matrix to model.
      W: LIST
      psum: List of break indices for inidivudal datasets.
      p: List of integers, dimensions of datasets.
      k: List of integers or None, dimensions of private spaces.
    """
    Phi = [Sigma_hat[i:j, i:j] - W_m @ W_m.T
           for W_m, i, j in zip(W, psum[:-1], psum[1:])]

    L = None
    if k is not None:
        resid_pcas = [_ppca(Phi_m, k_m) for k_m, Phi_m in zip(k, Phi)]
        L, s2s, _ = list(map(list, zip(*resid_pcas)))
        Phi = [torch.diag(torch.tensor([s2]*p_m)) for s2, p_m in zip(s2s, p)]
    return L, Phi


def _rho_mp_sim(N: int, p: List[int], nsims=100, device='cpu'):
    """Calculates the MCCA (Parra) solution to random data.

    Args:
      N: Integer. Sample size.
      p: List of integers. Dimensions of the datasets to simulate.
      nsims: Number of simulation iterations.
      device: Device to run on.
    """
    sim_res = []
    for _ in range(nsims):
        Y = [pd.DataFrame(np.random.normal(size=(N, p_m))) for p_m in p]
        Y_pcs = [pca(Y_m, 'all') for Y_m in Y]
        U_all = torch.cat([pc.U for pc in Y_pcs], dim = 1)
        UTU = U_all.T @ U_all
        rho = torch.linalg.eigvalsh(UTU)
        sim_res.append(torch.max(rho))
    sim_res = torch.Tensor(sim_res)
    return sim_res.mean(), np.sqrt(sim_res.var()/nsims)
