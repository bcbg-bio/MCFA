# pylint: disable=invalid-name
"""Expectation Maximization routines for MCFA model.

Do not call directly, see mcfa_model.py for usage.
"""

import time
import torch
import numpy as np
from typing import List


def _EM_step_no_private_stable(
        W: torch.Tensor, Phi: torch.Tensor, Y: torch.Tensor,
        Y_imp: None | torch.Tensor, p: List[int], device = 'cpu',
        rcond: float = 1e-08, impute: bool = False):
    """One EM step when there is no private structure.

    Args:
      W: p_all=sum(p) by d tensor. Current estimate of W.
      Phi: p_all by p_all tensor. Current estimate of Phi.
      Y: N by p_all torch tensor, the data.
      Y_imp: None or N by p_all torch tensor. If not None, then represents
        Y with imputed missing values filled in when impute=True.
      p: List of integers with the dimensionality of each dataset.
      rcond: Condition number for least squares.
      impute: Whether to impute missing data modes. 
    Returns:
      Tuple W_next, Phi_next with updated parameters.
    """
    # TODO(brielin): use device
    if device == 'gpu': raise NotImplementedError()
    d = W.shape[1]
    N, p_all = Y.shape
    missing_Y = torch.isnan(Y).any()

    if missing_Y:
        if not impute:
            raise ValueError('Y contains missing values but impute is False')
        else:
            # Observed and missing components
            obs_mask = ~torch.isnan(Y) + 0.0
            miss_mask = torch.isnan(Y) + 0.0
            # Number of samples missing per feature
            N_m = miss_mask.count_nonzero(axis=0).reshape((-1,1)).to(Y.dtype)
            if Y_imp is None:
                # Replace NA with zeros
                Y = torch.nan_to_num(Y, 0)
            else:
                # Replace NA with imputed values from previous iteration
                Y = torch.where(obs_mask, Y, Y_imp)

    Q_inv_W = torch.linalg.lstsq(W @ W.T + Phi, W, rcond=rcond).solution
    E_z = Y @ Q_inv_W # N x d
    sum_E_zzT = N*torch.eye(d) - N*(W.T @ Q_inv_W) + E_z.T @ E_z
    S22 = W @ W.T + Phi

    if missing_Y and impute:
        E_y = (W @ W.T) @ (torch.linalg.lstsq(S22, Y.T, rcond=rcond).solution)
        E_y_mask = (E_y * miss_mask.T) + 0.0
        Y_mask = (Y * obs_mask) + 0.0

        E_yzT = N_m*(W - ((W @ W.T) @ Q_inv_W)) + (E_y_mask @ E_z)
        E_yyT = N_m*(S22 - (W @ W.T) \
                @ (torch.linalg.lstsq(S22, W @ W.T, rcond=rcond).solution)) \
                + (E_y_mask @ E_y_mask.T)

        sum_E_yzT = torch.zeros((p_all, d)).double()
        sum_E_yzT += Y_mask.T @ E_z
        sum_E_yzT += E_yzT

        sum_E_yyT = torch.zeros((p_all, p_all))
        sum_E_yyT += Y_mask.T @ Y_mask
        sum_E_yyT += E_yyT
    else:
        sum_E_yzT = Y.T @ E_z
        sum_E_yyT = Y.T @ Y

    W_next = torch.linalg.lstsq(sum_E_zzT, sum_E_yzT.T, rcond=rcond).solution.T
    # Phi_next = Sigma - (Y.T @ E_z @ W_next.T)/N
    Phi_next = (sum_E_yyT - (W_next @ sum_E_yzT.T))/N
    psum = np.concatenate([[0], np.cumsum(p, 0)])
    for i_l, i_r in zip(psum[:-1], psum[1:]):
        Phi_next[i_l:i_r, i_r:] = 0
        Phi_next[i_r:, i_l:i_r] = 0
    if missing_Y and impute:
        # Replace NAs with imputed Y
        # This will still be set to None in fit_EM_iter if imp_y==False
        Y_next = torch.where(obs_mask, Y, E_y.T)
    else:
        Y_next = None
    return W_next, Phi_next, Y_next


def _EM_step_full_stable(W: torch.Tensor, L: torch.Tensor, Phi: torch.Tensor,
                         Y: torch.Tensor, Y_imp: None | torch.Tensor, p, k,
                         device='cpu', rcond: float = 1e-08,
                         impute: bool = False):
    """One EM step for the full model.

    Args:
      W: p_all=sum(p) by d tensor. Current estimate of W.
      L: p_all by k_all=sum(k) block diagonal tensor. Current estimate of L.
      Phi: p_all by p_all tensor. Current estimate of Phi.
      Y: N by p_all torch tensor, the data.
      Y_imp: N by p_all torch tensor or None. If not None, then represents
        Y with imputed missing values filled in when impute=True.
      p: List of integers with the dimensionality of each dataset.
      k: List of integers with the private dimensionality of each dataset.
      device: 'cpu' or 'gpu'.
      rcond: Condition number for least squares.
      impute: Whether to impute missing data. 
    Returns:
      Tuple W_next, L_next, Phi_next
    """
    N, p_all = Y.shape
    d = W.shape[1]
    k_all = L.shape[1]
    psum = np.concatenate([[0], np.cumsum(p, 0)])
    ksum = np.concatenate([[0], np.cumsum(k, 0)])
    missing_Y = torch.isnan(Y).any()

    if missing_Y:
        if not impute:
            raise ValueError('Y contains missing values but impute is False')
        else:
            # Observed and missing components
            obs_mask = ~torch.isnan(Y) + 0.0
            miss_mask = torch.isnan(Y) + 0.0
            # Number of samples missing per feature
            N_m = miss_mask.count_nonzero(axis=0).reshape((-1,1)).to(Y.dtype)
            if Y_imp is None:
                # Replace NA with zeros
                Y = torch.nan_to_num(Y, 0)
            else:
                # Replace NA with imputed values from previous iteration
                Y = torch.where(obs_mask, Y, Y_imp)

    S21 = torch.cat([W, L], axis=1).to(device)
    S22 = W @ W.T + L @ L.T + Phi
    S22inv_S21 = torch.linalg.lstsq(S22, S21, rcond=rcond).solution
    # TODO(brielin): transpose the next line.
    E_z_x = S22inv_S21.T @ Y.T
    mom_zx_zxT_sum = (N * torch.eye(k_all + d).to(device) - \
                      (N * S22inv_S21.T @ S21).to(device) + \
                      E_z_x @ E_z_x.T).to(device)
    zz_sum = mom_zx_zxT_sum[:d, :d].to(device)
    zx_sum = mom_zx_zxT_sum[:d, d:].to(device)
    xx_sum = mom_zx_zxT_sum[d:, d:].to(device)

    if impute and missing_Y:
        E_y = (W @ W.T) \
            @ (torch.linalg.lstsq(S22, Y.T, rcond=rcond).solution)
        E_y_mask = (E_y * miss_mask.T) + 0.0
        Y_mask = (Y * obs_mask) + 0.0

        # Make this only over missing samples (sum over N_m)
        E_yzT = N_m*(W - (W @ W.T) @ S22inv_S21[:, :d]) \
            + (E_y_mask @ E_z_x[:d, :].T)
        E_yxT = N_m*(L - (W @ W.T) @ S22inv_S21[:, d:]) \
            + (E_y_mask @ E_z_x[d:, :].T)
        E_yyT = N_m*(S22 - (W @ W.T) \
            @ (torch.linalg.lstsq(S22, W @ W.T, rcond=rcond).solution)) \
            + (E_y_mask @ E_y_mask.T)

        sum_E_yzT = torch.zeros((p_all, d)).double()
        sum_E_yzT += Y_mask.T @ E_z_x[:d, :].T
        sum_E_yzT += E_yzT

        sum_E_yxT = torch.zeros((p_all, k_all)).double()
        sum_E_yxT += Y_mask.T @ E_z_x[d:, :].T
        sum_E_yxT += E_yxT

        sum_E_yyT = torch.zeros((p_all, p_all))
        sum_E_yyT += Y_mask.T @ Y_mask
        sum_E_yyT += E_yyT
    else:
        sum_E_yzT = Y.T @ E_z_x[:d, :].T
        sum_E_yxT = Y.T @ E_z_x[d:, :].T
        sum_E_yyT = Y.T @ Y

    W_next = torch.linalg.lstsq(zz_sum, (sum_E_yzT - L @ zx_sum.T).T,
                                rcond=rcond).solution.T
    L_next = torch.linalg.lstsq(xx_sum, (sum_E_yxT - W @ zx_sum).T,
                                rcond=rcond).solution.T
    L_next = [L_next[i:j, l:m]
              for i, j, l, m in zip(psum[:-1], psum[1:], ksum[:-1], ksum[1:])]
    L_next = torch.block_diag(*L_next)
    Phi_next = sum_E_yyT + \
        W_next @ zz_sum @ W_next.T + \
        L_next @ xx_sum @ L_next.T + \
        2 * L_next @ zx_sum.T @ W_next.T - \
        2 * sum_E_yzT @ W_next.T - \
        2 * sum_E_yxT @ L_next.T
    Phi_next = torch.diag(torch.diagonal(Phi_next / N)).to(device)
    if missing_Y and impute:
        # Replace NAs with imputed Y
        # This will still be set to None in fit_EM_iter if imp_y==False
        Y_next = torch.where(obs_mask, Y, E_y.T)
    else:
        Y_next = None
    return W_next, L_next, Phi_next, Y_next


def _get_latent_worker(W: torch.Tensor, L: torch.Tensor, Phi: torch.Tensor,
                       Y: torch.Tensor, device='cpu', rcond: float = 1e-08):
    if L is not None:
        S21 = torch.cat([W, L], axis=1).to(device)
        S22 = W @ W.T + L @ L.T + Phi
    else:
        S21 = W
        S22 = W @ W.T + Phi
    S22inv_S21 = torch.linalg.lstsq(S22, S21, rcond=rcond).solution
    E_z_x = Y @ S22inv_S21
    return E_z_x


def calculate_rho(W: List[torch.Tensor], L: List[torch.Tensor],
                  Phi: List[torch.Tensor], Y: torch.Tensor, device: str ='cpu',
                  rcond: float = 1e-08, method: str = 'genvar'):
    d = W[1].shape[1]
    p = [W_m.shape[0] for W_m in W]
    psum = np.concatenate([[0], np.cumsum(p, 0)])
    Y = [Y[:, i:j] for i, j in zip(psum[:-1], psum[1:])]
    if L is not None:
        E_z_x_all = [_get_latent_worker(W_m, L_m, Phi_m, Y_m, device, rcond)
                     for W_m, L_m, Phi_m, Y_m in zip(W, L, Phi, Y)]
    else:
        E_z_x_all = [_get_latent_worker(W_m, None, Phi_m, Y_m, device, rcond)
                     for W_m, Phi_m, Y_m in zip(W, Phi, Y)]
    Z_all = [E_z_x[:, 0:d] for E_z_x in E_z_x_all]
    S_all = [torch.corrcoef(torch.stack([Z[:, d_m] for Z in Z_all], 1).T)
             for d_m in range(d)]
    if method == 'genvar':
        rho = [-torch.slogdet(S)[1] for S in S_all]
    elif method == 'sumcor':
        rho = [S.sum() for S in S_all]
    elif method == 'ssqcor':
        rho = [(S**2).sum() for S in S_all]
    rho = torch.stack(rho)
    return rho


def get_latent(W: List[torch.Tensor], L: List[torch.Tensor],
               Phi: List[torch.Tensor], Y: torch.Tensor,
               device='cpu', rcond: float = 1e-08):
    """Gets latent Z, X for the model (the E-step).

     Args:
       W: p_all=sum(p) by d tensor. Current estimate of W.
       L: p_all by k_all=sum(k) block diagonal tensor. Current estimate of L.
       Phi: p_all by p_all tensor. Current estimate of Phi.
       Y: N by p_all torch tensor, the data.
       device: 'cpu' or 'gpu'.
       rcond: Condition number for least squares.
    Returns:
       Tuple of (tensor, List[tensor]): N x d (posterior Z),
         list of N x k_m (posterior X).
    """
    # TODO(brielin): Figure out a way to do this calculation efficiently
    #   without cat/block_diag on W, L, Phi
    k = None if L is None else [L_m.shape[1] for L_m in L]
    ksum = np.concatenate([[0], np.cumsum(k, 0)])
    W = torch.cat(W, 0)
    L = None if L is None else torch.block_diag(*L)
    Phi = torch.block_diag(*Phi)
    d = W.shape[1]
    E_z_x = _get_latent_worker(W, L, Phi, Y, device, rcond)
    Z = E_z_x[:, 0:d]
    X = None if L is None else [E_z_x[:, (d+i):(d+j)]
                                for i, j in zip(ksum[:-1], ksum[1:])]
    return Z, X


def _loglik(Sigma: torch.Tensor, Sigma_hat: torch.Tensor, n: int) -> float:
    p = Sigma.shape[0]
    l = (n/2)*(p*np.log(2*np.pi) +
               torch.logdet(Sigma) +
               torch.trace(torch.linalg.lstsq(Sigma, Sigma_hat).solution))
    return l


def fit_EM_iter(Y, Sigma_hat, W, L, Phi, maxit = 1000, device = 'cpu',
                 rcond = 1e-08, delta = 1e-6, verbose = False,
                 impute: bool = False, imp_y: bool = False):
    """Iteratively fit EM.

    Args:
       Y: N by p_all torch tensor, the data.
       Sigma_hat: Y.T @ Y / N
       W: p_all=sum(p) by d tensor. Current estimate of W.
       L: p_all by k_all=sum(k) block diagonal tensor. Current estimate of L.
       Phi: p_all by p_all tensor. Current estimate of Phi.
       maxit: Maximum number of iterations to run.
       rcond: Tolerance for leastsquares.
       delta: Break when change in likelihood < delta.
       verbose: True to print progress.
       impute: Whether to impute missing data. 
       imp_y: Whether to impute using estimated Y each iteration.
    Returns:
       W: p_all=sum(p) by d tensor. New estimate of W.
       L: p_all by k_all=sum(k) block diagonal tensor. New estimate of L.
       Phi: p_all by p_all tensor. New estimate of Phi.
       l: List of current and new log-likelihood values. 
       cd: 
    """
    # TODO(brielin): Figure out a way to do this calculation efficiently
    #   without cat/block_diag on W, L, Phi
    p = [W_m.shape[0] for W_m in W]
    k = None if L is None else [L_m.shape[1] for L_m in L]
    psum = np.concatenate([[0], np.cumsum(p, 0)])
    ksum = None if L is None else np.concatenate([[0], np.cumsum(k, 0)])
    W = torch.cat(W, 0)
    L = None if L is None else torch.block_diag(*L)
    Phi = torch.block_diag(*Phi)

    N = Y.shape[0]
    Sigma = W @ W.T + Phi if L is None else W @ W.T + L @ L.T + Phi
    l = [_loglik(Sigma, Sigma_hat, N).tolist()]
    denom = 1 # torch.mean(Sigma_hat**2)
    cd = [torch.mean((Sigma - Sigma_hat)**2/denom).tolist()]

    L_it = None
    t0 = time.time()
    if verbose: print('iter:', 0, 'Likelihood:', l[0])
    Y_it = None # Initialize placeholder for imputed Y values
    for it in range(maxit):
        if L is None:
            W_it, Phi_it, Y_it = _EM_step_no_private_stable(
                W, Phi, Y, Y_it, p, device, rcond, impute)
            Sigma_it = W_it @ W_it.T + Phi_it
        else:
            W_it, L_it, Phi_it, Y_it = _EM_step_full_stable(
                W, L, Phi, Y, Y_it, p, k, device, rcond, impute)
            Sigma_it = W_it @ W_it.T + L_it @ L_it.T + Phi_it
        if not imp_y:
            Y_it = None # Do not use imputed Y for model imputation
        l_it = _loglik(Sigma_it, Sigma_hat, N).tolist()
        cd_it = torch.mean((Sigma_it - Sigma_hat)**2/denom).tolist()
        l_delta_it = (l[it] - l_it)/l_it
        cd_delta_it = cd[it] - cd_it
        l.append(l_it)
        cd.append(cd_it)
        W = W_it
        L = L_it
        Phi = Phi_it
        if verbose: print('Iter:', it+1, 'Likelihood:', l_it,
                    'Percent change:', l_delta_it, 'Time (s):', time.time()-t0)
        if delta is not None:
            if l_delta_it < delta: break
    W = [W[i:j, :] for i, j in zip(psum[:-1], psum[1:])]
    L = None if L is None else [
        L[i:j, l:m] for i, j, l, m in zip(
            psum[:-1], psum[1:], ksum[:-1], ksum[1:])]
    Phi = [Phi[i:j, i:j] for i, j in zip(psum[:-1], psum[1:])]
    return W, L, Phi, l, cd
