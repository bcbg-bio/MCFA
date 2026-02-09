# pylint: disable=invalid-name

import torch
import numpy as np
import sys
from pathlib import Path
import warnings
from datetime import datetime
import testdata

sys.path.insert(0, str(Path('../src').resolve()))
from multicor_fa._em import fit_EM_iter

warnings.filterwarnings('ignore')
torch.set_default_dtype(torch.float64)

torch.set_printoptions(precision=2, sci_mode=False)

np.random.seed(0)
torch.manual_seed(0)

def log_likelihood(y:torch.tensor, w:torch.tensor, l:torch.tensor,
                   psi:torch.tensor, obs_mask:torch.tensor, n:int):
    """Computes negative log-likelihood:
       (1/2) * (|p_o|*log(2*pi) + log|Sigma_o| + tr(Sigma_o_inv@Sigma_hat)
    
    Args:
      y: Tensor. p features x n samples dataset. 
      w: Tensor. p features x d shared factors.
      l: Tensor. p features x sum(k1...km) private factors. 
      psi: Tensor. p features x p features covariance matrix. 
      obs_mask: Tensor. p features x n samples binary matrix where 1 represents 
        a feature i is observed for a sample j, and 0 represents unobserved. 
      n: Integer. Number of samples.
    Returns: 
      Negative log-likelihood with current model parameters
    """
    ll = 0
    sigma = w@w.T + l@l.T + psi
    for i in range(n):
        obs_idx = torch.where(obs_mask[:,i])[0]
        sigma_o = sigma[obs_idx][:, obs_idx]
        p_oi = torch.count_nonzero(obs_mask[:,i]).item()
        y_oi_reshaped = y[obs_idx, i].reshape(-1,1)
        sigma_hat = (y_oi_reshaped @ y_oi_reshaped.T).to(sigma_o.dtype)
        ll_new = (1/2) * (p_oi*np.log(2*np.pi) + torch.logdet(sigma_o)
                + torch.trace(torch.linalg.lstsq(sigma_o, sigma_hat).solution))
        if torch.isnan(ll_new):
            print(f"NaN triggered at sample {i}")
            print(f"Observed indices: {obs_idx}")
            print(f"W[6]: {w[6]}")
            print(f"L[6]: {l[6]}")
            print(f"Sigma_o: {sigma_o.round(decimals=2)}")
            print(ll_new)
            break
        ll += ll_new
    return ll


def test_multiple(params:dict, n_iters:int=1000, verbose:bool=True,
                  delta:float=1e-6, true_init:bool=False, ll_exact:bool=False):

    print(f'\nData parameters: n={params["n"]}; p={params["p"]}')

    _, Y_true, W_true, L_true, Psi_true = testdata.simulate_data(
        params, private_var=True
    )
    p1 = params['p'][0]
    p2 = p1 + params['p'][1]
    n_samps = params['n']

    # Initialize parameters
    if true_init:
        print('  True parameter value initialization')
        W_init, L_init, Psi_init = W_true, L_true, Psi_true
    else:
        print('  Jittered true parameter value initialization')
        W_init = [(W_i * (1 + 0.5 * torch.randn_like(W_i))) for W_i in W_true]
        L_init = [(L_i * (1 + 0.5 * torch.randn_like(L_i))) for L_i in L_true]
        Psi_init = [(P_i * (1 + 0.5 * torch.randn_like(P_i.float())))
                    for P_i in Psi_true]
    Sigma_hat = Y_true @ Y_true.T / n_samps

    # Ground truth values
    WW = (torch.cat(W_true, dim=0)) @ (torch.cat(W_true, dim=0).T)
    LL = (torch.block_diag(*L_true)) @ (torch.block_diag(*L_true).T)
    P = torch.block_diag(*Psi_true)
    Sigma_true = WW + LL + P

    Y = Y_true.detach().clone()

    # Initial "true" LL
    loglik_init = log_likelihood(Y, torch.cat(W_true,0),
                                    torch.block_diag(*L_true),
                                    torch.block_diag(*Psi_true),
                                    (~torch.isnan(Y)), n_samps)
    print(f'  "True" Log Likelihood: {loglik_init.item()}\n')

    # Full LL
    W_full, L_full, Psi_full, _, _ = fit_EM_iter(
        Y.T, Sigma_hat, W_init, L_init, Psi_init, maxit=n_iters,
        delta=delta, verbose=verbose, impute='none', ll_exact=ll_exact
    )
    loglik_full = log_likelihood(Y, torch.cat(W_full,0),
                                torch.block_diag(*L_full),
                                torch.block_diag(*Psi_full),
                                (~torch.isnan(Y)), n_samps)
    print(f'  Full data Log Likelihood: {round(loglik_full.item(), 4)}')

    # Insert missingness
    for missing_pattern in range(5):
        if missing_pattern == 0:
            print('\nNo missing data')
            obs_mask = torch.ones_like(Y)
        else:
            print(f'\nMissing pattern {missing_pattern}')
            if missing_pattern == 1:
                Y[:p1, int((0.2*n_samps)):int((0.3*n_samps))] = float('nan')
                Y[p1:p2, int((0.35*n_samps)):int((0.5*n_samps))] = float('nan')
                Y[p2:, int((0.55*n_samps)):int((0.75*n_samps))] = float('nan')
            elif missing_pattern == 2:
                Y[:p1, int((0.1*n_samps)):int((0.35*n_samps))] = float('nan')
                Y[p1:p2, int((0.4*n_samps)):int((0.65*n_samps))] = float('nan')
                Y[p2:, int((0.7*n_samps)):int((0.95*n_samps))] = float('nan')
            elif missing_pattern == 3:
                Y[:p1, :int((0.33*n_samps))] = float('nan')
                Y[p1:p2, int((0.33*n_samps)):int((0.67*n_samps))] = float('nan')
                Y[p2:, int((0.67*n_samps)):] = float('nan')
            elif missing_pattern == 4:
                Y[:p1, :int((0.4*n_samps))] = float('nan')
                Y[p1:p2, int((0.3*n_samps)):int((0.7*n_samps))] = float('nan')
                Y[p2:, int((0.6*n_samps)):] = float('nan')
            obs_mask = ~torch.isnan(Y)

        # Zero imputation
        print('  MCFA with zero-filled data')
        Y_mean = Y.detach().clone().nan_to_num(0)
        zero_start = datetime.now()
        W_mean_final, L_mean_final, Psi_mean_final, _, _ = fit_EM_iter(
            Y_mean.T, Sigma_hat, W_init, L_init, Psi_init, maxit=n_iters,
            delta=delta, verbose=verbose, impute='none', ll_exact=ll_exact
        )
        WW_mean_final=torch.cat(W_mean_final,0)@torch.cat(W_mean_final,0).T
        LL_mean_final = torch.block_diag(*L_mean_final) \
            @ (torch.block_diag(*L_mean_final)).T
        Psi_mean_final = torch.block_diag(*Psi_mean_final)
        Sigma_mean_final = WW_mean_final + LL_mean_final + Psi_mean_final
        # Correlations
        WW_mean_corr = torch.corrcoef(
            torch.stack([WW.flatten(), WW_mean_final.flatten()], dim=0)
        )[0,1].item()
        LL_mean_corr = torch.corrcoef(
            torch.stack([LL.flatten(), LL_mean_final.flatten()], dim=0)
        )[0,1].item()
        Psi_mean_corr = torch.corrcoef(torch.stack([
            P[P.nonzero()].flatten(),
            Psi_mean_final[Psi_mean_final.nonzero()].flatten()
        ], dim=0))[0,1].item()
        Sigma_mean_corr = torch.corrcoef(torch.stack([
            Sigma_true.flatten(), Sigma_mean_final.flatten()
        ]))[0,1].item()
        loglik_mean = log_likelihood(Y_mean, torch.cat(W_mean_final,0),
                                        torch.block_diag(*L_mean_final),
                                        Psi_mean_final, obs_mask, n_samps)
        # Print results
        print(f'\tWW^T corr: {round(WW_mean_corr, 4)}')
        print(f'\tLL^T corr: {round(LL_mean_corr, 4)}')
        print(f'\tPsi corr: {round(Psi_mean_corr, 4)}')
        print(f'\tSigma corr: {round(Sigma_mean_corr, 4)}')
        print(f'  Final LL: {round(loglik_mean.item(), 4)}')
        # Compute runtime
        zero_end = datetime.now()
        print(f'  Time: {str(np.mean(zero_end - zero_start)) \
                            .split('.', maxsplit=1)[0]}\n')

        # Hybrid MCFA imputation
        print('  MCFA with approx imputation')
        hyb_start = datetime.now()
        W_hyb_final, L_hyb_final, Psi_hyb_final, _, _ = fit_EM_iter(
            Y.T, Sigma_hat, W_init, L_init, Psi_init, maxit=n_iters,
            delta=delta, verbose=verbose, impute='approx', ll_exact=ll_exact
        )
        WW_hyb_final = torch.cat(W_hyb_final,0) \
            @ torch.cat(W_hyb_final,0).T
        LL_hyb_final = torch.block_diag(*L_hyb_final) \
            @ (torch.block_diag(*L_hyb_final)).T
        Psi_hyb_final = torch.block_diag(*Psi_hyb_final)
        Sigma_hyb_final = WW_hyb_final + LL_hyb_final + Psi_hyb_final
        # Correlations
        WW_hyb_corr = torch.corrcoef(
            torch.stack([WW.flatten(), WW_hyb_final.flatten()], dim=0)
        )[0,1].item()
        LL_hyb_corr = torch.corrcoef(
            torch.stack([LL.flatten(), LL_hyb_final.flatten()], dim=0)
        )[0,1].item()
        Psi_hyb_corr = torch.corrcoef(torch.stack([
            P[P.nonzero()].flatten(),
            Psi_hyb_final[Psi_hyb_final.nonzero()].flatten()
        ], dim=0))[0,1].item()
        Sigma_hyb_corr = torch.corrcoef(torch.stack([
            Sigma_true.flatten(), Sigma_hyb_final.flatten()
        ]))[0,1].item()
        loglik_hyb = log_likelihood(Y, torch.cat(W_hyb_final,0),
                                        torch.block_diag(*L_hyb_final),
                                        Psi_hyb_final, obs_mask, n_samps)
        # Print results
        print(f'\tWW^T corr: {round(WW_hyb_corr, 4)}')
        print(f'\tLL^T corr: {round(LL_hyb_corr, 4)}')
        print(f'\tPsi corr: {round(Psi_hyb_corr, 4)}')
        print(f'\tSigma corr: {round(Sigma_hyb_corr, 4)}')
        print(f'  Final LL: {round(loglik_hyb.item(), 4)}')
        # Compute runtime
        hyb_end = datetime.now()
        print(f'Time: {str(np.mean(hyb_end - hyb_start))\
                            .split('.', maxsplit=1)[0]}\n')

        # Exact imputation
        print('  MCFA with exact imputation')
        loop_start = datetime.now()
        hyb_start = datetime.now()
        W_loop_final, L_loop_final, Psi_loop_final, _, _ = fit_EM_iter(
            Y.T, Sigma_hat, W_init, L_init, Psi_init, maxit=n_iters,
            delta=delta, verbose=verbose, impute='exact', ll_exact=ll_exact
        )
        WW_loop_final = torch.cat(W_loop_final,0) \
            @ torch.cat(W_loop_final,0).T
        LL_loop_final = torch.block_diag(*L_loop_final) \
            @ (torch.block_diag(*L_loop_final)).T
        Psi_loop_final = torch.block_diag(*Psi_loop_final)
        Sigma_loop_final = WW_loop_final + LL_loop_final + Psi_loop_final
        # Correlations
        WW_loop_corr = torch.corrcoef(
            torch.stack([WW.flatten(), WW_loop_final.flatten()], dim=0)
        )[0,1].item()
        LL_loop_corr = torch.corrcoef(
            torch.stack([LL.flatten(), LL_loop_final.flatten()], dim=0)
        )[0,1].item()
        Psi_loop_corr = torch.corrcoef(torch.stack([
            P[P.nonzero()].flatten(),
            Psi_loop_final[Psi_loop_final.nonzero()].flatten()
        ], dim=0))[0,1].item()
        Sigma_loop_corr = torch.corrcoef(torch.stack([
            Sigma_true.flatten(), Sigma_loop_final.flatten()
        ]))[0,1].item()
        loglik_loop = log_likelihood(Y, torch.cat(W_loop_final,0),
                                        torch.block_diag(*L_loop_final),
                                        Psi_loop_final, obs_mask, n_samps)
        # Print results
        print(f'\tWW^T corr: {round(WW_loop_corr, 4)}')
        print(f'\tLL^T corr: {round(LL_loop_corr, 4)}')
        print(f'\tPsi corr: {round(Psi_loop_corr, 4)}')
        print(f'\tSigma corr: {round(Sigma_loop_corr, 4)}')
        print(f'  Final LL: {round(loglik_loop.item(), 4)}')
        # Compute runtime
        loop_end = datetime.now()
        print(f'Time: {str(np.mean(loop_end - loop_start))\
                            .split('.', maxsplit=1)[0]}\n')

# Reduce p and increase n
parameters = {
    'd': 3,
    'k': [3, 4, 5], 
    'n': 1000,
    'p': [6, 7, 8],
    'sigsq': [1, 1, 1]
}

test_multiple(
    parameters, n_iters=1000, verbose=False, true_init=False, ll_exact=True
)

########## Output 01/29/2026 with seed 0 ##########
# Data parameters: n=1000; p=[6, 7, 8]
#   Jittered true parameter value initialization
#   "True" Log Likelihood: 37862.60202215713

#   Full data Log Likelihood: 37780.9925

# No missing data
#   MCFA with zero-filled data
#         WW^T corr: 0.9935
#         LL^T corr: 0.9799
#         Psi corr: 0.9748
#         Sigma corr: 0.9954
#   Final LL: 37780.9925
#   Time: 0:00:03

#   MCFA with approx imputation
#         WW^T corr: 0.9935
#         LL^T corr: 0.9799
#         Psi corr: 0.9748
#         Sigma corr: 0.9954
#   Final LL: 37780.9925
#         Time: 0:00:03

#   MCFA with exact imputation
#         WW^T corr: 0.9935
#         LL^T corr: 0.9799
#         Psi corr: 0.9748
#         Sigma corr: 0.9954
#   Final LL: 37780.9925
#   Time: 0:00:03


# Missing pattern 1
#   MCFA with zero-filled data
#         WW^T corr: 0.9881
#         LL^T corr: 0.9313
#         Psi corr: 0.9523
#         Sigma corr: 0.9924
#   Final LL: 32197.9932
#   Time: 0:00:04

#   MCFA with approx imputation
#         WW^T corr: 0.9883
#         LL^T corr: 0.9442
#         Psi corr: 0.9501
#         Sigma corr: 0.9743
#   Final LL: 32374.4417
#   Time: 0:00:00

#   MCFA with exact imputation
#         WW^T corr: 0.9938
#         LL^T corr: 0.9783
#         Psi corr: 0.9747
#         Sigma corr: 0.995
#   Final LL: 32020.0999
#   Time: 0:00:17


# Missing pattern 2
#   MCFA with zero-filled data
#         WW^T corr: 0.9413
#         LL^T corr: 0.8372
#         Psi corr: 0.9418
#         Sigma corr: 0.9743
#   Final LL: 26532.0045
#   Time: 0:00:03

#   MCFA with approx imputation
#         WW^T corr: 0.9763
#         LL^T corr: 0.9066
#         Psi corr: 0.928
#         Sigma corr: 0.9358
#   Final LL: 26393.5891
#   Time: 0:00:00

#   MCFA with exact imputation
#         WW^T corr: 0.9909
#         LL^T corr: 0.9701
#         Psi corr: 0.9658
#         Sigma corr: 0.9933
#   Final LL: 25807.5608
#   Time: 0:00:15


# Missing pattern 3
#   MCFA with zero-filled data
#         WW^T corr: 0.8743
#         LL^T corr: 0.782
#         Psi corr: 0.9341
#         Sigma corr: 0.9544
#   Final LL: 24625.8761
#   Time: 0:00:05

#   MCFA with approx imputation
#         WW^T corr: 0.9679
#         LL^T corr: 0.8882
#         Psi corr: 0.9204
#         Sigma corr: 0.9185
#   Final LL: 24216.6184
#   Time: 0:00:00

#   MCFA with exact imputation
#         WW^T corr: 0.9876
#         LL^T corr: 0.9628
#         Psi corr: 0.9627
#         Sigma corr: 0.9921
#   Final LL: 23597.6016
#   Time: 0:00:19


# Missing pattern 4
#   MCFA with zero-filled data
#         WW^T corr: 0.8566
#         LL^T corr: 0.7543
#         Psi corr: 0.914
#         Sigma corr: 0.9514
#   Final LL: 23517.2799
#   Time: 0:00:06

#   MCFA with approx imputation
#         WW^T corr: 0.961
#         LL^T corr: 0.8614
#         Psi corr: 0.9164
#         Sigma corr: 0.9055
#   Final LL: 22980.5733
#   Time: 0:00:01

#   MCFA with exact imputation
#         WW^T corr: 0.9881
#         LL^T corr: 0.9675
#         Psi corr: 0.9586
#         Sigma corr: 0.9919
#   Final LL: 22346.1248
#   Time: 0:00:22
