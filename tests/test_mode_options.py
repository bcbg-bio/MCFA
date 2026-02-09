# pylint: disable=invalid-name

import torch
import numpy as np
import math
import warnings
import pandas as pd
import testdata

import sys
from pathlib import Path

sys.path.insert(0, str(Path('../src').resolve()))
from multicor_fa.mcfa_model import fit

warnings.filterwarnings('ignore')
torch.set_default_dtype(torch.float64)

np.random.seed(0)
torch.manual_seed(0)

def pidx(mult, n):
    return int(math.floor(n*mult))

def log_likelihood(y:torch.tensor, w:torch.tensor, l:torch.tensor,
                   psi:torch.tensor, obs_mask:torch.tensor, n:int):
    ll = 0.0
    if l is None:
        sigma = w@w.T + psi
    else:
        sigma = w@w.T + l@l.T + psi
    for i in range(n):
        obs_idx = torch.where(obs_mask[i,:])[0]
        sigma_o = sigma[obs_idx][:, obs_idx]
        p_oi = torch.count_nonzero(obs_mask[i,:]).item()
        y_oi_reshaped = y[i, obs_idx].reshape(1,-1)
        sigma_hat = (y_oi_reshaped.T @ y_oi_reshaped).to(sigma_o.dtype)
        ll_new = (1/2) * (p_oi*np.log(2*np.pi) + torch.logdet(sigma_o) +
            torch.trace(torch.linalg.lstsq(sigma_o, sigma_hat).solution))
        if torch.isnan(ll_new):
            print(f'NaN triggered at sample {i}')
            print(f'Observed indices: {obs_idx}')
            print(f'W[6]: {w[6]}')
            if l is not None:
                print(f'L[6]: {l[6]}')
            print(f'Sigma_o: {sigma_o.round(decimals=2)}')
            print(ll_new)
            break
        ll += ll_new.item()
    return ll

def test_modes(parameters, patterns=5, n=1000, private_var=True,
               ll_exact=False):
    params = parameters.copy()
    mean_res_dict = {'WW^T': [], 'Phi': [], 'Sigma':[]}
    drop_res_dict = {'WW^T': [], 'Phi': [], 'Sigma':[]}
    approx_res_dict = {'WW^T': [], 'Phi': [], 'Sigma':[]}
    exact_res_dict = {'WW^T': [], 'Phi': [], 'Sigma':[]}
    if private_var:
        mean_res_dict['LL^T'] = []
        drop_res_dict['LL^T'] = []
        approx_res_dict['LL^T'] = []
        exact_res_dict['LL^T'] = []
    else:
        params['k'] = []
    params['n'] = n
    if private_var:
        Y_dict, _, _, _, _ = testdata.simulate_data(params, private_var=True)
        mcfa_res_full = fit(Y_dict, missing_modes='raise', n_pcs='all',
            d=params['d'], k=params['k'], verbose=False, ll_exact=ll_exact,
            center=False, scale=False)
        W_full_torch = torch.cat(
            [torch.from_numpy(val.values)
            for val in mcfa_res_full.W.values()], dim=0
        )
        L_full_torch = torch.block_diag(
            *[torch.from_numpy(val.values)
            for val in mcfa_res_full.L.values()]
        )
        Phi_full = torch.block_diag(
            *[torch.from_numpy(val.values)
            for val in mcfa_res_full.Phi.values()]
        )
    else:
        Y_dict, _, _, _, _ = testdata.simulate_data(params, private_var=False)
        mcfa_res_full = fit(Y_dict, missing_modes='raise', n_pcs='all',
            d=params['d'], k=[0,0,0], verbose=False, ll_exact=ll_exact,
            center=False, scale=False)
        W_full_torch = torch.cat(
            [torch.from_numpy(val.values)
            for val in mcfa_res_full.W.values()], dim=0
        )
        Phi_full = torch.block_diag(
            *[torch.from_numpy(val.values)
            for val in mcfa_res_full.Phi.values()]
        )

    # Add missingness
    for missing_pattern in range(patterns):
        if missing_pattern == 0:
            print('No missing data')
        else:
            print('Missing pattern', missing_pattern)
        Y_miss = Y_dict.copy()
        if missing_pattern == 1:
            Y_miss['mode0'] = Y_miss['mode0'].drop(
                index=Y_miss['mode0'].iloc[pidx(.2,n):pidx(.3,n)].index
            )
            Y_miss['mode1'] = Y_miss['mode1'].drop(
                index=Y_miss['mode1'].iloc[pidx(.35,n):pidx(.5,n)].index
            )
            Y_miss['mode2'] = Y_miss['mode2'].drop(
                index=Y_miss['mode2'].iloc[pidx(.55,n):pidx(.75,n)].index
            )
        elif missing_pattern == 2:
            Y_miss['mode0'] = Y_miss['mode0'].drop(
                index=Y_miss['mode0'].iloc[pidx(.1,n):pidx(.35,n)].index
            )
            Y_miss['mode1'] = Y_miss['mode1'].drop(
                index=Y_miss['mode1'].iloc[pidx(.4,n):pidx(.65,n)].index
            )
            Y_miss['mode2'] = Y_miss['mode2'].drop(
                index=Y_miss['mode2'].iloc[pidx(.7,n):pidx(.95,n)].index
            )
        elif missing_pattern == 3:
            Y_miss['mode0'] = Y_miss['mode0'].drop(
                index=Y_miss['mode0'].iloc[:pidx(.33,n)].index
            )
            Y_miss['mode1'] = Y_miss['mode1'].drop(
                index=Y_miss['mode1'].iloc[pidx(.33,n):pidx(.67,n)].index
            )
            Y_miss['mode2'] = Y_miss['mode2'].drop(
                index=Y_miss['mode2'].iloc[pidx(.67,n):].index
            )
        elif missing_pattern == 4:
            Y_miss['mode0'] = Y_miss['mode0'].drop(
                index=Y_miss['mode0'].iloc[:pidx(.4,n)].index
            )
            Y_miss['mode1'] = Y_miss['mode1'].drop(
                index=Y_miss['mode1'].iloc[pidx(.3,n):pidx(.7,n)].index
            )
            Y_miss['mode2'] = Y_miss['mode2'].drop(
                index=Y_miss['mode2'].iloc[pidx(.6,n):].index
            )
        Y_miss_tensor = torch.from_numpy(pd.concat([
            Y_miss['mode0'], Y_miss['mode1'], Y_miss['mode2']
        ], axis=1).values)
        Y_obs_mask = ~torch.isnan(Y_miss_tensor)

        Y_miss_copy1 = Y_miss.copy()
        Y_miss_copy2 = Y_miss.copy()
        Y_miss_copy3 = Y_miss.copy()
        if private_var:
            mean_res = fit(
                Y_miss, missing_modes='impute_mean',
                n_pcs='all', d=params['d'], k=params['k'], verbose=False,
                ll_exact=ll_exact, center=False, scale=False
            )
            try:
                drop_res = fit(
                    Y_miss_copy1, missing_modes='drop',
                    n_pcs='all', d=params['d'], k=params['k'],
                    verbose=False, ll_exact=ll_exact, center=False,
                    scale=False
                )
            except ValueError:
                drop_res = None
            approx_res = fit(
                Y_miss_copy2, missing_modes='impute_model_approx',
                n_pcs='all', d=params['d'], k=params['k'], verbose=False,
                ll_exact=ll_exact, center=False, scale=False
            )
            exact_res = fit(
                Y_miss_copy3, missing_modes='impute_model_exact',
                n_pcs='all', d=params['d'], k=params['k'],
                verbose=False, ll_exact=ll_exact, center=False, scale=False
            )
        else:
            mean_res = fit(
                Y_miss, missing_modes='impute_mean',
                n_pcs='all', d=params['d'], k=[0,0,0], verbose=False,
                ll_exact=ll_exact, center=False, scale=False
            )
            try:
                drop_res = fit(
                    Y_miss_copy1, missing_modes='drop',
                    n_pcs='all', d=params['d'], k=[0,0,0], verbose=False,
                    ll_exact=ll_exact, center=False, scale=False
                )
            except ValueError:
                drop_res = None
            approx_res = fit(
                Y_miss_copy2, missing_modes='impute_model_approx',
                n_pcs='all', d=params['d'], k=[0,0,0], verbose=False,
                ll_exact=ll_exact, center=False, scale=False
            )
            exact_res = fit(
                Y_miss_copy3, missing_modes='impute_model_exact',
                n_pcs='all', d=params['d'], k=[0,0,0], verbose=False,
                ll_exact=ll_exact, center=False, scale=False
            )

        if private_var:
            full_ll = log_likelihood(
                Y_miss_tensor, W_full_torch, L_full_torch,
                Phi_full, Y_obs_mask, n
            )
        else:
            full_ll = log_likelihood(
                Y_miss_tensor, W_full_torch, None,
                Phi_full, Y_obs_mask, n
            )
        print('  Full LL:', round(full_ll, 4))

        for name, res in {
            'Mean': mean_res, 'Drop': drop_res, 
            'Impute Approx': approx_res, 'Impute Exact': exact_res
        }.items():
            if res is not None:
                W_torch = torch.cat(
                    [torch.from_numpy(val.values)
                        for val in res.W.values()], dim=0
                )
                if private_var:
                    L_torch = torch.block_diag(
                        *[torch.from_numpy(val.values)
                            for val in res.L.values()]
                    )
                else:
                    L_torch = None
                Phi_pred = torch.block_diag(
                    *[torch.from_numpy(val.values)
                        for val in res.Phi.values()]
                )
                print('  ' + name, 'LL:', round(log_likelihood(
                    Y_miss_tensor, W_torch, L_torch,
                    Phi_pred, Y_obs_mask, n
                ), 4))
        print('')
params1 = {
    'd': 3,
    'k': [3, 4, 5],
    'p': [6, 7, 8],
    'sigsq': [1, 1, 1]
}

print('n = 1000, ll_exact=True')
test_modes(params1, n=1000, private_var=True, ll_exact=True)
print()
print('n = 1000, ll_exact=True, no private variance')
test_modes(params1, n=1000, private_var=False, ll_exact=True)

########## Output 01/29/2026 with seed 0 ##########
# n = 1000, ll_exact=True
# No missing data
#   Full LL: 37779.9928
#   Mean LL: 37779.9928
#   Drop LL: 37779.9928
#   Impute Approx LL: 37779.9928
#   Impute Exact LL: 37779.9928

# Missing pattern 1
# Missing modes detected for some samples, Imputing with the mean
# Missing modes detected for some samples, dropping samples with missing modes. There are 550 samples remaining.
# Missing modes detected in input, they will be imputed during model fitting using approximate EM.
# Missing modes detected in input, they will be imputed during model fitting using exact EM.
#   Full LL: 32032.5901
#   Mean LL: 32197.2514
#   Drop LL: 32056.2507
#   Impute Approx LL: 32348.0683
#   Impute Exact LL: 32018.7213

# Missing pattern 2
# Missing modes detected for some samples, Imputing with the mean
# Missing modes detected for some samples, dropping samples with missing modes. There are 250 samples remaining.
# Missing modes detected in input, they will be imputed during model fitting using approximate EM.
# Missing modes detected in input, they will be imputed during model fitting using exact EM.
#   Full LL: 28641.6404
#   Mean LL: 29062.9614
#   Drop LL: 28718.9683
#   Impute Approx LL: 29083.8928
#   Impute Exact LL: 28623.4242

# Missing pattern 3
# Missing modes detected for some samples, Imputing with the mean
# Missing modes detected for some samples, dropping samples with missing modes. There are 0 samples remaining.
# Missing modes detected in input, they will be imputed during model fitting using approximate EM.
# Missing modes detected in input, they will be imputed during model fitting using exact EM.
#   Full LL: 25612.2227
#   Mean LL: 26379.7575
#   Impute Approx LL: 26129.4894
#   Impute Exact LL: 25585.2196

# Missing pattern 4
# Missing modes detected for some samples, Imputing with the mean
# Missing modes detected for some samples, dropping samples with missing modes. There are 0 samples remaining.
# Missing modes detected in input, they will be imputed during model fitting using approximate EM.
# Missing modes detected in input, they will be imputed during model fitting using exact EM.
#   Full LL: 23127.3453
#   Mean LL: 24170.2951
#   Impute Approx LL: 23697.6108
#   Impute Exact LL: 23096.3916
#
# n = 1000, ll_exact=True, no private variance
# No missing data
#   Full LL: 22914.4556
#   Mean LL: 22914.4556
#   Drop LL: 22914.4556
#   Impute Approx LL: 22914.4556
#   Impute Exact LL: 22914.4556

# Missing pattern 1
# Missing modes detected for some samples, Imputing with the mean
# Missing modes detected for some samples, dropping samples with missing modes. There are 550 samples remaining.
# Missing modes detected in input, they will be imputed during model fitting using approximate EM.
# Missing modes detected in input, they will be imputed during model fitting using exact EM.
#   Full LL: 20110.11
#   Mean LL: 21594.8127
#   Drop LL: 20125.221
#   Impute Approx LL: 21805.287
#   Impute Exact LL: 20128.08

# Missing pattern 2
# Missing modes detected for some samples, Imputing with the mean
# Missing modes detected for some samples, dropping samples with missing modes. There are 250 samples remaining.
# Missing modes detected in input, they will be imputed during model fitting using approximate EM.
# Missing modes detected in input, they will be imputed during model fitting using exact EM.
#   Full LL: 18433.6201
#   Mean LL: 20226.3535
#   Drop LL: 18532.4355
#   Impute Approx LL: 20143.7113
#   Impute Exact LL: 19026.5464

# Missing pattern 3
# Missing modes detected for some samples, Imputing with the mean
# Missing modes detected for some samples, dropping samples with missing modes. There are 0 samples remaining.
# Missing modes detected in input, they will be imputed during model fitting using approximate EM.
# Missing modes detected in input, they will be imputed during model fitting using exact EM.
#   Full LL: 16892.6447
#   Mean LL: 18846.1104
#   Impute Approx LL: 18392.6536
#   Impute Exact LL: 16859.2233

# Missing pattern 4
# Missing modes detected for some samples, Imputing with the mean
# Missing modes detected for some samples, dropping samples with missing modes. There are 0 samples remaining.
# Missing modes detected in input, they will be imputed during model fitting using approximate EM.
# Missing modes detected in input, they will be imputed during model fitting using exact EM.
#   Full LL: 15456.8974
#   Mean LL: 17440.893
#   Impute Approx LL: 16795.6191
#   Impute Exact LL: 15410.1207
