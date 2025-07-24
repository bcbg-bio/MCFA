"""
Simulate (random) data to test MCFA model
"""
from typing import Dict, List, Union
import torch

type Params = Dict[str, Union[List[int], int]]

def simulate_factors(sim_params: Params, private_var: bool = True):
    """Samples Z and X from a standard normal distribution.
    Args: 
      sim_params: A dictionary of parameters d, k, n, p, and sigsq
        - d: Integer. Shared dimensionality across datasets.
        - k: List of integers. Private dimensionalities (k_m) for each dataset.
        - n: Integer. Total number of samples. 
        - p: List of integers. Full dimensionalities (p_m) for each datset. 
        - sigsq: List of integers. Private variances for each dataset. 
      private_var: Bool. Whether or not each dataset has private structure. If 
        private_var=False, k will be ignored. 
    Returns:
      A tuple (Z, X) if private_var=True or (Z, None) if private_var=False
        - Z: d (shared dimensions) by n (samples) tensor. Shared factors across 
             all datasets. 
        - X: List of k_m (private dimensions) by n (samples) tensors or None. 
             If private_var=True, contains private factors for each dataset. 
             Otherwise returns None and no private components will be modeled.  
    """
    d = sim_params['d']
    k = sim_params['k']
    n = sim_params['n']

    Z = torch.randn(d, n)

    if private_var:
        X = [torch.randn(k[i], n) for i in range(len(k))]
    else:
        X = None

    return Z, X


def simulate_params(sim_params: Params, private_var: bool = True):
    """Randomly generates loadings W and L and specific variance Phi.
    Args: 
      sim_params: A dictionary of parameters d, k, n, p, and sigsq
        - d: Integer. Shared dimensionality across datasets.
        - k: List of integers. Private dimensionalities (k_m) for each dataset.
        - n: Integer. Total number of samples. 
        - p: List of integers. Full dimensionalities (p_m) for each datset. 
        - sigsq: List of integers. Private variances for each dataset. 
      private_var: Bool. Whether or not each dataset has private structure. If 
        private_var=False, k and sigsq will be ignored. 
    Returns:
      A tuple (W, L, Phi, A) 
        - W: List of p_m (features) by d (shared dimensions) tensors. Shared 
             factor loadings for each dataset. 
        - L: List of p_m (features) by k_m (private dimensions) tensors or 
             None. If private_var=True, represents private factor loadings for 
             each dataset. Otherwise returns None and no private components 
             will be modeled. 
        - Phi: List of p_m (features) by p_m (features) tensors. Covariance 
             matrices for each dataset. If private_var=True, each tensor Phi_m 
             is a diagonal matrix generated from sigsq_m. If False, each matrix 
             is generated via A_m @ A_m.T 
        - A: List of p_m (features) by p_m (features) tensors or None. If 
             private_var=True, returns None. Otherwise returns normalized 
             matrices used to generate symmetric, non-diagonal Phi. 
    """
    d = sim_params['d']
    k = sim_params['k']
    p = sim_params['p']
    sigsq = sim_params['sigsq']

    W = [torch.randn(p[i], d) for i in range(len(p))]

    if private_var:
        L = [torch.randn(p[i], k[i]) for i in range(len(p))]
        A = None
        Phi = [torch.diag(torch.full((p[i],), sigsq[i])) for i in range(len(p))]
    else:
        L = None
        A = [
            (torch.randn(p[i], p[i]) / torch.sqrt(torch.tensor(p[i], dtype=torch.float32)))
            for i in range(len(p))
        ]
        Phi = [A[i] @ A[i].T for i in range(len(A))]

    return W, L, Phi, A


def simulate_noise(sim_params : Params, Phi : List[torch.tensor], 
                   A : Union[List[torch.tensor], None], 
                   private_var : bool = True):
    """Samples noise E from previously generated variance Phi. 
    Args: 
      sim_params: A dictionary of parameters d, k, n, p, and sigsq
        - d: Integer. Shared dimensionality across datasets.
        - k: List of integers. Private dimensionalities (k_m) for each dataset.
        - n: Integer. Total number of samples. 
        - p: List of integers. Full dimensionalities (p_m) for each datset. 
        - sigsq: List of integers. Private variances for each dataset. 
      Phi: List of p_m (features) by p_m (features) tensors. Covariance 
        matrices for each dataset. 
      A: A list of p_m (features) by p_m (features) tensors or None. Random 
        matrices used to generate Phi when private_var=False. 
      private_var: Bool. Whether or not each dataset has private structure.
    Returns:
      A list of noise tensors E_m with shapes p_m (features) by n (samples). 
    """
    p = sim_params['p']
    n = sim_params['n']

    if private_var:
        E = [
            torch.matmul(
                torch.linalg.cholesky(Phi[i]), 
                torch.randn(p[i], n)
            ) for i in range(len(p))
        ]
    else:
        E = [
            torch.matmul(A[i], torch.randn(p[i], n))
            for i in range(len(p))
        ]
    return E


def simulate_data(sim_params : Params, private_var : bool = True):
    """Simulate parameters, latent variables, and noise to generate Y according 
    to the formula Y = WZ + LX + E (or Y = WZ + E if no private structure)
    Args: 
      sim_params: A dictionary of parameters d, k, n, p, and sigsq
        - d: Integer. Shared dimensionality across datasets.
        - k: List of integers. Private dimensionalities (k_m) for each dataset.
        - n: Integer. Total number of samples. 
        - p: List of integers. Full dimensionalities (p_m) for each datset. 
        - sigsq: List of integers. Private variances for each dataset. 
      private_var: Bool. Whether or not each dataset has private structure. 
    Returns: 
      A tuple (Y, W, L, Phi)
        - Y: p (features) by n (samples) tensor. Simulated data observations 
             for all datasets. 
        - W: List of p_m (features) by d (shared dimensions) tensors. Shared 
             factor loadings for each dataset. 
        - L: List of p_m (features) by k_m (private dimensions) tensors. 
             Private factor loadings for each dataset. 
        - Phi: List of p_m (features) by p_m (features) tensors. Covariance 
             matrices for each dataset.
    Raises: 
      ValueError: If the number of datasets or dimensions are inconsistent
        across parameters, or if any values in sigsq are negative. 
    """
    if len(sim_params['k']) != len(sim_params['p']) \
       or len(sim_params['k']) != len(sim_params['sigsq']):
        raise ValueError("Params k, p, and sigsq must all be the same length.")
    if any([
        sim_params['p'][i] < sim_params['k'][i] + sim_params['d']
        for i in range(len(sim_params['k']))
    ]): 
        raise ValueError("Number of features p_m for a dataset must equal at least k_m + d.")
    if any([sigsq < 0 for sigsq in sim_params['sigsq']]):
        raise ValueError("Variance values in sigsq must be positive.")

    Z, X = simulate_factors(sim_params, private_var)
    W, L, Phi, A = simulate_params(sim_params, private_var)
    E = simulate_noise(sim_params, Phi, A, private_var)

    if private_var:
        print("Y = WZ + LX + E")
        Y = [torch.matmul(W[i], Z) + torch.matmul(L[i], X[i]) + E[i] for i in range(len(W))]
    else:
        print("No private structure, so Y = WZ + E")
        Y = [torch.matmul(W[i], Z) + E[i] for i in range(len(W))]

    # generate stacked Y data for input to EM step, if needed
    Y_stack = torch.cat(Y, dim=0)
    return Y_stack, W, L, Phi


def initialize_params(W : List[torch.tensor], L : List[torch.tensor], 
                      Phi : List[torch.tensor], private_var : bool = True):
    """
    Randomly initializes W, L, and Phi. 
    Args: 
      - W: List of p_m (features) by d (shared dimensions) tensors. Shared 
           factor loadings for each dataset. 
      - L: List of p_m (features) by k_m (private dimensions) tensors. 
           Private factor loadings for each dataset. 
      - Phi: List of p_m (features) by p_m (features) tensors. Covariance 
           matrices for each dataset.
      - private_var: Bool. Whether or not each dataset has private structure.
    Returns: 
      Tuple of (W_init, L_init, Phi_init)
        - W_init: List of p_m (features) by d (shared dimensions) tensors. 
                  Randomly initialized shared factor loading matrices. 
        - L_init: List of p_m (features) by k_m (private dimensions) tensors 
                  or None. If private_var=True, randomly initialized private 
                  factor loading matrices. Otherwise None and no private 
                  factors are modeled. 
        - Phi_init: List of p_m (features) by p_m (features) tensors. 
                  Identity matrices representing initial covariance values. 
    """
    W_init = [torch.randn(w.shape) for w in W]

    if private_var:
        L_init = [torch.randn(l.shape) for l in L]
    else:
        L_init = None

    Phi_init = [torch.eye(p.shape[0]) for p in Phi]

    return W_init, L_init, Phi_init
