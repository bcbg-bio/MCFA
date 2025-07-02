'''
Simulate (random) data to test MCFA model
'''

import torch

def simulate_factors(sim_params, private_var = True):
    '''
    Sample Z and X from standard normal
    '''
    d = sim_params['d']
    k = sim_params['k']
    n = sim_params['n']
    
    Z = torch.randn(d, n)  
    
    if private_var:
        X = [torch.randn(k[i], n) for i in range(len(k))]
    else:
        X = None

    return Z, X


def simulate_params(sim_params, private_var = True):
    '''
    Randomly generate loadings W and L and specific variance Phi
    '''
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


def simulate_noise(sim_params, Phi, A, private_var):
    '''
    Sample noise E from previously generated variance
    '''
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


def simulate_data(sim_params, private_var = True, verbose = False):
    '''
    Simulate parameters, latent variables, and noise to generate Y according to
    the formula Y = WZ + LX + E (or Y = WZ + E if no private factors)
    '''
    Z, X = simulate_factors(sim_params, private_var)
    W, L, Phi, A = simulate_params(sim_params, private_var)
    E = simulate_noise(sim_params, Phi, A, private_var)

    if private_var:
        if verbose:
            print("Y = WZ + LX + E")
        Y = [torch.matmul(W[i], Z) + torch.matmul(L[i], X[i]) + E[i] for i in range(len(W))]
    else:
        if verbose:
            print("No private factors, so Y = WZ + E")
        Y = [torch.matmul(W[i], Z) + E[i] for i in range(len(W))]

    # generate stacked Y data for input to EM step, if needed
    Y_stack = torch.cat(Y, dim=0)
    return Y_stack, W, L, Phi


def initialize_params(W, L, Phi, private_var = True):
    '''
    Randomly initialize W, L, and Phi
    '''
    W_init = [torch.randn(w.shape) for w in W]
    
    if private_var:
        L_init = [torch.randn(l.shape) for l in L]
    else:
        L_init = None
        
    Phi_init = [torch.eye(p.shape[0]) for p in Phi]

    return W_init, L_init, Phi_init

