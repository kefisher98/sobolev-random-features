###############################################################
# [README]
#
# Minimal fixed point system for Sobolev training, implemented
# asymptotically via operator-valued Cauchy transforms.
#
# We solve for V assuming it is diagonal, and that (Vc)_i is the same
# for all i = 1,...,k. Accordingly, we only store the scalars (Va, Vc).
#
# Empirically we observe that V takes on large values after the phase transition.
# We reparametrize V -> reparam(V) to help ``precondition'' the fixed point system.
#
# Then we solve for (f, fhat), and finally (q, qhat).
#
# The q overlap parameters are further separated into components 
# that scale with alignment \varpi.
#
###############################################################

import numpy as np
import matplotlib.pyplot as plt
from texSettings     import *
from scipy.optimize  import root
from scipy.special   import erf
from scipy.integrate import quad, dblquad
from scipy.linalg    import sqrtm
from scipy.sparse    import diags
import tqdm
import multiprocessing as mp
import contextlib as clib
from datetime import datetime
import os
import sys
import copy
import argparse
from types import SimpleNamespace

###############################################################
# global parameters
###############################################################

seed = 42

parser = argparse.ArgumentParser()

# required arguments
parser.add_argument("lbda", type=float, help="regularization")
parser.add_argument("rf_str", type=str, help="random features")

# optional arguments
parser.add_argument('--d', type=int, default=100, help="dimension to use for Monte Carlo simulations")
parser.add_argument('--ratio_nd', type=float, default=2.345, help="no. training samples / no. dimensions")
parser.add_argument('--nonlin_str', type=str, default='id', help="forward model nonlinearity")
parser.add_argument('--sigma_str', type=str, default='erf', help="activation function")
parser.add_argument('--root_alg', type=str, default='excitingmixing', help='root finding algorithm')
parser.add_argument('--k', type=int, default=1, help="no. of derivative directions")
parser.add_argument('--delta',      type = float, default = 0.,    help = "additive noise standard dev for data")
parser.add_argument('--tau',        type = float, default = 1.,    help = "relative weight of gradient term in training loss")

parser.add_argument('--ratio_pn_start', type=float, default=0.01, help="ratio_pn = linspace(start, end, num)")
parser.add_argument('--ratio_pn_end', type=float, default=4.0, help="ratio_pn = linspace(start, end, num)")
parser.add_argument('--ratio_pn_num', type=int, default=101, help="ratio_pn = linspace(start, end, num)")
parser.add_argument('--logspacing', action = 'store_true', help = "logarithmic spacing for pn scan")
parser.add_argument('--not_logspacing', dest = 'verbose', action = 'store_false')
parser.set_defaults(logspacing = False)

parser.add_argument('--verbose', action = 'store_true', help = "print output for pn scan")
parser.add_argument('--not_verbose', dest = 'verbose', action = 'store_false')
parser.set_defaults(verbose = False)

parser.add_argument('--save', type=bool, default=True, help="save results if true")

parser.add_argument('--root_tol', type=float, default=1e-7, help="error tolerance for root finding")

args = parser.parse_args()

rf_str = args.rf_str
nonlin_str = args.nonlin_str
sigma_str = args.sigma_str
d = args.d
k = args.k
tau = args.tau

###############################################################

match nonlin_str:
    case 'id':
        phi = lambda omega : omega
        d_phi = lambda omega : np.ones_like(omega)
        ddphi = lambda omega : np.zeros_like(omega)
    case 'arctan':
        phi = lambda omega : np.arctan(omega)
        d_phi = lambda omega : 1. / (1. + omega**2.)
        ddphi = lambda omega : -2 * omega / (1. + omega**2.)**2
    case 'cosh':
        phi = lambda omega : np.cosh(omega)
        d_phi = lambda omega : np.sinh(omega)
        ddphi = lambda omega : np.cosh(omega)
    case 'om2':
        phi = lambda omega : omega**2. - 1.
        d_phi = lambda omega : 2. * omega
        ddphi = lambda omega : 2. * np.ones_like(omega)
    case 'reci-cosh':
        phi = lambda omega : 1. / np.cosh(omega)
        d_phi = lambda omega : - np.sinh(omega) / np.cosh(omega)**2.
        ddphi = lambda omega : -1. / np.cosh(omega) + 2. * np.sinh(omega)**2. / np.cosh(omega)**3.
    case 'arctan-plus-reci-cosh':
        phi   = lambda omega : np.arctan(omega) + 1. / np.cosh(omega)
        d_phi = lambda omega : 1. / (1. + omega**2.) - np.sinh(omega) / np.cosh(omega)**2.
        ddphi = lambda omega : -2 * omega / (1. + omega**2.)**2 -1. / np.cosh(omega) + 2. * np.sinh(omega)**2. / np.cosh(omega)**3.
    case 'Gaussian':
        phi = lambda omega : np.exp(  -omega**2/2. ) / (2.**np.sqrt(np.pi))
        d_phi = lambda omega :- omega*np.exp(  -omega**2/2. )  / (2.**np.sqrt(np.pi))
        ddphi = lambda omega :( omega**2 * np.exp(  -omega**2/2. )  -  np.exp(  -omega**2/2. ) ) / (2.**np.sqrt(np.pi))
    case 'ReLU':
        phi  = lambda omega : np.where(omega > 0., omega, 0.)
        d_phi = lambda omega : np.where(omega > 0., 1., 0.)
        ddphi = lambda omega : np.zeros(np.shape(omega))
    case 'linGauss':
        phi   = lambda omega : omega/2. - np.exp(  -omega**2/2. ) 
        d_phi = lambda omega : 1./2. +  omega*np.exp(  -omega**2/2. )  
        ddphi = lambda omega : -omega**2 * np.exp(  -omega**2/2. )  +  np.exp( -omega**2/2. )  
    case 'cos+sin':
        phi   = lambda omega : np.cos(omega) + np.sin(omega)
        d_phi = lambda omega : np.cos(omega) - np.sin(omega)
        ddphi = lambda omega : -np.cos(omega) - np.sin(omega)
    case _:
        raise NotImplementedError(f"nonlin_str = {nonlin_str}")

base_dir = 'sobo-rf-theory-{}'.format(nonlin_str)

###############################################################
# activation function

def get_k0(sigma):
    x_quad, w_quad = make_gauss_quad1D(150, mu=0., var=1.)
    return np.dot(w_quad, sigma(x_quad) )

def get_k1(sigma):
    x_quad, w_quad = make_gauss_quad1D(150, mu=0., var=1.)
    return np.dot(w_quad, x_quad * sigma(x_quad) )

def get_ks(sigma, k0, k1):
    x_quad, w_quad = make_gauss_quad1D(150, mu=0., var=1.)
    tmp = np.dot( w_quad, sigma(x_quad)**2 )
    return np.sqrt(tmp - k0**2 - k1**2)

match sigma_str:
    case 'ReLU':
        sigma = lambda x : np.where(x > 0., x, 0.)
        sigma_prime = lambda x : np.where(x > 0., 1., 0.)

    case 'SiLU':
        sigma = lambda x : x / (1. + np.exp(-x))
        sigma_prime = lambda x : (1. + (1. + x) * np.exp(-x)) / (1. + np.exp(-x))**2.

    case 'erf':
        sigma = lambda x : erf(x)
        sigma_prime = lambda x : 2. / np.sqrt(np.pi) * np.exp(-x**2.)

    case 'sign':
        sigma = lambda x : np.where(x >= 0., 1., -1.)
        sigma_prime = lambda x : 0. * x

    case 'tanh':
        sigma = lambda x : np.tanh(x)
        sigma_prime = lambda x : 1. / np.cosh(x)**2

    case _:
        raise NotImplementedError(f"sigma_str= {sigma_str}")

###############################################################
# noise models -- additive Gaussian for function and gradient

delta      = args.delta
# ~ noise_cov  = np.zeros((k+1,k+1))
# ~ noise_cov[0,0] = delta**2
noise_cov  = np.eye(k+1) * delta**2

###############################################################
# helper functions

def make_gauss_quad1D(deg, mu=0., var=1.):
    # quadrature of degree `deg` for normal distribution with mean `mu` and variance `var`
    xq, wq = np.polynomial.hermite_e.hermegauss(deg)
    wq = wq/np.sqrt(2*np.pi)  # standard Gaussian normalization
    xq = np.sqrt(var)*xq + mu # shift and scale
    
    return xq, wq

def tensor_product_quadrature(x1, w1, x2, w2):
    # takes two quadrature rules (x1, w1) and (x2, w2) and forms the tensor product quadrature.
    x12 = np.meshgrid(x1, x2)
    x12 = np.column_stack( (x12[0].ravel(), x12[1].ravel()) )  # (nx1*nx2, 2)

    w12 = np.meshgrid(w1, w2)
    w12 = np.column_stack( (w12[0].ravel(), w12[1].ravel()) )
    w12 = np.prod(w12, axis=-1) # (nx1*nx2, )

    return x12, w12

def make_gauss_quad2D(deg, mu, sigma):
    L = np.linalg.cholesky(sigma) # sigma = L@L.T
    x1, w1 = make_gauss_quad1D(deg, mu=0., var=1.)
    x12, w12 = tensor_product_quadrature(x1, w1, x1, w1)

    # rotate by Cholesky factor
    x12 = x12 @ L.T + mu # (nx1*nx2, 2)
    
    return x12, w12

def get_gauss_quad( k, num_mc=3000 ):

    if k==1:
        lambdas_x, weights_x = make_gauss_quad1D(31, mu=0., var=1.)
    elif k==2:
        lambdas_x, weights_x = make_gauss_quad2D(21, np.zeros(2), np.eye(2))
    else:
        lambdas_x = np.random.randn(num_mc, k)
        weights_x = np.ones(num_mc) / num_mc

    return lambdas_x, weights_x

# generic quadrature generator for compact support
def get_quad(lam_min, lam_max, num_points = 100):
    lambdas, weights = np.polynomial.legendre.leggauss(num_points)
    lambdas = 0.5 * (lambdas + 1.) * (lam_max - lam_min) + lam_min
    weights = 0.5 * (lam_max - lam_min) * weights
    return lambdas, weights

def slice_by_lengths(arr, lengths):
    """Slice a 1D numpy array into chunks """
    assert sum(lengths) == len(arr), "lengths not equal"

    slices = []
    start = 0
    for length in lengths:
        end = start + length
        slices.append(arr[start:end])
        start = end
    return slices

def unpack(params):
    # (f, q2, q0, V)
    chunks = (2, 2, 2, 2)
    f, q2, q0, V = slice_by_lengths(params, chunks)

    return [ f, q2, q0, V ]

########################################################################
# scalar spectral densities

def mp_density(lam, c = 1.0, ):
    a = (1 - np.sqrt(c))**2
    b = (1 + np.sqrt(c))**2
#     if lam < a or lam > b or lam <= 0:
#         return 0.0
    return (1 / (2 * np.pi * c * lam)) * np.sqrt((b - lam) * (lam - a) * (lam >= a) * (lam <= b)) 

def semicircular_density(lam):
    if np.abs(lam) > 1:
        return 0.0
    return (2 / np.pi) * np.sqrt(1. - lam**2)

########################################################################
# generic function to lift scalar cauchy trafo to constant \otimes random matrix cauchy trafo
def operator_cauchy_transform(Z, A, density_func, lambdas, weights, atoms = None):

    d = Z.shape[0]
    result = np.zeros((d, d), dtype=np.complex128)
    
    resolvent = np.einsum('k,ij->kij', lambdas, A) # (nw, d, d)
    resolvent = np.linalg.inv(Z - resolvent)

    result = np.einsum('k,k,kij->ij', weights, density_func(lambdas), resolvent)

    if atoms is not None:
        tmp = np.einsum('k,ij->kij', atoms[0], A) # (n_atom, d, d)
        tmp = np.linalg.inv(Z - tmp)

        tmp = np.einsum('k, kij->ij', atoms[1], tmp)

        result += tmp
    
    return result

def operator_cauchy_transform_Nd(Z, coeff_mtx, density_func, lambdas, weights, atoms = None):
    # didnt implement atom part here since we won't need it

    # lambdas of shape (nw, ) if k=1 and (nw, k) otherwise
    # weights of shape (nw, )

    d  = Z.shape[0]
    nw = len(weights)
    result = np.zeros((d, d), dtype=np.complex128)
    
    # np.linalg.inv automatically vectorizes over inputs of size (nbatch, d, d)
    if k == 1:
        resolvent = np.einsum('k,ij->kij', lambdas, coeff_mtx[:,:,0]) # (nw, d, d)
    else:
        resolvent = np.einsum('lk,ijk->lij', lambdas, coeff_mtx) # (nw, d, d)

    resolvent = Z - resolvent
    resolvent = np.linalg.inv(resolvent)

    result = np.einsum('k, kij->ij', weights, resolvent)

    return result

# generic sum of free operator-valued cauchy trafos from subordinator
def get_g_sum(Z, G_x_func, G_y_func, max_iter=int(1e4), tol=1e-8, alpha=0.2, init_omega=None, return_omega=False):
    
    if init_omega is not None:
        omega = copy.copy(init_omega)
    else:
#         omega = 1j * np.eye(Z.shape[0])
        omega = 1j * np.random.randn(*Z.shape)

    # ~ for _ in tqdm.tqdm(range(max_iter), leave=False):
    for i in range(max_iter):
        G_x_omega = G_x_func(omega)
        try:
            h_x_omega = np.linalg.inv(G_x_omega) - omega
        except np.linalg.LinAlgError:
            raise ValueError("Matrix inversion failed in h_x computation")
        
        arg = h_x_omega + Z
        G_y_arg = G_y_func(arg)
        try:
            h_y = np.linalg.inv(G_y_arg) - arg
        except np.linalg.LinAlgError:
            raise ValueError("Matrix inversion failed in h_y computation")
        omega_new = h_y + Z
        omega_new = alpha * omega_new + (1. - alpha) * omega

        if np.linalg.norm(omega_new - omega, ord='fro') < tol:
            # ~ print('no. inner iters:', i)
            ret = (G_x_func(omega_new)[0,0],)
            if return_omega:
                ret  = ret + (omega_new,)
            return ret
            
        omega = omega_new
    print("Warning: omega(Z) iteration did not converge")
    ret = (G_x_func(omega_new)[0,0],)
    if return_omega:
        ret  = ret + (omega_new,)
    return ret

###############################################################

def compute_trace_rational_type_1(MP_param, z_0, z_1, z_2, z_3, b_0, b_1):
    """
    Normalized trace of 
    \[
        (b0 + b1*m)^{-1} (z0 + z1*m + sum gk (z2 + z3*m) gk)
    \]
    where m is Marchenko-Pastur and gk is Gaussian.
    """

    lift_dim = 2*k + 5

    Z = np.zeros( (lift_dim, lift_dim) )

    if k == 1:
        lambdas_x, weights_x = make_gauss_quad1D(31, mu=0., var=1.)
    elif k == 2:
        lambdas_x, weights_x = make_gauss_quad2D(21, np.zeros(2), np.eye(2))
    else:
        np.random.seed(seed)
        num_mc = 3000
        lambdas_x = np.random.randn(num_mc, k)
        weights_x = np.ones(num_mc) / num_mc
        np.random.seed(None)

    density_x = lambda lam: 1.

    coeff_mtx = np.zeros((lift_dim, lift_dim, k), dtype=complex)

    for kk in range(k):
        coeff_mtx[1, 2+2*(kk+1), kk] = 1.
        coeff_mtx[4+2*(kk+1), 0, kk] = 1.
    min_y = (1. - np.sqrt(MP_param))**2
    max_y = (1. + np.sqrt(MP_param))**2
    lambdas_y, weights_y = get_quad(min_y, max_y)
    density_y = lambda lam: mp_density(lam, MP_param)
    if MP_param > 1.:
        atoms_y = ([0.], [1. - 1. / MP_param])
    else:
        atoms_y = None

    A_y = np.zeros((lift_dim, lift_dim), dtype=complex)
    A_y[1,-1] = b_1
    A_y[3, 1] = z_1
    for kk in range(k):
        A_y[3+2*(kk+1), 1+2*(kk+1)] = z_3

    B_y = np.zeros((lift_dim, lift_dim), dtype=complex)
    B_y[0,-2] = B_y[1, 2] = 1
    B_y[1,-1] = b_0
    B_y[2,-2] = B_y[2,-1] = 1
    B_y[4, 0] = 1

    B_y[3, 1] = z_0
    B_y[3, 2] = B_y[4, 1] = -1

    for kk in range(k):
        B_y[3+2*(kk+1), 2+2*(kk+1) -1] = z_2
        B_y[3+2*(kk+1) ,2+2*(kk+1)] = -1
        B_y[4+2*(kk+1), 1+2*(kk+1)] = -1

    def custom_cauchy_transform_Nd_vectorized(Z, coeff_mtx, density_func, lambdas, weights, atoms=None):
        """ 
        Computes this specific Cauchy transform via SCHUR COMPLEMENTS.
        
        Parameters:
            lambdas of shape (nw, ) if k=1 and (nw, k) otherwise
            weights of shape (nw, )
        """

        # python passes by reference so changes to inputs are dangerous
        Zin = copy.copy(Z)
        coeff_mtx_in = copy.copy(coeff_mtx)

        d = Zin.shape[0]
        result = np.zeros((d, d), dtype=np.complex128)

        ###################
        # permute FIRST and SECOND rows
        Zin[[0,1]] = Zin[[1, 0]]
        coeff_mtx_in[[0,1]] = coeff_mtx_in[[1,0]]

        ###################
        # label schur sub-matrices [A, B ; C, D]

        A = Zin[0,0].item() # scalar element
        B = Zin[[0], 1:]    # (1, d-1)
        C = Zin[1:, [0]]    # (d-1, 1)
        D = Zin[1:, 1:]      # (d-1, d-1)

        Dinv = np.linalg.inv(D)

        ###################
        # schur complement

        if k == 1:
            lam_C = np.einsum('ij,k->kij', coeff_mtx_in[1:, [0], 0] , lambdas) # (nw, d-1, 1)
            lam_B = np.einsum('ij,k->kij', coeff_mtx_in[[0], 1:, 0] , lambdas) # (nw, 1, d-1)

        else:
            lam_C = np.einsum('ijk,lk->lij', coeff_mtx_in[1:, [0], :], lambdas ) # (nw, d-1, 1)
            lam_B = np.einsum('ijk,lk->lij', coeff_mtx_in[[0], 1:, :], lambdas ) # (nw, 1, d-1)

        schur = np.matmul(Dinv, C[None,:,:] - lam_C) # (nw, d-1, 1)
#     schur = np.einsum('kij,kjl->kil' , B[None,:,:] - lam_B, schur) # (nw, 1, 1)
        schur = np.matmul(B[None,:,:] - lam_B, schur) # (nw, 1, 1)
        schur = schur[:,0,0] # (nw, ) 
        schur = 1. / (A - schur) # (nw, )

        schur_lam_B = np.einsum('kij,k->kij', lam_B, schur) # (nw, 1, d-1)
        schur_lam_C = np.einsum('kij,k->kij', lam_C, schur) # (nw, d-1, 1)
        schur_lam_CB = np.matmul(lam_C, lam_B)  # (nw, d-1, d-1)
        schur_lam_CB = np.einsum('kij,k->kij', schur_lam_CB, schur) # (nw, d-1, d-1)

        #####################
        # reduce over quadrature

        schur = np.dot(schur, weights)

        schur_lam_B = np.einsum('kij,k->ij', schur_lam_B, weights)
        schur_lam_C = np.einsum('kij,k->ij', schur_lam_C, weights)
        schur_lam_CB = np.einsum('kij,k->ij', schur_lam_CB, weights)

        result[0, 0] = schur
        result[1:, [0]] = - Dinv @ ( C * schur - schur_lam_C)
        result[[0], 1:] = - (schur * B - schur_lam_B ) @ Dinv

        result[1:, 1:] = C @ B * schur 
        result[1:, 1:] -= schur_lam_C @ B 
        result[1:, 1:] -= C @ schur_lam_B 
        result[1:, 1:] += schur_lam_CB

        result[1:, 1:] = Dinv + Dinv @ result[1:, 1:] @ Dinv

        ###################
        # permute FIRST and SECOND columns
        result[:, [0,1]] = result[:, [1,0]]

        return result

    def G_xhat(Z):
        return custom_cauchy_transform_Nd_vectorized(Z, coeff_mtx, density_x, lambdas_x, weights_x)


    def G_yhat(Z):
        return operator_cauchy_transform(Z - B_y, A_y, density_y, lambdas_y, weights_y, atoms = atoms_y)

    return -np.real(get_g_sum(Z, G_xhat, G_yhat))[0]

def invert_3x3_cofactors_vectorized(A):
    """
    Invert a batch of 3x3 matrices using the method of cofactors.

    Parameters:
    A : numpy.ndarray of shape (nbatch, 3, 3)
        Batch of 3x3 matrices to invert

    Returns:
    numpy.ndarray of shape (nbatch, 3, 3)
        Batch of inverted matrices

    Raises:
    ValueError: If any matrix is singular (determinant is zero)
    """

    nbatch = A.shape[0]

    # Extract elements for readability
    a11, a12, a13 = A[:, 0, 0], A[:, 0, 1], A[:, 0, 2]
    a21, a22, a23 = A[:, 1, 0], A[:, 1, 1], A[:, 1, 2]
    a31, a32, a33 = A[:, 2, 0], A[:, 2, 1], A[:, 2, 2]

    # Calculate determinant
    det = (a11 * (a22 * a33 - a23 * a32) -
           a12 * (a21 * a33 - a23 * a31) +
           a13 * (a21 * a32 - a22 * a31))

    # Check for singular matrices
    if np.any(np.abs(det) < 1e-12):
        singular_indices = np.where(np.abs(det) < 1e-12)[0]
        raise ValueError(f"Singular matrix detected at indices: {singular_indices}")

    # Calculate cofactor matrix elements
    c11 = a22 * a33 - a23 * a32
    c12 = -(a21 * a33 - a23 * a31)
    c13 = a21 * a32 - a22 * a31

    c21 = -(a12 * a33 - a13 * a32)
    c22 = a11 * a33 - a13 * a31
    c23 = -(a11 * a32 - a12 * a31)

    c31 = a12 * a23 - a13 * a22
    c32 = -(a11 * a23 - a13 * a21)
    c33 = a11 * a22 - a12 * a21

    # Create adjugate matrix (transpose of cofactor matrix)
    adj = np.zeros((nbatch, 3, 3), dtype=complex)
    adj[:, 0, 0] = c11
    adj[:, 0, 1] = c21
    adj[:, 0, 2] = c31
    adj[:, 1, 0] = c12
    adj[:, 1, 1] = c22
    adj[:, 1, 2] = c32
    adj[:, 2, 0] = c13
    adj[:, 2, 1] = c23
    adj[:, 2, 2] = c33

    # Inverse = adjugate / determinant
    det_inv = 1.0 / det
    A_inv = adj * det_inv[:, np.newaxis, np.newaxis]

    return A_inv

def compute_trace_rational_type_2(MP_param, z_0, z_1, z_2, z_3, b_0, b_1):
    """
    Normalized trace of 
    \[
        (g1 * (b0 + b1*m) * g1)^{-1} (z0 + z1*m + sum gk (z2 + z3*m) gk)
    \]
    where m is Marchenko-Pastur and gk is Gaussian.
    """

    lift_dim = 2*k + 6

    Z = np.zeros( (lift_dim, lift_dim) )

    if k == 1:
        lambdas_x, weights_x = make_gauss_quad1D(31, mu=0., var=1.)
    elif k == 2:
        lambdas_x, weights_x = make_gauss_quad2D(31, np.zeros(2), np.eye(2))
    else:
        np.random.seed(seed)
        num_mc = 3000
        lambdas_x = np.random.randn(num_mc, k)
        weights_x = np.ones(num_mc) / num_mc
        np.random.seed(None)

    density_x = lambda lam: 1.

#     # pre-compute the sample-wise Gaussian perturbation matrices
#     coeff_mtx_x = np.zeros((len(weights_x), lift_dim, lift_dim,), dtype=complex)
# 
#     if k == 1:
#         for kk in range(k):
#             coeff_mtx_x[:, 1, 2+2*(kk+1)] = lambdas_x
#             if kk == 0:
#                 coeff_mtx_x[:, 1, -1] = lambdas_x
#                 coeff_mtx_x[:, 3, -3] = lambdas_x
# 
#             coeff_mtx_x[:, 5+2*(kk+1), 0, ] = lambdas_x
#     else:
#         for kk in range(k):
#             coeff_mtx_x[:, 1, 2+2*(kk+1), ] = lambdas_x[:,kk]
#             if kk == 0:
#                 coeff_mtx_x[:, 1, -1] = lambdas_x[:,kk]
#                 coeff_mtx_x[:, 3, -3] = lambdas_x[:,kk]
# 
#             coeff_mtx_x[:, 5+2*(kk+1), 0] = lambdas_x[:,kk]

    min_y = (1. - np.sqrt(MP_param))**2
    max_y = (1. + np.sqrt(MP_param))**2
    lambdas_y, weights_y = get_quad(min_y, max_y)
    density_y = lambda lam: mp_density(lam, MP_param)
    if MP_param > 1.:
        atoms_y = ([0.], [1. - 1. / MP_param])
    else:
        atoms_y = None

    A_y = np.zeros((lift_dim, lift_dim), dtype=complex)
    A_y[2,-2] = -b_1
    A_y[4, 1] = z_1
    for kk in range(k):
        A_y[4+2*(kk+1), 1+2*(kk+1)] = z_3

    B_y = np.zeros((lift_dim, lift_dim), dtype=complex)
    B_y[0,-3] = 1 
    B_y[1, 2] = 1
    B_y[2,-2] = -b_0
    B_y[2,-1] = 1
    B_y[3,-2] = 1

    B_y[5, 0] = 1

    B_y[4, 1] = z_0
    B_y[4, 2] = -1
    B_y[5, 1] = -1

    for kk in range(k):
        B_y[4+2*(kk+1), 2+2*(kk+1) -1] = z_2
        B_y[4+2*(kk+1) ,2+2*(kk+1)] = -1
        B_y[5+2*(kk+1), 1+2*(kk+1)] = -1

    def custom_cauchy_transform_Nd_vectorized(Z, lambdas, weights, ):
        """ 
        Computes this specific Cauchy transform via WOODBURY.
        
        Parameters:
            lambdas of shape (nw, ) if k=1 and (nw, k) otherwise
            weights of shape (nw, )
        """

        d = Z.shape[0]
        nw = len(weights)

        Zinv = np.linalg.inv(Z) # (d, d)

#         # rank three perturbation matrices
#         U = np.zeros((nw, d, 3), dtype=complex)
#         V = np.zeros((nw, 3, d), dtype=complex)
# 
#         U[:, 1, 0] = 1. # e2 basis
#         V[:, 0, :] = coeff_mtx_lam[:, 1, :] # second row of lam
# 
#         U[:, 3, 1] = 1. # e4 basis
#         V[:, 1, :] = coeff_mtx_lam[:, 3, :] # fourth row of lam
# 
#         U[:, :, 2] = coeff_mtx_lam[:, :, 0] # first column of lam
#         V[:, 2, 0] = 1. # e1.T basis

#         # inverse of 3x3 sub-matrix 
#         wood = np.matmul(Zinv[None,:,:], U)
#         wood = np.matmul(V, wood)

        # compute sparse matrix product Zinv @ U
        tmp1 = np.zeros((nw, d, 3), dtype=complex)
        tmp1[:,:,0] = Zinv[None,:,1]
        tmp1[:,:,1] = Zinv[None,:,3]
        if k == 1:
            tmp1[:,:,2] = np.einsum('i,k->ki', Zinv[:, 5+2*(1)], lambdas) 
        else:
            tmp1[:,:,2] = np.einsum('ij,kj->ki', Zinv[:, 5+2*(np.arange(k)+1)], lambdas) 

        # compute sparse matrix product V @ (Zinv @ U)
        wood = np.zeros((nw, 3, 3), dtype=complex)

        if k == 1:
            wood[:,0,:] = np.einsum('ki,k->ki', tmp1[:,2+2*(1),:], lambdas)
            wood[:,0,:] += np.einsum('ki,k->ki', tmp1[:,-1,:], lambdas)

            wood[:,1,:] = np.einsum('ki,k->ki', tmp1[:,-3,:], lambdas)

        else:
            wood[:,0,:] = np.einsum('kij,ki->kj', tmp1[:,2+2*(np.arange(k)+1), :], lambdas)
            wood[:,0,:] += np.einsum('kj,k->kj', tmp1[:,-1,:], lambdas[:,0])

            wood[:,1,:] += np.einsum('kj,k->kj', tmp1[:,-3,:], lambdas[:,0])

        wood[:,2,:] = tmp1[:,0,:]

        wood[:,0,0] -= 1.
        wood[:,1,1] -= 1.
        wood[:,2,2] -= 1.

        wood_inv = invert_3x3_cofactors_vectorized(wood) # (nw, 3, 3)

#         # assemble Woodbury inverse
#         result = np.matmul(U, wood_inv)
#         result = np.matmul(result, V)
        tmp1[:,:,1] = Zinv[None,:,3]
        if k == 1:
            tmp1[:,:,2] = np.einsum('i,k->ki', Zinv[:, 5+2*(1)], lambdas)

        # compute sparse matrix product U @ wood_inv
        tmp2 = np.zeros((nw, d, 3), dtype='complex')

        tmp2[:,1,:] = wood_inv[:,0,:]
        tmp2[:,3,:] = wood_inv[:,1,:]

        if k == 1:
            tmp2[:,5+2*(1),:] = np.einsum('k,kj->kj', lambdas, wood_inv[:,2,:]) 
        else:
            tmp2[:,5+2*(np.arange(k)+1),:] = np.einsum('ki,kj->kij', lambdas, wood_inv[:,2,:]) 

        # compute sparse matrix product (U @ wood_inv) @ V
        result = np.zeros((nw, d, d), dtype='complex')

        if k == 1:
            result[:,:,2+2*(1)] = np.einsum('ki,k->ki', tmp2[:,:,0], lambdas)
            result[:,:,-1] = np.einsum('ki,k->ki', tmp2[:,:,0], lambdas)

            result[:,:,-3] = np.einsum('ki,k->ki', tmp2[:,:,1], lambdas)
        else:
            result[:,:,2+2*(np.arange(k)+1)] = np.einsum('ki,kj->kij', tmp2[:,:,0], lambdas)
            result[:,:,-1] = np.einsum('ki,k->ki', tmp2[:,:,0], lambdas[:,0])

            result[:,:,-3] = np.einsum('ki,k->ki', tmp2[:,:,1], lambdas[:,0])

        result[:,:,0] = tmp2[:,:,2]

        #################### 

        result = np.einsum('kij,k->ij', result, weights) # (d, d)

        result = Zinv - Zinv @ result @ Zinv

        return result

    def G_xhat(Z):
#         return custom_cauchy_transform_Nd_vectorized(Z, coeff_mtx, density_x, lambdas_x, weights_x)
        return custom_cauchy_transform_Nd_vectorized(Z,  lambdas_x, weights_x)

    def G_yhat(Z):
        return operator_cauchy_transform(Z - B_y, A_y, density_y, lambdas_y, weights_y, atoms = atoms_y)

    return -np.real(get_g_sum(Z, G_xhat, G_yhat))[0]



def compute_trace_rational_type_3(MP_param, z_0, z_1, z_2, z_3, b_0, b_1, b_2, b_3, num_mc=3000  ):
    """
    Normalized trace of
    \[
        (b0 + b1*m)^{-1} (z0 + z1*m + sum gk (z2 + z3*m) gk) (b2 + b3*m)^{-1} (z0 + z1*m + sum gk (z2 + z3*m) gk)
    \]
    where m is Marchenko-Pastur and gk is Gaussian.
    """

    # block matrix size and locations
    lift_dim = 4*(k+1)+5
    rows     = [ 0, 1, 3,         3+2*(k+1), 5+2*(k+1) ] # first index of each block row
    cols     = [ 0, 1, 2*(k+1)+1, 3+2*(k+1), 3+4*(k+1) ] # first index of each block column

    # quadrature and densities
    lambdas_x, weights_x = get_gauss_quad( k, num_mc   )
    lambdas_y, weights_y = get_quad( (1. - np.sqrt(MP_param))**2,  (1. + np.sqrt(MP_param))**2 )
    density_x            = lambda lam: 1.
    density_y            = lambda lam: mp_density(lam, MP_param)
    atoms_y              = ([0.], [1. - 1. / MP_param]) if MP_param>1. else None

    # coefficients of Gaussian diagonals
    coeff_mtx                                                    = np.zeros((lift_dim, lift_dim, k), dtype=complex)
    coeff_mtx[ (rows[4]+3)::2,         cols[0],               :] = np.eye(k)  
    coeff_mtx[  rows[3],              (cols[1]+3):cols[2]:2,  :] = np.eye(k)
    coeff_mtx[ (rows[2]+3):rows[3]:2,  cols[2],               :] = np.eye(k)
    coeff_mtx[  rows[1],              (cols[3]+3):cols[4]:2,  :] = np.eye(k) 

    # A_y: coefficients of m=ΘᵀΘ
    A_y                                   = np.zeros((lift_dim, lift_dim), dtype=complex)
    A_y[rows[4]:,        cols[1]:cols[2]] = np.diag(np.hstack((  [z_1,0],  np.tile([z_3,0], k))  ))
    A_y[rows[3],         cols[2]+1]       = b_3
    A_y[rows[2]:rows[3], cols[3]:cols[4]] = np.diag(np.hstack((  [z_1,0],  np.tile([z_3,0], k))  ))
    A_y[rows[1],         cols[4]+1]       = b_1

    # B_y: constant coefficients
    B_y                                   =   np.zeros((lift_dim, lift_dim), dtype=complex)
    ds                                    = [ np.hstack(([z_0,0], np.tile([z_2,0], k) )), np.tile([0,-1],k+1)[1:], np.tile([0,-1], k+1)[1:]  ]
    B_y[rows[4]:,        cols[1]:cols[2]] = diags( ds, [0,-1,1]  ).toarray()
    B_y[rows[2]:rows[3], cols[3]:cols[4]] = diags( ds, [0,-1,1]  ).toarray()
    B_y[rows[3]:rows[4], cols[2]:cols[3]] = [[0,b_2], [1.,1.]]
    B_y[rows[1]:rows[2], cols[4]:]        = [[0,b_0], [1.,1.]]
    B_y[rows[0]:rows[1], cols[4]:]        =  [1.,0.]
    B_y[rows[1],         cols[3]+1]       = 1.
    B_y[rows[2]+1,       cols[2]]         = 1.
    B_y[rows[3],         cols[1]+1]       = 1.
    B_y[rows[4]+1,       cols[0]]         = 1.

    G_xhat = lambda Z: operator_cauchy_transform_Nd( Z,       coeff_mtx, density_x, lambdas_x, weights_x)
    G_yhat = lambda Z: operator_cauchy_transform(    Z - B_y, A_y,       density_y, lambdas_y, weights_y, atoms = atoms_y)

    Z = np.zeros( (lift_dim, lift_dim) )
    return -np.real(get_g_sum(Z, G_xhat, G_yhat, tol=1e-5, max_iter=2000   ))[0]


def compute_trace_rational_type_4(MP_param, z_0, z_1, z_2, z_3, b_0, b_1, b_2, b_3, num_mc=3000  ):
    """
    Normalized trace of
    \[
        (b0 + b1*m)^{-1} (z0 + z1*m + sum gk (z2 + z3*m) gk) (g_1(b2 + b3*m)g_1)^{-1} (z0 + z1*m + sum gk (z2 + z3*m) gk)
    \]
    where m is Marchenko-Pastur and gk is Gaussian.
    """

    # block matrix size and locations
    lift_dim = 4*(k+1)+6
    rows     = [ 0, 1, 3,         3+2*(k+1), 6+2*(k+1) ] # first index of each block row
    cols     = [ 0, 1, 2*(k+1)+1, 4+2*(k+1), 4+4*(k+1) ] # first index of each block column

    # quadrature and densities
    lambdas_x, weights_x = get_gauss_quad( k, num_mc   )
    lambdas_y, weights_y = get_quad( (1. - np.sqrt(MP_param))**2,  (1. + np.sqrt(MP_param))**2 )
    density_x            = lambda lam: 1.
    density_y            = lambda lam: mp_density(lam, MP_param)
    atoms_y              = ([0.], [1. - 1. / MP_param]) if MP_param>1. else None

    # coefficients of Gaussian diagonals
    coeff_mtx                                                    = np.zeros((lift_dim, lift_dim, k), dtype=complex)
    coeff_mtx[ (rows[4]+3)::2,         cols[0],               :] = np.eye(k)  
    coeff_mtx[  rows[3],              (cols[1]+3):cols[2]:2,  :] = np.eye(k)
    coeff_mtx[ (rows[2]+3):rows[3]:2,  cols[2],               :] = np.eye(k)
    coeff_mtx[  rows[1],              (cols[3]+3):cols[4]:2,  :] = np.eye(k) 
    coeff_mtx[  rows[3]:rows[4],       cols[2]:cols[3],       0] = [[0,0,1.],[0,0,0],[1.,0,0]]

    # A_y: coefficients of m=ΘᵀΘ
    A_y                                   =  np.zeros((lift_dim, lift_dim), dtype=complex)
    A_y[rows[4]:,        cols[1]:cols[2]] =  np.diag(np.hstack((  [z_1,0],  np.tile([z_3,0], k))  ))
    A_y[rows[3]+1,       cols[2]+1]       = -b_3
    A_y[rows[2]:rows[3], cols[3]:cols[4]] =  np.diag(np.hstack((  [z_1,0],  np.tile([z_3,0], k))  ))
    A_y[rows[1],         cols[4]+1]       =  b_1

    # B_y: constant coefficients
    B_y                                   =   np.zeros((lift_dim, lift_dim), dtype=complex)
    ds                                    = [ np.hstack(([z_0,0], np.tile([z_2,0], k) )), np.tile([0,-1],k+1)[1:], np.tile([0,-1], k+1)[1:]  ]
    B_y[rows[4]:,        cols[1]:cols[2]] = diags( ds, [0,-1,1]  ).toarray()
    B_y[rows[2]:rows[3], cols[3]:cols[4]] = diags( ds, [0,-1,1]  ).toarray()
    B_y[rows[3]:rows[4], cols[2]:cols[3]] = [[0,0,0],[0,-b_2,1.], [0,1.,0]]
    B_y[rows[1]:rows[2], cols[4]:]        = [[0,b_0], [1.,1.]]
    B_y[rows[0]:rows[1], cols[4]:]        =  [1.,0.]
    B_y[rows[1],         cols[3]+1]       = 1.
    B_y[rows[2]+1,       cols[2]]         = 1.
    B_y[rows[3],         cols[1]+1]       = 1.
    B_y[rows[4]+1,       cols[0]]         = 1.

    G_xhat = lambda Z: operator_cauchy_transform_Nd( Z,       coeff_mtx, density_x, lambdas_x, weights_x)
    G_yhat = lambda Z: operator_cauchy_transform(    Z - B_y, A_y,       density_y, lambdas_y, weights_y, atoms = atoms_y)

    Z = np.zeros( (lift_dim, lift_dim) )
    return -np.real(get_g_sum(Z, G_xhat, G_yhat, tol=1e-5, max_iter=2000))[0]



def compute_trace_rational_type_5(MP_param, z_0, z_1, z_2, z_3, b_0, b_1, b_2, b_3, num_mc=3000, same_gauss=True  ):
    """
    Normalized trace of
    \[
        (g_i(b0 + b1*m)g_i)^{-1} (z0 + z1*m + sum gk (z2 + z3*m) gk) (g_j(b2 + b3*m)g_j)^{-1} (z0 + z1*m + sum gk (z2 + z3*m) gk)
    \]
    where m is Marchenko-Pastur and gk is Gaussian. If same_gauss=True (default), then i=j. 
    """

    # block matrix size and locations
    lift_dim = 4*(k+1)+7
    rows     = [ 0, 1, 4,         4+2*(k+1), 7+2*(k+1) ] # first index of each block row
    cols     = [ 0, 1, 2*(k+1)+1, 4+2*(k+1), 4+4*(k+1) ] # first index of each block column
    j        = 0 if (same_gauss or k<2) else 1

    # quadrature and densities
    lambdas_x, weights_x = get_gauss_quad( k, num_mc   )
    lambdas_y, weights_y = get_quad( (1. - np.sqrt(MP_param))**2,  (1. + np.sqrt(MP_param))**2 )
    density_x            = lambda lam: 1.
    density_y            = lambda lam: mp_density(lam, MP_param)
    atoms_y              = ([0.], [1. - 1. / MP_param]) if MP_param>1. else None

    # coefficients of Gaussian diagonals
    coeff_mtx                                                    = np.zeros((lift_dim, lift_dim, k), dtype=complex)
    coeff_mtx[ (rows[4]+3)::2,         cols[0],               :] = np.eye(k)  
    coeff_mtx[  rows[3],              (cols[1]+3):cols[2]:2,  :] = np.eye(k)
    coeff_mtx[ (rows[2]+3):rows[3]:2,  cols[2],               :] = np.eye(k)
    coeff_mtx[  rows[1],              (cols[3]+3):cols[4]:2,  :] = np.eye(k) 
    coeff_mtx[  rows[3]:rows[4],       cols[2]:cols[3],       0] = [[0,0,1.],[0,0,0],[1.,0,0]]
    coeff_mtx[  rows[1]:rows[2],       cols[4]:,              j] = [[0,0,1.],[0,0,0],[1.,0,0]]

    # A_y: coefficients of m=ΘᵀΘ
    A_y                                   =  np.zeros((lift_dim, lift_dim), dtype=complex)
    A_y[rows[4]:,        cols[1]:cols[2]] =  np.diag(np.hstack((  [z_1,0],  np.tile([z_3,0], k))  ))
    A_y[rows[3]+1,       cols[2]+1]       = -b_3
    A_y[rows[2]:rows[3], cols[3]:cols[4]] =  np.diag(np.hstack((  [z_1,0],  np.tile([z_3,0], k))  ))
    A_y[rows[1]+1,       cols[4]+1]       = -b_1

    # B_y: constant coefficients
    B_y                                   =   np.zeros((lift_dim, lift_dim), dtype=complex)
    ds                                    = [ np.hstack(([z_0,0], np.tile([z_2,0], k) )), np.tile([0,-1],k+1)[1:], np.tile([0,-1], k+1)[1:]  ]
    B_y[rows[4]:,        cols[1]:cols[2]] = diags( ds, [0,-1,1]  ).toarray()
    B_y[rows[2]:rows[3], cols[3]:cols[4]] = diags( ds, [0,-1,1]  ).toarray()
    B_y[rows[3]:rows[4], cols[2]:cols[3]] = [[0,0,0],[0,-b_2,1.], [0,1.,0]]
    B_y[rows[1]:rows[2], cols[4]:]        = [[0,0,0],[0,-b_0,1.], [0,1.,0]]
    B_y[rows[0]:rows[1], cols[4]:]        =  [1.,0.,0.]
    B_y[rows[1],         cols[3]+1]       = 1.
    B_y[rows[2]+1,       cols[2]]         = 1.
    B_y[rows[3],         cols[1]+1]       = 1.
    B_y[rows[4]+1,       cols[0]]         = 1.

    G_xhat = lambda Z: operator_cauchy_transform_Nd( Z,       coeff_mtx, density_x, lambdas_x, weights_x)
    G_yhat = lambda Z: operator_cauchy_transform(    Z - B_y, A_y,       density_y, lambdas_y, weights_y, atoms = atoms_y)

    Z = np.zeros( (lift_dim, lift_dim) )
    return -np.real(get_g_sum(Z, G_xhat, G_yhat, tol=1e-6, max_iter=1500))[0]



###############################################################

def update_V_hat(V, alpha):
    ret = np.zeros(2)
    ret[0] = alpha / (1. + V[0])
    ret[1] = alpha * tau / (1. + tau * V[1])
    return ret

def compute_rhs_Va(alpha, gamma, V_hat, settings):
    # Evaluate 1/p trace(Ainv * M00) via operator valued Cauchy trafo

    k1, ks = settings.k1, settings.ks
    dk1, dks = settings.dk1, settings.dks

    MP_param = 1 / gamma # p / d
    
    lbda = args.lbda / alpha
    z_0 = alpha*lbda + V_hat[0] * ks**2 
    z_1 = V_hat[0] * k1**2
    z_2 = V_hat[1] * dks**2
    z_3 = V_hat[1] * dk1**2

    b_0 = ks**2
    b_1 = k1**2

    rhs_Va = compute_trace_rational_type_1(MP_param, z_0, z_1, z_2, z_3, b_0, b_1)

    return rhs_Va

def compute_rhs_Vc(alpha, gamma, V_hat, settings):
    # Evaluate 1/p trace(Ainv * (D1 @ M00 @ D1)) via operator valued Cauchy trafo

    k1, ks = settings.k1, settings.ks
    dk1, dks = settings.dk1, settings.dks

    MP_param = 1 / gamma # p / d
    
    lbda = args.lbda / alpha
    z_0 = alpha*lbda + V_hat[0] * ks**2 
    z_1 = V_hat[0] * k1**2
    z_2 = V_hat[1] * dks**2
    z_3 = V_hat[1] * dk1**2

    b_0 = dks**2
    b_1 = dk1**2

    rhs_Vc = compute_trace_rational_type_2(MP_param, z_0, z_1, z_2, z_3, b_0, b_1)

    return rhs_Vc

######################################################

def compute_rhs_f_mc(alpha, gamma, V_hat, f_hat, settings, num_mc=30, ):

    k1, ks = settings.k1, settings.ks
    dk1, dks = settings.dk1, settings.dks

    d = settings.d
    p    = int(settings.d/gamma)

    rhs_f = np.zeros_like(f_hat)

    np.random.seed(seed) # fix seed to make convergence faster
    for _ in range(num_mc):

        Theta = np.random.randn( d, p ) / np.sqrt(d)
        ThTh  = Theta.T @ Theta

        zeta  = np.random.randn(p, k)

        M00 = k1**2 * ThTh + ks**2 * np.eye(p)
        M11 = dk1**2 * ThTh + dks**2 * np.eye(p)

        DkM11Dk = np.zeros( (p,p) )
        for kk in range(k):
            Dk = np.diag(zeta[:,kk])
            DkM11Dk += Dk @ M11 @ Dk
        lbda = args.lbda / alpha
        Ainv = np.linalg.inv(alpha*lbda*np.eye(p) + V_hat[0]*M00 + V_hat[1] * DkM11Dk)
        
        ###################

        # updates
        rhs_f[0] += 1/d * k1**2 * np.trace( Ainv @ ThTh ) * f_hat[0] 
       
        D0 = np.diag(zeta[:,0]) # by symmetry, suffices to randomly pick one zeta
        rhs_f[1] += 1/d * dk1**2 * np.trace( Ainv @ D0 @ ThTh @ D0 ) * f_hat[1] 

    rhs_f = rhs_f / num_mc

    return rhs_f

def compute_rhs_fa(alpha, gamma, V_hat, f_hat, settings):
    # Evaluate 1/p trace(Ainv * M00) via operator valued Cauchy trafo

    k1, ks = settings.k1, settings.ks
    dk1, dks = settings.dk1, settings.dks

    MP_param = 1 / gamma # p / d
    
    lbda = args.lbda / alpha
    z_0 = alpha*lbda + V_hat[0] * ks**2 
    z_1 = V_hat[0] * k1**2
    z_2 = V_hat[1] * dks**2
    z_3 = V_hat[1] * dk1**2

    b_0 = 0.
    b_1 = k1**2

    rhs_fa = compute_trace_rational_type_1(MP_param, z_0, z_1, z_2, z_3, b_0, b_1)

    # scale by gamma since rhs_fa is normalized by d and not p
    rhs_fa = 1 / gamma * rhs_fa * f_hat[0] 
 
    return rhs_fa

def compute_rhs_fb(alpha, gamma, V_hat, f_hat, settings):
    # Evaluate 1/p trace(Ainv * (D1 @ M00 @ D1)) via operator valued Cauchy trafo

    k1, ks = settings.k1, settings.ks
    dk1, dks = settings.dk1, settings.dks

    MP_param = 1 / gamma # p / d
    
    lbda = args.lbda / alpha
    z_0 = alpha*lbda + V_hat[0] * ks**2 
    z_1 = V_hat[0] * k1**2
    z_2 = V_hat[1] * dks**2
    z_3 = V_hat[1] * dk1**2

    b_0 = 0. 
    b_1 = dk1**2

    rhs_fb= compute_trace_rational_type_2(MP_param, z_0, z_1, z_2, z_3, b_0, b_1)

    # scale by gamma since rhs_fa is normalized by d and not p
    rhs_fb = 1 / gamma * rhs_fb * f_hat[1]

    return rhs_fb

######################################################


def compute_q_q_hat_mc(alpha, gamma, V_hat, f, f_hat, settings, num_mc=30, ):
    # solve for (qa, qc, qhata, qhatc)

    k1,  ks  = settings.k1, settings.ks
    dk1, dks = settings.dk1, settings.dks

    d = settings.d
    p = int(settings.d/gamma)

    params_q2 = np.zeros( 4 )
    params_q0 = np.zeros( 4 )

    np.random.seed(seed) # fix seed to make convergence faster
    for _ in range(num_mc):

        Theta = np.random.randn( d, p ) / np.sqrt(d)
        ThTh  = Theta.T @ Theta

        zeta  = np.random.randn(p, k)

        M00 = k1**2 * ThTh + ks**2 * np.eye(p)
        M11 = dk1**2 * ThTh + dks**2 * np.eye(p)

        DkM11Dk = np.zeros((p,p))
        for kk in range(k):
            Dk = np.diag(zeta[:,kk])
            DkM11Dk += Dk @ M11 @ Dk
        lbda = args.lbda / alpha
        Ainv = np.linalg.inv(alpha*lbda*np.eye(p) + V_hat[0]*M00 + V_hat[1] * DkM11Dk)

        ###################

        A00A = Ainv @ M00 @ Ainv

        ADkM11DkA = Ainv @ DkM11Dk @ Ainv


        D0M11D0 = np.diag(zeta[:,0]) @ M11 @ np.diag(zeta[:,0])
        #D1M11D1 = np.diag(zeta[:,1]) @ M11 @ np.diag(zeta[:,1])
        AD0M11D0A = Ainv @ D0M11D0 @ Ainv
        #AD1M11D1A = Ainv @ D1M11D1 @ Ainv

        AThThA = Ainv @ ThTh @ Ainv

        ADkThThDkA = np.zeros((p,p))
        for kk in range(k):
            Dk = np.diag(zeta[:,kk])
            ADkThThDkA += Dk @ ThTh @ Dk
        ADkThThDkA = Ainv @ ADkThThDkA @ Ainv

        AD0ThThD0A = Ainv @ np.diag(zeta[:,0]) @ ThTh @ np.diag(zeta[:,0]) @ Ainv
        #AD1ThThD1A = Ainv @ np.diag(zeta[:,1]) @ ThTh @ np.diag(zeta[:,1]) @ Ainv


        ##################

        # [q_a, q_c, qhat_a, qhat_c]
        # 
        # This matrix is the same regardless of order of coefficient.
        # Only the rhs vector b changes.

        Aq = np.zeros( (4, 4) )

        Aq[0] = np.array( [ 1,                     0,                  -np.trace(A00A @ M00)/p,           (1/k) * -np.trace(ADkM11DkA @ M00 )/p ] )
        Aq[1] = np.array( [ 0,                     1,                  k * -np.trace(A00A @ (DkM11Dk/k)) / p, -np.trace(ADkM11DkA @ (DkM11Dk/k)) / p] )
        Aq[2] = np.array( [-V_hat[0]**2 / alpha,   0,                    1,                               0] )
        Aq[3] = np.array( [ 0,                    -V_hat[1]**2 / alpha,  0,                               1] )
    
        ###################
        # solve for q0

        b0 = np.zeros( 4 )
        b0[0] = k1**2 / gamma * f_hat[0]**2 * np.trace(AThThA @ M00) / p 
        b0[1] = k * k1**2 / gamma * f_hat[0]**2 * np.trace(AThThA @ (DkM11Dk/k)) / p
        b0[2] = V_hat[0]**2 * (noise_cov[0,0] + settings.var_phi + (abs(settings.k0)<=1e-8)*settings.E_phi**2) / alpha - 2*V_hat[0]*f[0]*f_hat[0]/alpha
        b0[3] = V_hat[1]**2 * np.trace(noise_cov[1:,1:]) / alpha

        params_q0 += np.linalg.solve(Aq, b0)
 
        ###################
        # solve for q2

        bv = np.zeros( 4 )
        bv[0] = dk1**2 / gamma * f_hat[1]**2 * (1/k) * np.trace(ADkThThDkA @ M00) / p
        bv[1] = dk1**2 / gamma * f_hat[1]**2 * np.trace(ADkThThDkA @ (DkM11Dk/k)) / p
        bv[2] = 0.
        bv[3] = V_hat[1]**2 * (settings.var_dphi + (abs(settings.dk0)<=1e-8)*settings.E_dphi**2)/ alpha - 2*V_hat[1]*f[1]*f_hat[1] / alpha 
    
        params_q2 += np.linalg.solve(Aq, bv)

    params_q2 = params_q2 / num_mc
    params_q0 = params_q0 / num_mc

    #######################

    chunks = (2,2)
    q2, q2_hat = slice_by_lengths(params_q2, chunks)
    q0, q0_hat = slice_by_lengths(params_q0, chunks)

    return q2, q0, q2_hat, q0_hat


def compute_q_q_hat(alpha, gamma, V_hat, f, f_hat, settings ):
    # solve for (qa, qc, qhata, qhatc)

    k1,  ks  = settings.k1, settings.ks
    dk1, dks = settings.dk1, settings.dks

    MP_param = 1 / gamma # p / d
    
    lbda = args.lbda / alpha
    z_0 = alpha*lbda + V_hat[0] * ks**2
    z_1 = V_hat[0] * k1**2
    z_2 = V_hat[1] * dks**2
    z_3 = V_hat[1] * dk1**2

    t_AM00_AM00          = compute_trace_rational_type_3(MP_param, z_0, z_1, z_2, z_3,  ks**2,   k1**2,  ks**2,   k1**2)
    t_AD1M11D1_AM00      = compute_trace_rational_type_4(MP_param, z_0, z_1, z_2, z_3,  ks**2,   k1**2, dks**2,  dk1**2)
    t_AD1M11D1A_D1M11D1  = compute_trace_rational_type_5(MP_param, z_0, z_1, z_2, z_3, dks**2,  dk1**2, dks**2,  dk1**2)
    t_AThThA_M00         = compute_trace_rational_type_3(MP_param, z_0, z_1, z_2, z_3,  ks**2,   k1**2,      0,       1)
    t_AThThA_D1M11D1     = compute_trace_rational_type_4(MP_param, z_0, z_1, z_2, z_3,      0,       1, dks**2,  dk1**2)
    t_AD1ThThD1A_M00     = compute_trace_rational_type_4(MP_param, z_0, z_1, z_2, z_3,  ks**2,   k1**2,      0,       1) 
    t_AD1ThThD1A_D1M11D1 = compute_trace_rational_type_5(MP_param, z_0, z_1, z_2, z_3,      0,       1, dks**2,  dk1**2)
    t_AD1M11D1A_D2M11D2  = compute_trace_rational_type_5(MP_param, z_0, z_1, z_2, z_3, dks**2,  dk1**2, dks**2,  dk1**2, same_gauss=False) if k>1 else t_AD1M11D1A_D1M11D1
    t_AD1ThThD1A_D2M11D2 = compute_trace_rational_type_5(MP_param, z_0, z_1, z_2, z_3,      0,       1, dks**2,  dk1**2, same_gauss=False) if k>1 else t_AD1ThThD1A_D1M11D1

    #################
    Aq    = np.zeros( (4, 4) )
    Aq[0] = np.array( [ 1,                     0,                  -t_AM00_AM00,         -t_AD1M11D1_AM00  ] )
    Aq[1] = np.array( [ 0,                     1,                  -k*t_AD1M11D1_AM00,   -t_AD1M11D1A_D1M11D1 -((k**2-k)/k)*t_AD1M11D1A_D2M11D2] )
    Aq[2] = np.array( [-V_hat[0]**2 / alpha,   0,                    1,                               0] )
    Aq[3] = np.array( [ 0,                    -V_hat[1]**2 / alpha,  0,                               1] )

    ###################
    # solve for q0

    b0    = np.zeros( 4 )
    b0[0] = k1**2 / gamma * f_hat[0]**2 * t_AThThA_M00
    b0[1] = k * k1**2 / gamma * f_hat[0]**2 * t_AThThA_D1M11D1
    b0[2] = V_hat[0]**2 * (noise_cov[0,0] + settings.var_phi + (abs(settings.k0)<=1e-8)*settings.E_phi**2) / alpha - 2*V_hat[0]*f[0]*f_hat[0]/alpha
    b0[3] = V_hat[1]**2 * np.trace(noise_cov[1:,1:]) / alpha

    params_q0 = np.linalg.solve(Aq, b0)
 
    ###################
    # solve for q2

    bv    = np.zeros( 4 )
    bv[0] = dk1**2 / gamma * f_hat[1]**2 * t_AD1ThThD1A_M00
    bv[1] = dk1**2 / gamma * f_hat[1]**2 * ( t_AD1ThThD1A_D1M11D1 + ((k**2-k)/k)*t_AD1ThThD1A_D2M11D2  )
    bv[2] = 0.
    bv[3] = V_hat[1]**2 * (settings.var_dphi + (abs(settings.dk0)<=1e-8)*settings.E_dphi**2)/ alpha - 2*V_hat[1]*f[1]*f_hat[1] / alpha 
    
    params_q2 = np.linalg.solve(Aq, bv)

    #######################

    chunks = (2,2)
    q2, q2_hat = slice_by_lengths(params_q2, chunks)
    q0, q0_hat = slice_by_lengths(params_q0, chunks)

    return q2, q0, q2_hat, q0_hat




def eval_dPsiwbeta_mc( alpha, gamma, f_hat, q2_hat, q0_hat, V_hat, settings, num_mc=30, ):
    k1, ks = settings.k1, settings.ks
    dk1, dks = settings.dk1, settings.dks

    d    = settings.d
    p    = int(settings.d/gamma)
    if args.verbose:
        print(f"d = {d}, p = {p}")

    dPsiwbeta = 0.
    for _ in range(num_mc):

        Theta = np.random.randn( d, p ) / np.sqrt(d)
        ThTh  = Theta.T @ Theta

        zeta  = np.random.randn(p, k)

        M00 = k1**2 * ThTh + ks**2 * np.eye(p)
        M11 = dk1**2 * ThTh + dks**2 * np.eye(p)

        DkM11Dk = np.zeros((p,p))
        for kk in range(k):
            Dk = np.diag(zeta[:,kk])
            DkM11Dk += Dk @ M11 @ Dk
        lbda = args.lbda / alpha
        Ainv = np.linalg.inv(alpha*lbda*np.eye(p) + V_hat[0]*M00 + V_hat[1] * DkM11Dk)

        ##########
        mask_0 = np.zeros(k+1)
        mask_0[1] = 1.

        Xi_0  = k1**2 / gamma * f_hat[0]**2 * ThTh 
        Xi_0 += q0_hat[0]*M00 + q0_hat[1]*DkM11Dk
        
        lbda = args.lbda / alpha
        dPsiwbeta += -alpha * lbda / 2 / p * np.trace(Ainv @ Xi_0 @ Ainv)

    dPsiwbeta /= num_mc

    return dPsiwbeta 

def eval_dPsiybeta(alpha, gamma, f_hat, q2_hat, q0_hat, V_hat, settings, ):

    dPsiybetaL2_0 = -0.5 * q0_hat[0]
    dPsiybetaL2_2 = -0.5 * q2_hat[0]

    dPsiybetaH1k_0 = -0.5 * q0_hat[1]
    dPsiybetaH1k_2 = -0.5 * q2_hat[1]

    return dPsiybetaL2_0, dPsiybetaL2_2, dPsiybetaH1k_0, dPsiybetaH1k_2 

def report( f, q2, q0, V, headline):

    match headline:
        case 'hat':
            print('Conjugate parameters:')
            print('f_hat  =', f )
            print('q2_hat =', q2 )
            print('q0_hat =', q0 )
            print('V_hat  =', V )
            print('##########################')
        case 'residual':
            print('Mismatch from eqs:')
            print('res_f  =', f )
            print('res_q2 =', q2 )
            print('res_q0 =', q0 )
            print('res_V  =', V )
            print('##########################')
        case 'final':
            print('##########################')
            print('Found the following parameters:')
            print('f  =', f)
            print('q2 =', q2)
            print('q0 =', q0)
            print('V  =', V)
            print('##########################')
        case _:
            print('Current parameters:')
            print('f  =', f)
            print('q2 =', q2)
            print('q0 =', q0)
            print('V  =', V)
            print('##########################')

def reparam(V):
    # Reparameterize V to "precondition" fixed point system.
    # Since the root finding succeeds fairly easily for small V,
    # it is sensible to choose a parameterization that is locally
    # linear around zero.

#     Vtilde = np.arctan(V)
    Vtilde = np.log(1.+V)
    # ~ Vtilde = V

    return Vtilde

def inv_reparam(Vtilde):
#     V = np.tan(Vtilde)
    V = np.exp(Vtilde) - 1.
    # ~ V = Vtilde

    return V

def eq_sys_V(params_V, alpha, gamma, settings, ):

    V = params_V[:2]

    V_hat = update_V_hat(inv_reparam(V), alpha)

    rhs_Va = reparam(compute_rhs_Va(alpha, gamma, V_hat, settings, ))
    rhs_Vc = reparam(compute_rhs_Vc(alpha, gamma, V_hat, settings, ))

    if args.verbose:
        print(f"V = {V}")
        print(f"V_hat = {V_hat}")

    #####################

    ret_V = np.zeros_like(params_V)
    ret_V[0] = rhs_Va - V[0]
    ret_V[1] = rhs_Vc - V[1]

    if args.verbose:
        print('Mismatch from eqs:')
        print('ret_V   =', ret_V)
        print('##########################')

    return ret_V

###############################################################

def damped_fixed_point(params_V, eq_sys, alpha, gamma, settings, tol=1e-3, max_iter=500, damp=0.2 ):

    params_old = params_V
#     for kk in range(max_iter):
    for kk in tqdm.tqdm(range(max_iter)):
        # eq_sys returns the _residual_
        params_new = params_old + eq_sys(params_old, alpha, gamma, settings, )

        res = np.linalg.norm(params_new - params_old)
        if res < tol and args.verbose:
            print(f"Converged after {kk} steps")
            break
        else:
            params_old = damp * params_new + (1. - damp) * params_old

    if kk == max_iter-1: #and args.verbose:
        print(f"-W- Failed to converge after {max_iter} steps")

    return params_new

###############################################################

def get_analytical_error_results(params_V, alpha, gamma, settings):
    
    print('params_V =', params_V)

    if args.root_alg == "excitingmixing" or args.root_alg == "lm" :
        options = {
            'ftol': 1e-2,  # Relative tolerance for the residual (function value)
            'xtol': 1e-2,  # Relative tolerance for the solution vector x
#             'maxiter': 200 # Maximum iterations
        }

        res = root(eq_sys_V, x0 = params_V, tol=args.root_tol, 
                args = (alpha, gamma, settings, ), 
                method = args.root_alg, 
#                 options = options,
                )

        V = res.x

        success = res.success
        print("Solution:", res.x)
        print("Function value at solution:", res.fun)
        print(f"Success: {res.success}, {res.message}")
        print("No. of iterations needed:", res.nit)

    elif args.root_alg == "fixedpoint":
        V = damped_fixed_point(params_V, eq_sys_V, alpha, gamma, settings, )

    else:
        raise NotImplementedError(f'-E- root_alg = {args.root_alg} invalid')

    V_hat = update_V_hat(inv_reparam(V), alpha)

    ################################

    f_hat = V_hat * np.hstack( (settings.E_dphi, settings.E_ddphi ) )

    f = np.zeros_like(f_hat)
    f[0] = compute_rhs_fa(alpha, gamma, V_hat, f_hat, settings )
    f[1] = compute_rhs_fb(alpha, gamma, V_hat, f_hat, settings )

    ################################
    
    #q2, q0, q2_hat, q0_hat = compute_q_q_hat_mc(alpha, gamma, V_hat, f, f_hat, settings, )
    q2, q0, q2_hat, q0_hat = compute_q_q_hat(alpha, gamma, V_hat, f, f_hat, settings, )

    ################################

    params = np.hstack( (f, q2, q0, inv_reparam(V)) )
    params_hat = np.hstack( (f_hat, q2_hat, q0_hat, V_hat) )

#     if args.verbose:
    report( f, q2, q0, inv_reparam(V), 'final' )
    report( f_hat, q2_hat, q0_hat, V_hat, 'hat' )

    ################################

    ### compute training error via MC at varpi=0
    dPsiybetaL2_0, dPsiybetaL2_2, dPsiybetaH1k_0, dPsiybetaH1k_2 = eval_dPsiybeta(alpha, gamma, f_hat, q2_hat, q0_hat, V_hat, settings)

#     eps_train = -(dPsiybetaL2 + dPsiybetaH1k + dPsiwbeta)/alpha 

    eps_train_L2_0 = -dPsiybetaL2_0 / alpha
    eps_train_L2_2 = -dPsiybetaL2_2 / alpha

    eps_train_H1k_0 = -dPsiybetaH1k_0 / alpha
    eps_train_H1k_2 = -dPsiybetaH1k_2 / alpha

    # compute testing error at varpi=0
    eps_test_L2_0   = noise_cov[0,0] + q0[0] + settings.var_phi + (abs(settings.k0)<=1e-8)*settings.E_phi**2   - 2*settings.E_dphi*f[0]
    eps_test_L2_2   = q2[0]

    eps_test_H1k_0  = np.trace(noise_cov[1:,1:]) + q0[1]
    eps_test_H1k_2  = settings.var_dphi + (abs(settings.dk0)<=1e-8)*settings.E_dphi**2  - 2*f[1]*settings.E_ddphi + q2[1]

    #################

#     if args.verbose:
    print('Training error (L2,  varpi=0)  =', eps_train_L2_0)
    print('Training error (H1k, varpi=0)  =', eps_train_H1k_0)

    print('Testing  error (L2,  varpi=0)  =', eps_test_L2_0)
    print('Testing  error (H1k, varpi=0)  =', eps_test_H1k_0)

    ret = [eps_train_L2_0, eps_train_L2_2, eps_train_H1k_0, eps_train_H1k_2, eps_test_L2_0, eps_test_L2_2, eps_test_H1k_0, eps_test_H1k_2], params, params_hat

    return ret

###############################################################

def scan_over_pn_ratios(_args, save = False):
    
    r_pn = _args[0]
    r_pn_num = len(r_pn)
    r_nd = _args[1]
    
    ################################

    settings = SimpleNamespace()

    settings.k0 = get_k0(sigma)
    settings.k1 = get_k1(sigma)
    settings.ks = get_ks(sigma, settings.k0, settings.k1)

    settings.dk0 = get_k0(sigma_prime)
    settings.dk1 = get_k1(sigma_prime)
    settings.dks = get_ks(sigma_prime, settings.dk0, settings.dk1)

    # -------- condition on Vk and theta0

    settings.d = d

    # precompute quadratures
    Z, w_Z = make_gauss_quad1D(150, mu=0., var=1.)

    E_phi = np.dot(w_Z, phi(Z))
    E_dphi = np.dot(w_Z, d_phi(Z))
    E_ddphi = np.dot(w_Z, ddphi(Z))

    var_phi  = np.dot(w_Z, phi(Z)**2) - E_phi**2
    var_dphi = np.dot(w_Z, d_phi(Z)**2) - E_dphi**2

    var_phi_dphi = np.dot(w_Z, (phi(Z)-E_phi)*(d_phi(Z)-E_dphi))

    settings.E_phi = E_phi 
    settings.E_dphi  = E_dphi 
    settings.E_ddphi = E_ddphi

    settings.var_phi  = var_phi
    settings.var_dphi = var_dphi

    settings.var_phi_dphi = var_phi_dphi
    
    ################################

    # [Va, Vc, Va_hat, Vc_hat]
    params_V = 0.01 * np.ones(2) 
    
    ################################

    ratio_pn = r_pn[r_pn != k+1] # do not sample at phase transition point

    if save:
        now = datetime.now()
        dt_string = now.strftime('%Y_%m_%d_%H_%M_%S')
        data_dir = 'data/{}/{}'.format(base_dir, dt_string)
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

    ################################

    ratio_pn_num = len(ratio_pn) # not necessarily equal to r_pn_num

    # save summary statistics separated into varpi components
    # eps_total = eps_0 + eps_2 * varpi**2 
    eps_train_L2_0, eps_train_L2_2   = np.zeros(ratio_pn_num), np.zeros(ratio_pn_num)
    eps_train_H1k_0, eps_train_H1k_2 = np.zeros(ratio_pn_num), np.zeros(ratio_pn_num)
    eps_test_L2_0, eps_test_L2_2     = np.zeros(ratio_pn_num), np.zeros(ratio_pn_num)
    eps_test_H1k_0, eps_test_H1k_2   = np.zeros(ratio_pn_num), np.zeros(ratio_pn_num)

    ################################
    params_saved = np.zeros( (ratio_pn_num, 8) )
    params_hat_saved = np.zeros( (ratio_pn_num, 8) )

    for i in tqdm.tqdm(range(ratio_pn_num)):

        print('ratio_pn = {}'.format(ratio_pn[i]))
        alpha = 1. / ratio_pn[i]
        gamma = 1. / (r_nd * ratio_pn[i])

        ret, params, params_hat = get_analytical_error_results(params_V, alpha, gamma, settings,) 

        _,_,_,V = unpack(params) 
        params_V = reparam(V) # initialize from previous soln

        eps_train_L2_0[i]  = ret[0]
        eps_train_L2_2[i]  = ret[1]
        eps_train_H1k_0[i] = ret[2]
        eps_train_H1k_2[i] = ret[3]
        eps_test_L2_0[i]   = ret[4]
        eps_test_L2_2[i]   = ret[5]
        eps_test_H1k_0[i]  = ret[6]
        eps_test_H1k_2[i]  = ret[7]

        params_saved[i] = params
        params_hat_saved[i] = params_hat

    if save:
        
        np.save(data_dir + '/lbda.npy', args.lbda)
        np.save(data_dir + '/Delta.npy', delta)
        np.save(data_dir + '/sigma.npy', sigma_str)
        np.save(data_dir + '/nonlin.npy', nonlin_str)
        np.save(data_dir + '/theta.npy', rf_str)
        np.save(data_dir + '/ratio-nd.npy', r_nd)
        np.save(data_dir + '/ratio-pn.npy', ratio_pn)
        np.save(data_dir + '/tau.npy', tau)

        np.save(data_dir + '/eps-train-L2-0.npy', eps_train_L2_0)
        np.save(data_dir + '/eps-train-L2-2.npy', eps_train_L2_2)
        np.save(data_dir + '/eps-train-H1k-0.npy', eps_train_H1k_0)
        np.save(data_dir + '/eps-train-H1k-2.npy', eps_train_H1k_2)
        np.save(data_dir + '/eps-test-L2-0.npy', eps_test_L2_0)
        np.save(data_dir + '/eps-test-L2-2.npy', eps_test_L2_2)
        np.save(data_dir + '/eps-test-H1k-0.npy', eps_test_H1k_0)
        np.save(data_dir + '/eps-test-H1k-2.npy', eps_test_H1k_2)

        np.save(data_dir + '/params.npy', params_saved)
        np.save(data_dir + '/params-hat.npy', params_hat_saved)
        np.save(data_dir + '/k.npy', k)
        np.save(data_dir + '/dims.npy', d)

        plt.figure()
        labels=[]
        plt.plot(ratio_pn, eps_train_L2_0, '-',  color = 'steelblue')
        labels.append(r'$\varepsilon_{L2,\mathrm{train}}$')
        plt.plot(ratio_pn, eps_train_H1k_0, '-', color = 'green')
        labels.append(r'$\varepsilon_{H1k,\mathrm{train}}$')
        plt.plot(ratio_pn, eps_test_L2_0,  '-', color = 'orange')
        labels.append(r'$\varepsilon_{L2, \mathrm{test}}$' )
        plt.plot(ratio_pn, eps_test_H1k_0,  '-', color = 'purple')
        labels.append(r'$\varepsilon_{H1k, \mathrm{test}}$' )
        plt.grid()
        plt.xlabel(r'$\alpha^{-1} = p / n$')
        plt.ylabel(r'$\varepsilon$')
        plt.legend(labels)
        plt.title(r'\begin{{center}}$\lambda = {}$, $\Delta = {}$, $\alpha / \gamma = n/d = {}$,\\ $\sigma =$ {}, $\Theta =$ {}, $\phi =$ {}, $\varpi = 0$\end{{center}}'.format(args.lbda, delta, r_nd, sigma_str, rf_str, nonlin_str, ))
        plt.savefig(data_dir + '/eps.pdf', bbox_inches = 'tight')
        plt.close()
    
    # return results
    if ratio_pn_num < r_pn_num:
        idx = np.argmax(ratio_pn > k+1)
        eps_train_L2_0   = np.insert(eps_train_L2_0, idx, 0., axis = 0)
        eps_train_L2_2   = np.insert(eps_train_L2_2, idx, 0., axis = 0)
        eps_train_H1k_0  = np.insert(eps_train_H1k_0, idx, 0., axis = 0)
        eps_train_H1k_2  = np.insert(eps_train_H1k_2, idx, 0., axis = 0)
        eps_test_L2_0    = np.insert(eps_test_L2_0, idx, 0., axis = 0)
        eps_test_L2_2    = np.insert(eps_test_L2_2, idx, 0., axis = 0)
        eps_test_H1k_0   = np.insert(eps_test_H1k_0, idx, 0., axis = 0)
        eps_test_H1k_2   = np.insert(eps_test_H1k_2, idx, 0., axis = 0)
        params_saved     = np.insert(params_saved, idx, np.zeros_like(params), axis = 0)
        params_hat_saved = np.insert(params_hat_saved, idx, np.zeros_like(params_hat), axis = 0)
    
    return eps_train_L2_0,eps_train_L2_2,eps_train_H1k_0,eps_train_H1k_2,eps_test_L2_0,eps_test_L2_2,eps_test_H1k_0,eps_test_H1k_2,params_saved,params_hat_saved

def multiProcessingEvaluation(worker, _args, nProc = None):

    if nProc is None:
        nProcc = max(1, mp.cpu_count() - 2)
    else:
        nProcc = nProc
        
    nr = len(_args)
    eps_train_0 = np.zeros((nr, nr))
    eps_train_2 = np.zeros((nr, nr))
    eps_test_0 = np.zeros((nr, nr))
    eps_test_2 = np.zeros((nr, nr))
    eps_train_H1k_0 = np.zeros((nr, nr))
    eps_train_H1k_2 = np.zeros((nr, nr))
    eps_test_H1k_0 = np.zeros((nr, nr))
    eps_test_H1k_2 = np.zeros((nr, nr))
    params_0 = np.zeros((nr, nr, 8))
    params_hat_0 = np.zeros((nr, nr, 8))

    with mp.get_context("spawn").Pool(processes = nProcc) as pool:

        for i, result in enumerate(tqdm.tqdm(pool.imap(worker, _args), total=nr)):
            
            eps_train_L2_0_tmp = result[0]
            eps_train_L2_2_tmp = result[1]
            eps_train_H1k_0_tmp = result[2]
            eps_train_H1k_2_tmp = result[3]
            eps_test_L2_0_tmp = result[4]
            eps_test_L2_2_tmp = result[5]
            eps_test_H1k_0_tmp = result[6]
            eps_test_H1k_2_tmp = result[7]
            params_0_tmp = result[8]
            params_hat_0_tmp = result[9]
            
            eps_train_0[i] = eps_train_L2_0_tmp
            eps_train_2[i] = eps_train_L2_2_tmp
            eps_test_0[i] = eps_test_L2_0_tmp
            eps_test_2[i] = eps_test_L2_2_tmp
            eps_train_H1k_0[i] = eps_train_H1k_0_tmp
            eps_train_H1k_2[i] = eps_train_H1k_2_tmp
            eps_test_H1k_0[i] = eps_test_H1k_0_tmp
            eps_test_H1k_2[i] = eps_test_H1k_2_tmp
            params_0[i] = params_0_tmp
            params_hat_0[i] = params_hat_0_tmp

    return eps_train_0, eps_train_2,eps_test_0,eps_test_2,eps_train_H1k_0,eps_train_H1k_2,eps_test_H1k_0,eps_test_H1k_2,params_0,params_hat_0


def plot_2d_generalization_landscape(rmin = -2., rmax = 3., nr = 64, nProc = 8):
    
    ratios_nd = np.logspace(rmin, rmax, nr)
    ratios_pd = np.logspace(rmin, rmax, nr)
    Ratios_nd, Ratios_pd = np.meshgrid(ratios_nd, ratios_pd, indexing = 'ij')
    
    _args = [(ratios_pd / ratios_nd[i], ratios_nd[i]) for i in range(nr)]
    
    eps_train_0, eps_train_2,eps_test_0,eps_test_2,eps_train_H1k_0,eps_train_H1k_2,eps_test_H1k_0,eps_test_H1k_2,params_0,params_hat_0 = multiProcessingEvaluation(scan_over_pn_ratios, _args, nProc = nProc)
    
    data_dir = 'data/2d-landscapes/paper/sobo-training/rf_{}_lbda_{}_delta_{}_nonlin_{}_sigma_{}'.format(rf_str, args.lbda, delta, nonlin_str, sigma_str)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    
    np.save(data_dir + '/lbda.npy', args.lbda)
    np.save(data_dir + '/k.npy', args.k)
    np.save(data_dir + '/Delta.npy', delta)
    np.save(data_dir + '/sigma.npy', sigma_str)
    np.save(data_dir + '/nonlin.npy', nonlin_str)
    np.save(data_dir + '/theta.npy', rf_str)
    np.save(data_dir + '/tau.npy', tau)
    np.save(data_dir + '/Ratios-nd.npy', Ratios_nd)
    np.save(data_dir + '/Ratios-pd.npy', Ratios_pd)
    
    np.save(data_dir + '/eps-train-L2-0.npy', eps_train_0)
    np.save(data_dir + '/eps-train-L2-2.npy', eps_train_2)
    np.save(data_dir + '/eps-train-H1k-0.npy', eps_train_H1k_0)
    np.save(data_dir + '/eps-train-H1k-2.npy', eps_train_H1k_2)
    np.save(data_dir + '/eps-test-L2-0.npy', eps_test_0)
    np.save(data_dir + '/eps-test-L2-2.npy', eps_test_2)
    np.save(data_dir + '/eps-test-H1k-0.npy', eps_test_H1k_0)
    np.save(data_dir + '/eps-test-H1k-2.npy', eps_test_H1k_2)
    
    np.save(data_dir + '/params.npy', params_0)
    np.save(data_dir + '/params-hat.npy', params_hat_0)
    
    cm = 'Spectral_r'
    nlevels = 25
    
    # plot l2 training error
    eps_train = eps_train_0 + k * eps_train_2
    
    plt.figure()
    p1 = plt.contourf(Ratios_nd, Ratios_pd, np.log(eps_train), levels = nlevels, cmap = cm)
    plt.contour(Ratios_nd, Ratios_pd, np.log(eps_train), levels = nlevels, colors = 'black', linewidths = .5, antialiased = True)
    plt.colorbar(p1, label = r'$\log \varepsilon^{L^2}_{\mathrm{train}}$')
    plt.ylabel(r'$p/d$')
    plt.xlabel(r'$n/d$')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(r'\begin{{center}}$\lambda = {}$, $\Delta^2 = {}$, $\tau = {}$,\\ $\sigma =$ {}, $\Theta =$ {}, $\phi =$ {}, $k =$ {}\end{{center}}'.format(args.lbda, delta**2, tau, sigma_str, rf_str, nonlin_str, k))
    plt.savefig(data_dir + '/l2-err-train.pdf', bbox_inches = 'tight')
    plt.close()
    
    # plot l2 generalization error
    eps_test = eps_test_0 + k * eps_test_2
    
    plt.figure()
    p1 = plt.contourf(Ratios_nd, Ratios_pd, np.log(eps_test), levels = nlevels, cmap = cm)
    plt.contour(Ratios_nd, Ratios_pd, np.log(eps_test), levels = nlevels, colors = 'black', linewidths = .5, antialiased = True)
    plt.colorbar(p1, label = r'$ \log \varepsilon^{L^2}_{\mathrm{gen}}$')
    plt.ylabel(r'$p/d$')
    plt.xlabel(r'$n/d$')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(r'\begin{{center}}$\lambda = {}$, $\Delta^2 = {}$, $\tau = {}$,\\ $\sigma =$ {}, $\Theta =$ {}, $\phi =$ {}, $k =$ {}\end{{center}}'.format(args.lbda, delta**2, tau, sigma_str, rf_str, nonlin_str, k))
    plt.savefig(data_dir + '/l2-err-gen.pdf', bbox_inches = 'tight')
    plt.close()
    
    # plot H1k error expectation 
    eps_h1k = eps_train_H1k_0 + k * eps_train_H1k_2
    
    plt.figure()
    p1 = plt.contourf(Ratios_nd, Ratios_pd, np.log(eps_h1k), levels = nlevels, cmap = cm)
    plt.contour(Ratios_nd, Ratios_pd, np.log(eps_h1k), levels = nlevels, colors = 'black', linewidths = .5, antialiased = True)
    plt.colorbar(p1, label = r'$\log E \left[ \varepsilon^{H^{1,k}}_{\mathrm{train}} \right]$')
    plt.ylabel(r'$p/d$')
    plt.xlabel(r'$n/d$')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(r'\begin{{center}}$\lambda = {}$, $\Delta^2 = {}$, $\tau = {}$,\\ $\sigma =$ {}, $\Theta =$ {}, $\phi =$ {}, $k =$ {}\end{{center}}'.format(args.lbda, delta**2, tau, sigma_str, rf_str, nonlin_str, k))
    plt.savefig(data_dir + '/h1k-err-train.pdf', bbox_inches = 'tight')
    plt.close()
    
    # plot H1k error expectation 
    eps_h1k = eps_test_H1k_0 + k * eps_test_H1k_2
    
    plt.figure()
    p1 = plt.contourf(Ratios_nd, Ratios_pd, np.log(eps_h1k), levels = nlevels, cmap = cm)
    plt.contour(Ratios_nd, Ratios_pd, np.log(eps_h1k), levels = nlevels, colors = 'black', linewidths = .5, antialiased = True)
    plt.colorbar(p1, label = r'$\log E \left[ \varepsilon^{H^{1,k}}_{\mathrm{gen}} \right]$')
    plt.ylabel(r'$p/d$')
    plt.xlabel(r'$n/d$')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(r'\begin{{center}}$\lambda = {}$, $\Delta^2 = {}$, $\tau = {}$,\\ $\sigma =$ {}, $\Theta =$ {}, $\phi =$ {}, $k =$ {}\end{{center}}'.format(args.lbda, delta**2, tau, sigma_str, rf_str, nonlin_str, k))
    plt.savefig(data_dir + '/h1k-err-gen.pdf', bbox_inches = 'tight')
    plt.close()

########################################################################

if __name__ == '__main__':
    
    # ~ plot_2d_generalization_landscape(rmin = -1.5, rmax = 2.5, nr = 90)
    
    ####################################################################
    
    r_pn_start = args.ratio_pn_start
    r_pn_end   = args.ratio_pn_end
    r_pn_num   = args.ratio_pn_num
    if args.logspacing:
        r_pn = np.logspace(np.log10(r_pn_start), np.log10(r_pn_end), r_pn_num)
    else:
        r_pn = np.linspace(r_pn_start, r_pn_end, r_pn_num)
    r_nd = args.ratio_nd
    scan_over_pn_ratios( (r_pn, r_nd), save = True)
