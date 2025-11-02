########################################################################
# [README]
#
# L2 training theory script
########################################################################

import numpy as np
import matplotlib.pyplot as plt
from texSettings import *
from scipy.optimize import root
from scipy.special import erf
from scipy.integrate import quad, dblquad
import tqdm
from datetime import datetime
import os
import sys
import copy
import argparse
from types import SimpleNamespace

########################################################################
##### global parameters
########################################################################

seed = 42

parser = argparse.ArgumentParser()

# required arguments

parser.add_argument("lbda",   type = float, help = "regularization")
parser.add_argument("rf_str", type = str,   help = "random features")

# optional arguments

parser.add_argument('--ratio_nd',   type = float, default = 2.345, help = "no. training samples / no. dimensions")
parser.add_argument('--nonlin_str', type = str,   default = 'id',  help = "forward model nonlinearity")
parser.add_argument('--sigma_str',  type = str,   default = 'erf', help = "activation function")
parser.add_argument('--epsabs',     type = float, default = 1e-3,  help = "error tolerance for 2D quadratures")
parser.add_argument('--root_alg',   type = str,   default = 'excitingmixing', help = 'root finding algorithm')
parser.add_argument('--k',          type = int,   default = 1,     help = "no. of derivative directions")
parser.add_argument('--delta',      type = float, default = 0.,    help = "additive noise standard dev for data")

# options for p/n scan

parser.add_argument('--ratio_pn_start', type = float, default = 0.01, help = "ratio_pn = linspace(start, end, num)")
parser.add_argument('--ratio_pn_end',   type = float, default = 4.0,  help = "ratio_pn = linspace(start, end, num)")
parser.add_argument('--ratio_pn_num',   type = int,   default = 501,  help = "ratio_pn = linspace(start, end, num)")

parser.add_argument('--verbose', action = 'store_true', help = "print output for pn scan")
parser.add_argument('--not_verbose', dest = 'verbose', action = 'store_false')
parser.set_defaults(verbose = False)

parser.add_argument('--logspacing', action = 'store_true', help = "logarithmic spacing for pn scan")
parser.add_argument('--not_logspacing', dest = 'verbose', action = 'store_false')
parser.set_defaults(logspacing = False)

# which fixed-point system is solved, and how the terms are evaluated

parser.add_argument('--solve_only_sigma', action = 'store_true')
parser.add_argument('--solve_full_system', dest = 'solve_only_sigma', action = 'store_false')
parser.set_defaults(solve_only_sigma = True)

parser.add_argument('--use_simplified_updates', action = 'store_true')
parser.add_argument('--use_quadrature_updates', dest = 'use_simplified_updates', action = 'store_false')
parser.set_defaults(use_simplified_updates = True)

parser.add_argument('--h1k_error_via_stieltjes', action = 'store_true')
parser.add_argument('--h1k_error_via_MC', dest = 'h1k_error_via_stieltjes', action = 'store_false')
parser.set_defaults(h1k_error_via_stieltjes = True)

# read in, and set some globals without args namespace

args = parser.parse_args()

k          = args.k
rf_str     = args.rf_str
nonlin_str = args.nonlin_str
sigma_str  = args.sigma_str

########################################################################
##### ground truth ridge function

match nonlin_str:
    case 'id':
        phi = lambda omega : omega
        dphi = lambda omega : np.ones_like(omega)
    case 'arctan':
        phi = lambda omega : np.arctan(omega)
        dphi = lambda omega : 1. / (1. + omega**2.)
    case 'cosh':
        phi = lambda omega : np.cosh(omega)
        dphi = lambda omega : np.sinh(omega)
    case 'om2':
        phi = lambda omega : omega**2. - 1.
        dphi = lambda omega : 2. * omega
    case 'reci-cosh':
        phi = lambda omega : 1. / np.cosh(omega)
        dphi = lambda omega : - np.sinh(omega) / np.cosh(omega)**2.
    case 'arctan-plus-reci-cosh':
        phi = lambda omega : np.arctan(omega) + 1. / np.cosh(omega)
        dphi = lambda omega : 1. / (1. + omega**2.) - np.sinh(omega) / np.cosh(omega)**2.
    case 'Gaussian':
        phi = lambda omega : np.exp(  -omega**2/2. ) / (2.**np.sqrt(np.pi))
        dphi = lambda omega : -omega*np.exp(  -omega**2/2. )  / (2.**np.sqrt(np.pi))
    case 'ReLU':
        phi  = lambda omega : np.where(omega > 0., omega, 0.)
        dphi = lambda omega : np.where(omega > 0., 1., 0.)
    case 'linGauss':
        phi   = lambda omega : omega/2. - np.exp(  -omega**2/2. )
        dphi = lambda omega : 1./2. +  omega*np.exp(  -omega**2/2. )
    case 'cos+sin':
        phi   = lambda omega : np.cos(omega) + np.sin(omega)
        dphi = lambda omega : np.cos(omega) - np.sin(omega)
    case _:
        raise NotImplementedError(f"nonlin_str = {nonlin_str}")

########################################################################
##### activation functions

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
        sigma  = lambda x : np.where(x > 0., x, 0.)
        dsigma = lambda x : np.where(x > 0., 1., 0.)

    case 'SiLU':
        sigma  = lambda x : x / (1. + np.exp(-x))
        dsigma = lambda x : (1. + (1. + x) * np.exp(-x)) / (1. + np.exp(-x))**2.

    case 'erf':
        sigma  = lambda x : erf(x)
        dsigma = lambda x : 2. / np.sqrt(np.pi) * np.exp(-x**2.)

    case 'sign':
        sigma  = lambda x : np.where(x >= 0., 1., -1.)
        dsigma = lambda x : 0. * x

    case 'tanh':
        sigma = lambda x : np.tanh(x)
        dsigma = lambda x : 1. / np.cosh(x)**2

    case _:
        raise NotImplementedError(f"sigma_str= {sigma_str}")

########################################################################
##### noise models -- additive Gaussian for function and gradient

delta     = args.delta
noise_cov = np.eye(k+1) * delta**2

########################################################################
##### Stieltjes transform of random feature spectral density

if rf_str == 'iid_gauss':

    def g_mu(z, gamma):
        sigma2 = 1./gamma
        return ( sigma2*(1-gamma) - z - np.sqrt( (z - sigma2*(1. + gamma))**2. - 4. * gamma * sigma2**2 ) ) / (2. * gamma * z * sigma2)

    def g_mu_prime(z, gamma):
        sigma2 = 1/gamma

        f = sigma2*(1-gamma) - z - np.sqrt( (z-sigma2*(1+gamma))**2 - 4 * gamma * sigma2**2 ) 
        g = 2*gamma*z*sigma2

        fprime = -1 - (z - sigma2*(gamma+1)) / np.sqrt( (z - sigma2*(gamma+1))**2 - 4 * gamma * sigma2**2 )
        gprime = 2 * gamma * sigma2

        return (fprime*g - f*gprime) / g**2

elif rf_str == 'ortho':
    
    def g_mu(z, gamma):
        return -1. / z * (1. - min(1., 1./gamma)) + min(1., 1./gamma) / (max(1., 1./gamma) - z)

    def g_mu_prime(z, gamma):
        return 1. / z**2. * (1. - min(1., 1./gamma)) + min(1., 1./gamma) / (max(1., 1./gamma) - z)**2.

########################################################################
##### quadratures for Gaussians

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

########################################################################
##### fixed point RHS function

# solve only for sigma in iteration, and compute the rest afterwards

def eq_sys_sigma_a(sigma_a: float, settings, ):
    
    alpha = settings.alpha
    gamma = settings.gamma
    k0, k1, ks = settings.k0, settings.k1, settings.ks
    
    sigma_a_hat = alpha / (1. + sigma_a)
    
    # ~ lbda = args.lbda / gamma
    lbda = args.lbda / alpha
    z = (alpha * lbda + ks**2 * sigma_a_hat) / (k1**2 * sigma_a_hat)
    
    rhs_sigma_a  = gamma / sigma_a_hat * (1 - z*g_mu(-z,gamma))
    rhs_sigma_a += gamma * ks**2 / k1**2 / sigma_a_hat * (1/z * (1/gamma - 1) + g_mu(-z,gamma))
    
    return rhs_sigma_a - sigma_a

def determine_other_overlaps_from_sigma(sigma_a: float, settings, ):
    
    alpha = settings.alpha
    gamma = settings.gamma
    k0, k1, ks = settings.k0, settings.k1, settings.ks
    
    sigma_a_hat = alpha / (1. + sigma_a)
    
    # ~ lbda = args.lbda / gamma
    lbda = args.lbda / alpha
    z = (alpha * lbda + ks**2 * sigma_a_hat) / (k1**2 * sigma_a_hat)
    
    fa_hat = settings.alpha * settings.E_dphi / (1. + sigma_a)
    fa = fa_hat / sigma_a_hat * (1. - z * g_mu(-z,gamma) )
    
    rhs_qhat = alpha / (1. + sigma_a)**2. * (delta**2. - 2. * fa * settings.E_dphi)
    if np.abs(settings.k0 > 1e-14):
        rhs_qhat += alpha / (1. + sigma_a)**2. * settings.var_phi
    else:
        rhs_qhat += alpha / (1. + sigma_a)**2. * settings.E_phi2
    
    rhs_q  = fa_hat**2 / sigma_a_hat**2 * (1 - 2*z*g_mu(-z,gamma) + z**2*g_mu_prime(-z,gamma))
    rhs_q += fa_hat**2 * ks**2 / k1**2 / sigma_a_hat**2 * (g_mu(-z,gamma) - z*g_mu_prime(-z,gamma)) 
    
    q_matr = np.zeros((2,2))
    q_matr[0,0] = -alpha / (1. + sigma_a)**2.
    q_matr[0,1] = 1.
    q_matr[1,0] = 1.
    q_matr[1,1] = -gamma / sigma_a_hat**2 * (1 - 2*z*g_mu(-z,gamma) + z**2*g_mu_prime(-z,gamma))
    q_matr[1,1]+= -gamma * ks**4 / k1**4 / sigma_a_hat**2 * (1/z**2 * (1/gamma -1) + g_mu_prime(-z,gamma))
    q_matr[1,1]+= -2*gamma * ks**2 / k1**2 / sigma_a_hat**2 * (g_mu(-z,gamma) - z*g_mu_prime(-z,gamma))
    
    q_rhs  = np.array([rhs_qhat, rhs_q])
    q = np.linalg.inv(q_matr) @ q_rhs

    return fa, q[0], sigma_a

# solve for the full set of overlap params in fixed-point iteration

def update_hats( fa: float, qa: float, Va: float, settings, ):
    
    alpha = settings.alpha
    gamma = settings.gamma
    
    ####################################################################
    
    if args.use_simplified_updates:
    
        ################################################################
        ##### Using simplified updates
        
        Va_hat = alpha / (1. + Va)
        fa_hat = alpha * settings.E_dphi / (1. + Va)
        qa_hat = alpha / (1. + Va)**2. * (noise_cov[0,0] + qa - 2. * fa * settings.E_dphi)
        
        if np.abs(settings.k0 > 1e-14):
            # sa = E[phi]
            qa_hat += alpha / (1. + Va)**2. * settings.var_phi
        else:
            # sa = 0
            qa_hat += alpha / (1. + Va)**2. * settings.E_phi2
    
    else:
        
        ################################################################
        ##### Using quadratures instead
        
        # auxiliary variances
        s12 = 1. - fa**2 / qa
        s22 = Va

        # quadrature
        xi_quad, w_xi             = make_gauss_quad1D(150, mu=0., var=1.)
        omega_quad, w_omega       = make_gauss_quad1D(150, mu=0., var=s12)
        xi_omega_quad, w_xi_omega = tensor_product_quadrature(xi_quad, w_xi, omega_quad, w_omega)

        # means
        m1 = ( fa / np.sqrt(qa) ) * xi_omega_quad[:,0]
        if np.abs(settings.k0 > 1e-14): 
            sa = settings.E_phi
        else:
            sa = 0.
        m2 = sa   +   np.sqrt(qa) * xi_omega_quad[:,0]

        # conjugate  parameters
        fa_hat = alpha * (1./ ((1. + s22) * s12)) * np.dot(w_xi_omega,   xi_omega_quad[:,1] * phi(xi_omega_quad[:,1] + m1) )
        qa_hat = alpha         * (1./ (1. + s22))**2.     * np.dot(w_xi_omega,   delta**2. + (phi(xi_omega_quad[:,1] + m1) - m2)**2. )
        Va_hat = alpha         * (1./ (1. + s22))
    
    ####################################################################

    return np.array([ fa_hat, qa_hat, Va_hat ])

def report( f: np.ndarray, q: np.ndarray, V: np.ndarray, headline: str):

    match headline:
        case 'hat':
            print('Conjugate parameters:')
            print('f_hat =', f )
            print('q_hat =', q )
            print('V_hat =', V )
            print('##########################')
        case 'residual':
            print('Mismatch from eqs:')
            print('res_f =', f )
            print('res_q =', q )
            print('res_V =', V )
            print('##########################')
        case 'final':
            print('##########################')
            print('Found the following parameters:')
            print('f  =', f)
            print('q  =', q)
            print('V  =', V)
            print('##########################')
        case _:
            print('Current parameters:')
            print('f  =', f)
            print('q  =', q)
            print('V  =', V)
            print('##########################')

def eq_sys(params, settings, ):
    
    alpha, gamma = settings.alpha, settings.gamma
    k0, k1, ks = settings.k0, settings.k1, settings.ks

    fa = params[0]
    qa = params[1]
    Va = params[2]

    if settings.verbose:
        report( *params, 'current', )

    params_hat = update_hats( *params, settings)

    if settings.verbose:
        report( *params_hat, 'hat', )

    fa_hat = params_hat[0]
    qa_hat = params_hat[1]
    Va_hat = params_hat[2]
    
    if args.use_simplified_updates:
        
        ################################################################
        ##### STIELJES
        # ~ lbda = args.lbda / gamma
        lbda = args.lbda / alpha
        z = (alpha * lbda + ks**2 * Va_hat) / (k1**2 * Va_hat)

        if settings.verbose:
            print('Stieltjes trafo arg z =', z)
            print('##########################')

        rhs_fa = fa_hat / Va_hat * (1. - z * g_mu(-z,gamma) )

        rhs_qa  = (fa_hat**2 + gamma * qa_hat) / Va_hat**2 * (1 - 2*z*g_mu(-z,gamma) + z**2*g_mu_prime(-z,gamma))
        rhs_qa += gamma * ks**4 / k1**4 / Va_hat**2 * qa_hat * (1/z**2 * (1/gamma -1) + g_mu_prime(-z,gamma)) 
        rhs_qa += (2*gamma*qa_hat + fa_hat**2) * ks**2 / k1**2 / Va_hat**2 * (g_mu(-z,gamma) - z*g_mu_prime(-z,gamma)) 

        rhs_Va  = gamma / Va_hat * (1 - z*g_mu(-z,gamma))
        rhs_Va += gamma * ks**2 / k1**2 / Va_hat * (1/z * (1/gamma - 1) + g_mu(-z,gamma)) 
    
    else:
        
        ################################################################
        ##### MC 
        np.random.seed(seed)
        d    = max( 100, int(gamma*100)  )
        p    = int(d/gamma)

        Theta = np.random.randn( d, p ) / np.sqrt(d)
        ThTh  = Theta.T @ Theta
        
        
        # ~ lbda = args.lbda / gamma
        lbda = args.lbda / alpha
        A         = (alpha*lbda + ks**2 * Va_hat)*np.eye(p) + k1**2 * Va_hat * ThTh

        Ainv      = np.linalg.inv(A)
        AinvThTh  = Ainv @ ThTh
        Ainv2     = Ainv @ Ainv
        Ainv2ThTh = Ainv2 @ ThTh
        M2        = AinvThTh @ AinvThTh

        rhs_fa = 1./d * k1**2 * fa_hat * np.trace(AinvThTh)

        rhs_qa  = k1**4              * (fa_hat**2 + gamma * qa_hat ) * AinvThTh @ AinvThTh
        rhs_qa += k1**2 * ks**2 * (2* gamma * qa_hat + fa_hat**2) * Ainv2ThTh
        rhs_qa += gamma * ks**4 * qa_hat * Ainv2
        rhs_qa = 1./d * np.trace(rhs_qa)

        rhs_Va = 1./d * np.trace(gamma * k1**2 * AinvThTh + gamma * ks**2 * Ainv)

    ####################################################################
    
    ret_fa = rhs_fa - fa 
    ret_qa = rhs_qa - qa
    ret_Va = rhs_Va - Va 

    if settings.verbose:
        print('Mismatch from eqs:')
        print('ret_fa =', ret_fa)
        print('ret_qa =', ret_qa)
        print('ret_Va =', ret_Va)
        print('##########################')

    return np.array([ret_fa, ret_qa, ret_Va])

########################################################################
##### generic fixed-point solves / root finders

def damp(new, old, damping=0.7):
    return (1.-damping)*new + damping*old

def damped_fixed_point(params_0, eq_sys, settings, tol=1e-4, max_iter=100000, ):

    params_old = params_0
    for kk in range(max_iter):
        # eq_sys returns the _residual_
        params_new = params_old + eq_sys(params_old, settings, )

        if np.max(np.abs(params_new - params_old))<tol:
            if settings.verbose:
                print(f"converged after {kk} steps")
            break
        else:
            params_old = damp(params_new, params_old)

    if settings.verbose and kk == max_iter-1:
        print(f"failed to converge after {max_iter} steps")

    return params_new

########################################################################
##### main function to solve one instance of the fixed-point system and return errors

def get_analytical_error_results(ratio_pn, params_0, settings, num_mc=200, ):
    
    ####################################################################
    ##### solve fixed point system
    
    if args.root_alg == "excitingmixing" or args.root_alg == "lm":
        
        if args.solve_only_sigma:
            res = root(eq_sys_sigma_a, x0 = params_0[2], tol= 1e-6, args = (settings, ), method = args.root_alg)
            sigma_a = np.squeeze(res.x)
        else: 
            res = root(eq_sys, x0 = params_0, tol= 1e-6, args = (settings, ), method = args.root_alg)
            params = res.x
        
    elif args.root_alg == "fixedpoint":
        if args.solve_only_sigma:
            sigma_a = np.squeeze(damped_fixed_point(params_0[2], eq_sys_sigma_a, settings, ))
        else:
            params = damped_fixed_point(params_0, eq_sys, settings, )
        
    else:
        
        raise NotImplementedError(f'-E- root_alg = {args.root_alg} invalid')
    
    ####################################################################
    ##### find all overlaps
    
    if args.solve_only_sigma:
        fa, qa, Va = determine_other_overlaps_from_sigma(sigma_a, settings)
        params = np.array([fa, qa, Va])
    else:
        fa, qa, Va = params
        
    if settings.verbose:
        print("#################################")
        print("#################################")
        print("#################################")
        print("alpha =", settings.alpha)
        print("gamma =", settings.gamma)
        report( *params, 'current', )

    params_hat = update_hats( *params, settings)

    if args.verbose:
        report( *params_hat, 'hat', )

    fa_hat, qa_hat, Va_hat = params_hat
    
    ####################################################################
    ##### now compute errors from this
    
    alpha, gamma = settings.alpha, settings.gamma
    k1, ks = settings.k1, settings.ks
    dk1, dks = settings.dk1, settings.dks
    
    ####################################################################
    ##### compute training error
    
    eps_train_L2 = qa_hat / settings.alpha
    
    ####################################################################
    ##### compute L2 testing error
    
    if args.use_simplified_updates:
        
        eps_test_L2 = delta**2 + qa - 2. * fa * settings.E_dphi
        if np.abs(settings.k0 > 1e-14):
            # sa = E[phi]
            eps_test_L2 += settings.var_phi
        else:
            # sa = 0
            eps_test_L2 += settings.E_phi2
            
    else:
        
        if np.abs(settings.k0 > 1e-14):
            sa = settings.E_phi
        else:
            sa = 0.
        Z, w_Z = make_gauss_quad1D(150, mu=0., var=1.)
        eps_test_L2 = noise_cov[0,0] + qa - fa**2 + np.dot(w_Z, (sa + fa*Z - phi(Z))**2 )
        
    ####################################################################
    ##### compute H1k training = testing error
    
    # ~ lbda = args.lbda / gamma
    lbda = args.lbda / alpha
    z = (alpha * lbda + ks**2 * Va_hat) / (k1**2 * Va_hat)
    whatsquared = gamma * ks**2. / (k1**4. * Va_hat**2.) * qa_hat * (1/z**2 * (1/gamma -1) + g_mu_prime(-z,gamma))
    whatsquared += (fa_hat**2. + gamma * qa_hat) / (k1**2. * Va_hat**2.) * (g_mu(-z,gamma) - z*g_mu_prime(-z,gamma))
    
    if settings.verbose:
        # print intermediate results and qc overlap param
        print('qa - k_*^2 w^2 =', qa - ks**2. * whatsquared)
        print("qc =", whatsquared * (dk1**2. + dks**2.))
    
    if args.h1k_error_via_stieltjes:
        
        ################################################################
        ##### H1k generalization error via Stieltjes transform

        # the H1k generalization error is a generalized chi2 distribution with 2(k+1) dof,
        # i.e. eps = eps_0 + z.T @ eps_test_H1k_2 @ z, where z ~ N(0, I_2)

        eps_test_H1k_0 = np.trace(noise_cov[1:,1:]) + k*(whatsquared * (dk1**2. + dks**2.))

        eps_test_H1k_2 = np.array( [[settings.E_dphi2 - 2*settings.E_dphi*fa + fa**2, -(settings.E_dphi - fa)*np.sqrt(qa - ks**2 * whatsquared - fa**2)],
            [-(settings.E_dphi - fa)*np.sqrt(qa - ks**2 * whatsquared - fa**2), qa - ks**2 * whatsquared - fa**2 ] ])
   
    else:
        
        ################################################################
        ##### MC for trace of qc in H1k generalization error
        
        d    = max( 100, int(gamma*100)  )
        p    = int(d/gamma)

        eps_test_H1k = 0

        for _ in tqdm.tqdm(range(num_mc)):
        
            Theta = np.random.randn( d, p ) / np.sqrt(d)
            ThTh = Theta.T @ Theta
            
            Xi    = ks**2. * qa_hat * np.eye(p) + k1**2. * (fa_hat**2. / gamma + qa_hat) * ThTh
            # ~ lbda = args.lbda / gamma
            lbda = args.lbda / alpha
            A         = (alpha*lbda + ks**2 * Va_hat)*np.eye(p) + k1**2 * Va_hat * ThTh
            Ainv      = np.linalg.inv(A)
            
            M = (Ainv @ Xi @ Ainv) * (dk1**2. * ThTh + dks**2. * np.eye(p))
            
            # zeta = np.random.randn(p, k)
            
            Vk = np.random.randn(d, k)
            zeta = Theta.T @ Vk
            qc = zeta.T @ M @ zeta / p
            eps_test_H1k += np.trace(qc)
            
            # eps_test_H1k += np.trace(M)/p
        
        eps_test_H1k_0 = eps_test_H1k / num_mc
        
        eps_test_H1k_2 = np.array( [[settings.E_dphi2 - 2*settings.E_dphi*fa + fa**2, -(settings.E_dphi - fa)*np.sqrt(qa - ks**2 * whatsquared - fa**2)],
            [-(settings.E_dphi - fa)*np.sqrt(qa - ks**2 * whatsquared - fa**2), qa - ks**2 * whatsquared - fa**2 ] ])
    
    ####################################################################
    ##### regularization term lbda/2 ||w||^2 from training
    
    # ~ lbda = args.lbda / gamma
    lbda = args.lbda / alpha
    eps_train_reg = lbda / 2. * whatsquared

    ####################################################################
    
    if settings.verbose:
        print('Training error (L2)        =', eps_train_L2)
        print('Training error (reg)       =', eps_train_reg)
        print('Testing  error (L2)        =', eps_test_L2)
        print('Testing  error (H1k base)  =', eps_test_H1k_0)
        print('Testing  error (H1k expec) =', eps_test_H1k_0 + k * np.trace(eps_test_H1k_2))
    
    return eps_train_L2, eps_train_reg, eps_test_L2, eps_test_H1k_0, eps_test_H1k_2, params, params_hat

########################################################################
##### wrappers for different types of parameter scans, and plots of results

def scan_over_pn_ratios(r_pn_start, r_pn_end, r_pn_num, r_nd, logspacing = False):
    
    settings = SimpleNamespace()

    settings.verbose = args.verbose

    settings.k0 = get_k0(sigma)
    settings.k1 = get_k1(sigma)
    settings.ks = get_ks(sigma, settings.k0, settings.k1)
    
    settings.dk0 = get_k0(dsigma)
    settings.dk1 = get_k1(dsigma)
    settings.dks = get_ks(dsigma, settings.dk0, settings.dk1)
    
    print('###########################################################')
    print('k0 =', settings.k0)
    print('k1 =', settings.k1)
    print('ks =', settings.ks)
    print('dk0 =', settings.dk0)
    print('dk1 =', settings.dk1)
    print('dks =', settings.dks)
    print('###########################################################')
    
    # precompute quadratures
    Z, w_Z = make_gauss_quad1D(150, mu=0., var=1.)

    E_phi = np.dot(w_Z, phi(Z))
    E_phi2 = np.dot(w_Z, phi(Z)**2.)
    E_dphi = np.dot(w_Z, dphi(Z))
    E_dphi2 = np.dot(w_Z, dphi(Z)**2.)
    var_phi  = E_phi2 - E_phi**2
    var_dphi = E_dphi2 - E_dphi**2
    
    settings.E_phi = E_phi
    settings.E_phi2 = E_phi2
    settings.E_dphi  = E_dphi 
    settings.E_dphi2 = E_dphi2
    settings.var_phi  = var_phi
    settings.var_dphi = var_dphi

    print('E[phi] =', settings.E_phi)
    print('E[phi^2] =', settings.E_phi2)
    print('E[phi prime] =', settings.E_dphi)
    print('E[(phi prime)^2] =', settings.E_dphi2)
    print('Var[phi] =', settings.var_phi)
    print('Var[phi_prime] =', settings.var_dphi)
    print('###########################################################')

    ################################

    params_0 = np.array([0.2, 0.3, 0.4])

    ################################

    if logspacing:
        ratio_pn = np.logspace(np.log10(r_pn_start), np.log10(r_pn_end), r_pn_num)
    else:
        ratio_pn = np.linspace(r_pn_start, r_pn_end, r_pn_num)

    now = datetime.now()
    dt_string = now.strftime('%Y_%m_%d_%H_%M_%S')
    base_dir = 'rf-theory-{}'.format(nonlin_str)
    data_dir = 'data/{}/{}'.format(base_dir, dt_string)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    ################################

    eps_train_L2, eps_train_reg, eps_test_L2, eps_test_H1k_0, eps_test_H1k_2 = np.zeros(len(ratio_pn)), np.zeros(len(ratio_pn)), np.zeros(len(ratio_pn)), np.zeros(len(ratio_pn)), np.zeros((len(ratio_pn),2,2))

    for i in tqdm.tqdm(range(len(ratio_pn))):

        # ~ print('ratio_pn = {}'.format(ratio_pn[i]))
        settings.alpha = 1. / ratio_pn[i]
        settings.gamma = 1. / (r_nd * ratio_pn[i])

        eps_train_tmp, eps_train_reg_tmp, eps_test_tmp, eps_test_H1k_0_tmp, eps_test_H1k_2_tmp, params_0_tmp, params_hat_0_tmp = get_analytical_error_results(ratio_pn[i], params_0, settings)
        
        eps_train_L2[i]    = eps_train_tmp
        eps_train_reg[i]   = eps_train_reg_tmp
        eps_test_L2[i]     = eps_test_tmp
        eps_test_H1k_0[i]  = eps_test_H1k_0_tmp
        eps_test_H1k_2[i]  = eps_test_H1k_2_tmp
        params_0 = params_0_tmp

    np.save(data_dir + '/lbda.npy', args.lbda)
    np.save(data_dir + '/k.npy', args.k)
    np.save(data_dir + '/Delta.npy', delta)
    np.save(data_dir + '/sigma.npy', sigma_str)
    np.save(data_dir + '/nonlin.npy', nonlin_str)
    np.save(data_dir + '/theta.npy', rf_str)
    np.save(data_dir + '/ratio-nd.npy', r_nd)
    np.save(data_dir + '/ratio-pn.npy', ratio_pn)
    np.save(data_dir + '/eps-train-L2.npy', eps_train_L2)
    np.save(data_dir + '/eps-train-reg.npy', eps_train_reg)
    np.save(data_dir + '/eps-test-L2.npy', eps_test_L2)
    np.save(data_dir + '/eps-test-H1k-0.npy', eps_test_H1k_0)
    np.save(data_dir + '/eps-test-H1k-2.npy', eps_test_H1k_2)

    plt.figure()
    plt.plot(ratio_pn, eps_train_L2, '-', label = r'$\varepsilon_{\mathrm{train}}$', color = 'steelblue')
    plt.plot(ratio_pn, eps_train_reg, '-', label = r'$\lambda ||w||^2/2$', color = 'navy')
    plt.plot(ratio_pn, eps_test_L2,  '-', label = r'$\varepsilon_{\mathrm{test}}$'  , color = 'orange')
    plt.plot(ratio_pn, eps_test_H1k_0,  '-', label = r'$\varepsilon_{\mathrm{test}, H1k}$'  , color = 'firebrick')
    plt.grid()
    plt.xlabel(r'$\alpha^{-1} = p / n$')
    plt.ylabel(r'$\varepsilon$')
    plt.legend()
    if logspacing:
        plt.xscale('log')
    
    plt.title(r'\begin{{center}}$\lambda = {}$, $\Delta = {}$, $\alpha / \gamma = n/d = {}$,\\ $\sigma =$ {}, $\Theta =$ {}, $\phi =$ {}, $k =$ {}\end{{center}}'.format(args.lbda, delta, r_nd, sigma_str, rf_str, nonlin_str, k))
    plt.savefig(data_dir + '/eps.pdf', bbox_inches = 'tight')
    plt.close()

def single_prediction(r_pn, r_nd, verb = True, settings = None):
    
    ################################
    
    if settings is None:
    
        settings = SimpleNamespace()

        settings.verbose = verb

        settings.k0 = get_k0(sigma)
        settings.k1 = get_k1(sigma)
        settings.ks = get_ks(sigma, settings.k0, settings.k1)
        
        settings.dk0 = get_k0(dsigma)
        settings.dk1 = get_k1(dsigma)
        settings.dks = get_ks(dsigma, settings.dk0, settings.dk1)
        
        # precompute quadratures
        Z, w_Z = make_gauss_quad1D(150, mu=0., var=1.)

        E_phi = np.dot(w_Z, phi(Z))
        E_phi2 = np.dot(w_Z, phi(Z)**2.)
        E_dphi = np.dot(w_Z, dphi(Z))
        E_dphi2 = np.dot(w_Z, dphi(Z)**2.)
        var_phi  = E_phi2 - E_phi**2
        var_dphi = E_dphi2 - E_dphi**2
        
        settings.E_phi = E_phi
        settings.E_phi2 = E_phi2
        settings.E_dphi  = E_dphi 
        settings.E_dphi2 = E_dphi2
        settings.var_phi  = var_phi
        settings.var_dphi = var_dphi

    ################################

    params_0 = np.array([0.2, 0.3, 0.4])

    ################################
    
    if verb:
        print('ratio_pn = {}'.format(r_pn))
        print('ratio_nd = {}'.format(r_nd))
    settings.alpha = 1. / r_pn
    settings.gamma = 1. / (r_nd * r_pn)
    eps_train_tmp, eps_train_reg_tmp, eps_test_tmp, eps_test_H1k_0_tmp, eps_test_H1k_2_tmp, params_0_tmp, params_hat_0_tmp = get_analytical_error_results(r_pn, params_0, settings)
    return eps_train_tmp, eps_train_reg_tmp, eps_test_tmp, eps_test_H1k_0_tmp, eps_test_H1k_2_tmp, params_0_tmp, params_hat_0_tmp
    
def plot_2d_generalization_landscape(rmin = -2., rmax = 3., nr = 64):
    
    ratios_nd = np.logspace(rmin, rmax, nr)
    ratios_pd = np.logspace(rmin, rmax, nr)
    Ratios_nd, Ratios_pd = np.meshgrid(ratios_nd, ratios_pd, indexing = 'ij')
    
    eps_train = np.zeros((nr, nr))
    eps_train_reg = np.zeros((nr, nr))
    eps_test = np.zeros((nr, nr))
    eps_test_H1k_0 = np.zeros((nr, nr))
    eps_test_H1k_2 = np.zeros((nr, nr, 2, 2))
    params_0 = np.zeros((nr, nr, 3))
    params_hat_0 = np.zeros((nr, nr, 3))
    
    settings = SimpleNamespace()
    settings.verbose = False
    settings.k0 = get_k0(sigma)
    settings.k1 = get_k1(sigma)
    settings.ks = get_ks(sigma, settings.k0, settings.k1)
    settings.dk0 = get_k0(dsigma)
    settings.dk1 = get_k1(dsigma)
    settings.dks = get_ks(dsigma, settings.dk0, settings.dk1)
    Z, w_Z = make_gauss_quad1D(150, mu=0., var=1.)
    E_phi = np.dot(w_Z, phi(Z))
    E_phi2 = np.dot(w_Z, phi(Z)**2.)
    E_dphi = np.dot(w_Z, dphi(Z))
    E_dphi2 = np.dot(w_Z, dphi(Z)**2.)
    var_phi  = E_phi2 - E_phi**2
    var_dphi = E_dphi2 - E_dphi**2
    settings.E_phi = E_phi
    settings.E_phi2 = E_phi2
    settings.E_dphi  = E_dphi 
    settings.E_dphi2 = E_dphi2
    settings.var_phi  = var_phi
    settings.var_dphi = var_dphi
    
    for i in tqdm.tqdm(range(nr)):
        for j in range(nr):
            r_nd = ratios_nd[i]
            r_pd = ratios_pd[j]
            r_pn = r_pd / r_nd
            eps_train_tmp, eps_train_reg_tmp, eps_test_tmp, eps_test_H1k_0_tmp, eps_test_H1k_2_tmp, params_0_tmp, params_hat_0_tmp = single_prediction(r_pn, r_nd, verb = False, settings = settings)
            eps_train[i,j] = eps_train_tmp
            eps_train_reg[i,j] = eps_train_reg_tmp
            eps_test[i,j] = eps_test_tmp
            eps_test_H1k_0[i,j] = eps_test_H1k_0_tmp
            eps_test_H1k_2[i,j] = eps_test_H1k_2_tmp
            params_0[i,j,:] = params_0_tmp
            params_hat_0[i,j,:] = params_hat_0_tmp

    data_dir = 'data/2d-landscapes/paper/l2-training/rf_{}_lbda_{}_delta_{}_nonlin_{}_sigma_{}'.format(rf_str, args.lbda, delta, nonlin_str, sigma_str)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    
    np.save(data_dir + '/lbda.npy', args.lbda)
    np.save(data_dir + '/k.npy', args.k)
    np.save(data_dir + '/Delta.npy', delta)
    np.save(data_dir + '/sigma.npy', sigma_str)
    np.save(data_dir + '/nonlin.npy', nonlin_str)
    np.save(data_dir + '/theta.npy', rf_str)
    np.save(data_dir + '/Ratios-nd.npy', Ratios_nd)
    np.save(data_dir + '/Ratios-pd.npy', Ratios_pd)
    np.save(data_dir + '/eps-train-L2.npy', eps_train)
    np.save(data_dir + '/eps-train-reg.npy', eps_train_reg)
    np.save(data_dir + '/eps-test-L2.npy', eps_test)
    np.save(data_dir + '/eps-test-H1k-0.npy', eps_test_H1k_0)
    np.save(data_dir + '/eps-test-H1k-2.npy', eps_test_H1k_2)
    np.save(data_dir + '/params.npy', params_0)
    np.save(data_dir + '/params-hat.npy', params_hat_0)
    
    cm = 'Spectral_r'
    nlevels = 25
    
    # plot l2 training error
    plt.figure()
    p1 = plt.contourf(Ratios_nd, Ratios_pd, np.log(eps_train), levels = nlevels, cmap = cm)
    plt.contour(Ratios_nd, Ratios_pd, np.log(eps_train), levels = nlevels, colors = 'black', linewidths = .5, antialiased = True)
    plt.colorbar(p1, label = r'$\log \varepsilon^{L^2}_{\mathrm{train}}$')
    plt.ylabel(r'$p/d$')
    plt.xlabel(r'$n/d$')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(r'\begin{{center}}$\lambda = {}$, $\Delta^2 = {}$,\\ $\sigma =$ {}, $\Theta =$ {}, $\phi =$ {}, $k =$ {}\end{{center}}'.format(args.lbda, delta**2, sigma_str, rf_str, nonlin_str, k))
    plt.savefig(data_dir + '/l2-err-train.pdf', bbox_inches = 'tight')
    plt.close()
    
    # plot l2 generalization error
    plt.figure()
    p1 = plt.contourf(Ratios_nd, Ratios_pd, np.log(eps_test), levels = nlevels, cmap = cm)
    plt.contour(Ratios_nd, Ratios_pd, np.log(eps_test), levels = nlevels, colors = 'black', linewidths = .5, antialiased = True)
    plt.colorbar(p1, label = r'$ \log \varepsilon^{L^2}_{\mathrm{gen}}$')
    plt.ylabel(r'$p/d$')
    plt.xlabel(r'$n/d$')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(r'\begin{{center}}$\lambda = {}$, $\Delta^2 = {}$,\\ $\sigma =$ {}, $\Theta =$ {}, $\phi =$ {}, $k =$ {}\end{{center}}'.format(args.lbda, delta**2, sigma_str, rf_str, nonlin_str, k))
    plt.savefig(data_dir + '/l2-err-gen.pdf', bbox_inches = 'tight')
    plt.close()
    
    # plot regularization term
    plt.figure()
    p1 = plt.contourf(Ratios_nd, Ratios_pd, np.log(eps_train_reg), levels = nlevels, cmap = cm)
    plt.contour(Ratios_nd, Ratios_pd,np.log(eps_train_reg), levels = nlevels, colors = 'black', linewidths = .5, antialiased = True)
    plt.colorbar(p1, label = r'$ \log \left( \frac{\lambda}{2} || w^* ||^2 \right)$')
    plt.ylabel(r'$p/d$')
    plt.xlabel(r'$n/d$')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(r'\begin{{center}}$\lambda = {}$, $\Delta^2 = {}$,\\ $\sigma =$ {}, $\Theta =$ {}, $\phi =$ {}, $k =$ {}\end{{center}}'.format(args.lbda, delta**2, sigma_str, rf_str, nonlin_str, k))
    plt.savefig(data_dir + '/regularization-term.pdf', bbox_inches = 'tight')
    plt.close()
    
    # plot H1k error expectation 
    diag = np.array([0,1])
    eps_h1k = eps_test_H1k_0 + k * np.sum(eps_test_H1k_2[:,:,diag,diag], axis = -1)
    
    plt.figure()
    p1 = plt.contourf(Ratios_nd, Ratios_pd, np.log(eps_h1k), levels = nlevels, cmap = cm)
    plt.contour(Ratios_nd, Ratios_pd, np.log(eps_h1k), levels = nlevels, colors = 'black', linewidths = .5, antialiased = True)
    plt.colorbar(p1, label = r'$\log E \left[ \varepsilon^{H^{1,k}}_{\mathrm{gen}} \right]$')
    plt.ylabel(r'$p/d$')
    plt.xlabel(r'$n/d$')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(r'\begin{{center}}$\lambda = {}$, $\Delta^2 = {}$,\\ $\sigma =$ {}, $\Theta =$ {}, $\phi =$ {}, $k =$ {}\end{{center}}'.format(args.lbda, delta**2, sigma_str, rf_str, nonlin_str, k))
    plt.savefig(data_dir + '/h1k-err-gen.pdf', bbox_inches = 'tight')
    plt.close()

########################################################################

if __name__ == '__main__':
    
    # ~ plot_2d_generalization_landscape(rmin = -1.5, rmax = 2.5, nr = 300)
    
    ################################
    
    # ~ ratio_pn = 0.8
    # ~ single_prediction(ratio_pn, args.ration_nd)

    ################################
    
    scan_over_pn_ratios(args.ratio_pn_start, args.ratio_pn_end, args.ratio_pn_num, args.ratio_nd, logspacing = args.logspacing)
