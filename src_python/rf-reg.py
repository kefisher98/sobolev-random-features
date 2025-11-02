"""
[README]

Monte Carlo simulations for Sobolev training with uninformed gradient model,
with gradient projections "v = N(0, I_d)" for each column of Vk.

Note the same Vk and theta0 will be used across all p/n ratios, but these random variables
will change across different {n_stat} runs unless {seed_varpi} is pre-specified.
"""

import numpy as np
import matplotlib.pyplot as plt
from texSettings import *
from scipy.optimize import minimize
from scipy.special import erf
import tqdm
import multiprocessing as mp
import contextlib as clib
from datetime import datetime
import os
import sys
import argparse


###############################################################
# global parameters
###############################################################

parser = argparse.ArgumentParser()

# required arguments
parser.add_argument("d", type=int, help="dimension")
parser.add_argument("lbda", type=float, help="regularization")
parser.add_argument("rf_str", type=str, help="random features")

# optional arguments
parser.add_argument('--ratio_nd', type=float, default=2.345, help="no. training samples / no. dimensions")
parser.add_argument('--n_test', type=int, default=10000, help="no. generalization samples")
parser.add_argument('--k', type=int, default=1, help="no. of derivative directions")
parser.add_argument('--a_0', type=float, default=1., help="weight for l2 term in sobolev training objective")
parser.add_argument('--a_1', type=float, default=1., help="weight for h1 semi norm term in training objective")
parser.add_argument('--nProc', type=int, default=10, help="number of processors to use") 
parser.add_argument('--n_stat', type=int, default=100, help="repeat experiments this many times")
parser.add_argument('--Delta', type=float, default=0., help="magnitude of noise")
parser.add_argument('--nonlin_str', type=str, default='id', help="forward model nonlinearity")
parser.add_argument('--sigma_str', type=str, default='erf', help="activation function")
parser.add_argument('--noise_model', type=str, default='Additive Normal', help="noise model")
parser.add_argument('--noise_relate', type=str, default='iid', help="relationship between y and y prime noise")

parser.add_argument('--seed_varpi', type=int, default=None, help="If None, use a different seed to create varpi for each n_stat run. Otherwise, use the same seed.")

parser.add_argument('--ratio_pn_start', type=float, default=0.01, help="ratio_pn = linspace(start, end, num)")
parser.add_argument('--ratio_pn_end', type=float, default=4.0, help="ratio_pn = linspace(start, end, num)")
parser.add_argument('--ratio_pn_num', type=int, default=101, help="ratio_pn = linspace(start, end, num)")

# idiomatic and correct way to allow for Boolean argparse arguments
parser.add_argument('--dont_save_constantly', action='store_false', default=True, dest='save_constantly', help="Save results at every p/n")

args = parser.parse_args()

# dimension
d = args.d
# expected norm squared of theta_0
# no. training samples / no. dimensions
ratio_nd = args.ratio_nd
# no. derivative directions for training / no. dimensions
# ratio_kd = args.ratio_kd
k = args.k
# repeat experiments this many times
n_stat = args.n_stat
# no. generalization samples
n_test = args.n_test
# random features
# rf_str = 'iid_gauss'
# rf_str = 'ortho'
rf_str = args.rf_str
nonlin_str = args.nonlin_str
sigma_str = args.sigma_str

# weight for l2 term in sobolev training objective
a_0 = args.a_0
# weight for h1 semi norm term in training objective
a_1 = args.a_1

###############################################################

match nonlin_str:
    case 'id':
        phi = lambda omega : omega
        phi_prime = lambda omega : np.ones_like(omega)
    case 'arctan':
        phi = lambda omega : np.arctan(omega)
        phi_prime = lambda omega : 1. / (1. + omega**2.)
    case 'cosh':
        phi = lambda omega : np.cosh(omega)
        phi_prime = lambda omega : np.sinh(omega)
    case 'om2':
        phi = lambda omega : omega**2. - 1.
        phi_prime = lambda omega : 2. * omega
    case 'reci-cosh':
        phi = lambda omega : 1. / np.cosh(omega)
        phi_prime = lambda omega : - np.sinh(omega) / np.cosh(omega)**2.
    case 'id-plus-reci-cosh':
        phi = lambda omega : omega + 1. / np.cosh(omega)
        phi_prime = lambda omega : np.ones_like(omega) - np.sinh(omega) / np.cosh(omega)**2.
    case 'arctan-plus-reci-cosh':
        phi = lambda omega : np.arctan(omega) + 1. / np.cosh(omega)
        phi_prime = lambda omega : 1. / (1. + omega**2.) - np.sinh(omega) / np.cosh(omega)**2.
    case _:
        raise NotImplementedError(f"nonlin_str = {nonlin_str}")

if a_1 == 0.:
    base_dir = 'rf-sobo-l2-{}'.format(nonlin_str)
elif a_0 > 0. and a_1 > 0.:
    base_dir = 'rf-sobo-{}'.format(nonlin_str)
elif a_0 == 0.:
    base_dir = 'rf-sobo-H1k-{}'.format(nonlin_str)
else:
    print('Error')
    exit()

###############################################################
# noise models
# all models return ( additive noise, multiplicative noise ) 

# CHOOSE model here
Delta            = args.Delta
noise_desc       = args.noise_model #[ 'Additive Normal', 'Additive t', 'Multiplicative Normal', 'Corrupt Signal' ][0]
grad_noise_model = args.noise_relate #[ 'zero', 'match', 'mix', 'iid' ][3]      

# define noise models
match noise_desc:
    case 'Additive Normal':
        get_noise        = lambda n :  (Delta * np.random.normal(size=n), 1)
        noise_desc_short = 'add_normal' 
    case 'Additive t':
        dof              = 3
        get_noise        = lambda n :  (Delta * np.random.standard_t( 3, size=n),  1) 
        noise_desc_short = 'add_t'
    case 'Multiplicative Normal':
        get_noise        = lambda n :  (0, 1 + Delta * np.random.normal(size=n))
        noise_desc_short = 'mult_normal'
    case 'Corrupt Signal':
        par              = 0.9
        noise_desc_short = 'corrupt'
        def get_noise(n): 
            coin         = np.random.binomial(1, par, size=n)
            return ( (1-coin)*Delta, coin )
    case _:
        raise NotImplementedError(f"noise_desc = {noise_desc}")


# grad noise model
# eta is [ additive noise of y, multiplicative noise of y ]
match grad_noise_model:
    case 'zero':
        get_grad_noise = lambda n,*args   :  (0,1)
    case 'match':
        get_grad_noise = lambda n,d,eta   : (eta[0]/np.sqrt(d), eta[1] )
    case 'iid':
        def get_grad_noise(n,d,*args):
            eta = get_noise((d,n))
            return (eta[0]/np.sqrt(d), eta[1])
    case 'mix':
        par = 0.9
        def get_grad_noise(n,d,eta):
            coin              = np.random.binomial(1, par, size=(d,n))
            new_add, new_mult = get_noise((d,n))
            return ( (coin*eta[0] + (1-coin)*new_add)/np.sqrt(d) , coin*eta[1] + (1-coin)*new_mult   )
    case _:
        raise NotImplementedError(f"grad_noise_model= {grad_noise_model}")

###############################################################
# activation function

def make_gauss_quad1D(deg, mu=0., var=1.):
    # quadrature of degree `deg` for normal distribution with mean `mu` and variance `var`
    xq, wq = np.polynomial.hermite_e.hermegauss(deg)
    wq = wq/np.sqrt(2*np.pi)  # standard Gaussian normalization
    xq = np.sqrt(var)*xq + mu # shift and scale

    return xq, wq

match sigma_str:
    case 'ReLU':
        sigma       = lambda x : np.where(x > 0., x, 0.)
        sigma_prime = lambda x : np.where(x > 0., 1., 0.)

    case 'SiLU':
        sigma       = lambda x : x / (1. + np.exp(-x))
        sigma_prime = lambda x : (1. + (1. + x) * np.exp(-x)) / (1. + np.exp(-x))**2.

    case 'erf':
        sigma = lambda x : erf(x)
        sigma_prime = lambda x : 2. / np.sqrt(np.pi) * np.exp(-x**2.)

    case _:
        raise NotImplementedError(f"sigma_str= {sigma_str}")

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

k0 = get_k0(sigma)
k1 = get_k1(sigma)
ks = get_ks(sigma, k0, k1)

dk0 = get_k0(sigma_prime)
dk1 = get_k1(sigma_prime)
dks = get_ks(sigma_prime, dk0, dk1)

###############################################################

def sample_haar_on(dim):
    A = np.random.randn(dim, dim)
    Q, R = np.linalg.qr(A)
    L = np.sign(np.diag(R))
    return Q * L[None,:]

def multiProcessingSampling(worker, args, seed_varpi, nSim = 50, nProc = None,
                                    errors_ret_dim = 7):

    if nProc is None:
        nProcc = max(1, mp.cpu_count() - 2)
    else:
        nProcc = nProc

    errors_ret = np.zeros((nSim, errors_ret_dim))
    varpi_ret = np.zeros((nSim, k))
    sa_ret = np.zeros(nSim)
    sb_ret = np.zeros((nSim, k))
    fa_ret = np.zeros(nSim)
    fb_ret = np.zeros((nSim, k))
    qa_ret = np.zeros(nSim)
    qb_ret = np.zeros((nSim, k))
    qc_ret = np.zeros((nSim, k, k))

    seeds = np.random.randint(low = 0, high = 2**32 - 1, size = nSim)

    list_of_args = [ (seed_varpi[i], seeds[i], *args) for i in range(nSim) ] 

    with mp.get_context("spawn").Pool(processes = nProcc) as pool:

        for kk, result in enumerate(tqdm.tqdm(pool.imap(worker, list_of_args), total=nSim)):
            errors_ret[kk] = result[:errors_ret_dim]
            varpi_ret[kk] = result[errors_ret_dim].flatten()
            sa_ret[kk] = result[errors_ret_dim + 1].item()
            sb_ret[kk] = result[errors_ret_dim + 2].flatten()
            fa_ret[kk] = result[errors_ret_dim + 3].item()
            fb_ret[kk] = result[errors_ret_dim + 4].flatten()
            qa_ret[kk] = result[errors_ret_dim + 5].item()
            qb_ret[kk] = result[errors_ret_dim + 6].flatten()
            qc_ret[kk] = result[errors_ret_dim + 7]

    return errors_ret, varpi_ret, sa_ret, sb_ret, fa_ret, fb_ret, qa_ret, qb_ret, qc_ret

###############################################################

def generate_and_train(_args):

    seed_varpi = _args[0]
    seed = _args[1]
    ratio_pn = _args[2]

    n = max(int(ratio_nd * d), 1) # no. samples
    p = max(int(ratio_pn * n), 1) # no. features

    # NOTE: fix seed to draw theta0 and Vk to ensure consistent across ratio_pn 
    np.random.seed(seed_varpi)

    theta0 = np.random.randn(d, 1) / np.sqrt(d) # (d, 1)

    # subspace basis for directional Sobolev training
    Vk = np.random.randn(d, k)

    # NOTE: set seed independently of varpi
    # ~ np.random.seed(seed)

    # observation noise
    add_y,       mult_y        = get_noise( n )                                
    add_y_prime, mult_y_prime  = get_grad_noise( n, d, [add_y, mult_y] )       

    # data for training 
    X       = np.random.randn(d, n) # training samples x
    true_features = theta0.T @ X # (1, n)

    y = np.sum( phi(true_features), axis=0 ) # (n, )
    y = mult_y * y + add_y # (n, )

    y_prime = np.outer( theta0, phi_prime(true_features) ) # (d, n)
    VkTy_prime = Vk.T @ y_prime  # (k, n)

    # TODO: change {add/mult}_y_prime to dimension (k, n)
#     VkTy_prime = mult_y_prime * VkTy_prime + add_y_prime  # new samples y_prime       

    # set up random feature matrix
    if rf_str == 'iid_gauss':
        Theta = np.random.randn(d, p) / np.sqrt(d)

    elif rf_str == 'ortho':

        U = sample_haar_on(d)
        V = sample_haar_on(p)
        D = np.zeros((d, p))
        np.fill_diagonal(D, max(1, np.sqrt(p / d)))
        Theta = U.T @ D @ V

    else:
        raise NotImplementedError(f"rf_str = {rf_str}.")


    # compute optimal weights w_hat
    preactivation = Theta.T @ X          # (p, n)
    Z       = sigma(preactivation)       # (p, n)
    Z_prime = sigma_prime(preactivation) # (p, n)
    VkTheta = Vk.T @ Theta               # (k, p)

    alpha = 1. / ratio_pn
    lbda = args.lbda / alpha
    
    # pure L2 training
    if a_1 == 0.:
        if p <= n:
            w_hat = np.linalg.solve(a_0 * Z @ Z.T + lbda * n * np.eye(p), a_0 * Z @ y)

        elif p > n:
            w_hat = Z @ np.linalg.solve(a_0 * Z.T @ Z + lbda * n * np.eye(n), a_0 * y)

    # Sobolev training
    elif a_0 >= 0. and a_1 > 0.:
        if p <= (k+1)*n:
            AAT   = a_0 * Z @ Z.T
            AY    = a_0 * Z @ y

            ZpZp = Z_prime @ Z_prime.T # (p,p)
            for ii in range(k):
                AAT  += a_1 * np.diag(VkTheta[ii,:].flatten()) @ ZpZp @ np.diag(VkTheta[ii,:].flatten())
                AY   += a_1 * np.diag(VkTheta[ii,:].flatten()) @ Z_prime @ VkTy_prime[ii,:]

            w_hat = np.linalg.solve(AAT + lbda * n * np.eye(p), AY)

        elif p > (k+1)*n:
            # assemble regression matrix without for-loops as much as possible
            A = np.zeros( ((k+1)*n, p) )

            A[:n,:] = np.sqrt(a_0) * Z.T
            for ii in np.arange(k):
                start = (ii+1) * n
                end   = (ii+2) * n
                A[start:end,:] = np.sqrt(a_1) * Z_prime.T @ np.diag(VkTheta[ii,:].flatten())

            Y = np.hstack([np.sqrt(a_0) * y, np.sqrt(a_1) * VkTy_prime.flatten()]) # ((k+1)*n,)
            w_hat = A.T @ np.linalg.solve(A @ A.T + lbda * n * np.eye((k+1)*n), Y)
            
    ###########################################
    # return overlap parameters for diagnostics
    varpi = Vk.T @ theta0
    sa = k0 * np.sum(w_hat)
    sb = dk0 * Vk.T @ Theta @ w_hat
    fa = k1 * np.sum(theta0.flatten() * (Theta @ w_hat))
    fb = (dk1 * Vk.T @ Theta @ np.diag(w_hat.flatten()) @ Theta.T @ theta0).flatten()
    qa = np.sum(w_hat * ((ks**2. * np.eye(p) + k1**2. * Theta.T @ Theta) @ w_hat))
    qb = Vk.T @ Theta @ np.diag(w_hat.flatten()) @ (ks * dks * np.eye(p) + k1 * dk1 * Theta.T @ Theta) @ w_hat
    qc = Vk.T @ Theta @ np.diag(w_hat.flatten()) @ (dks**2. * np.eye(p) + dk1**2. * Theta.T @ Theta) @  np.diag(w_hat.flatten()) @ Theta.T @ Vk

    ###########################################
    # training error
    y_hat = np.dot(w_hat, Z)
    y_prime_hat = Theta @ (np.diag(w_hat.flatten()) @ Z_prime)

    eps_train     = np.sum((y - y_hat)**2.) / n   
    eps_train_reg = lbda * np.sum(w_hat**2)
    eps_train_h1  = np.sum(  (y_prime - y_prime_hat)**2.) / n  
    eps_train_h1k = np.sum(  ( VkTy_prime - Vk.T @ y_prime_hat )**2.) / n

    ###########################################
    # generalization error at w_hat

    # noise for generalization error 
    add_y,       mult_y        = get_noise( n_test )                                                
    add_y_prime, mult_y_prime  = get_grad_noise( n_test, d, [add_y, mult_y] )                            

    X       = np.random.randn(d, n_test) # new samples x
    true_features = theta0.T @ X  

    y       = mult_y * np.sum( phi(true_features), axis=0 ) + add_y # new samples y (n, )
    y_prime = np.outer( theta0, np.sum( phi_prime(true_features), axis=0 ) ) # new samples y_prime       

    preactivation = Theta.T @ X
    Z = sigma(preactivation)
    Z_prime = sigma_prime(preactivation)

    y_hat = np.dot(w_hat, Z)
    y_prime_hat = Theta @ (np.diag(w_hat.flatten()) @ Z_prime)

    eps_gen = np.sum((y - y_hat)**2.) / n_test
    eps_gen_h1 = np.sum((y_prime - y_prime_hat)**2.) / n_test
    eps_gen_h1k = np.sum( ( Vk.T @ (y_prime - y_prime_hat) )**2.) / n_test
    
    ###########################################
    # output
    
    ret = eps_train, eps_train_h1, eps_train_h1k, eps_train_reg, eps_gen, eps_gen_h1, eps_gen_h1k
    ret += varpi, sa, sb, fa, fb, qa, qb, qc

    return ret

###############################################################

def single_run(ratio_pn, store_results = True):
    
    n = int(ratio_nd * d)
    p = int(ratio_pn * int(ratio_nd * d))
    alpha = 1. / ratio_pn
    gamma = 1. / (ratio_pn * ratio_nd)
    
    if store_results:
        data_dir = 'data/single_run/alpha_{}_gamma_{}_d_{}_k_{}_sigma_{}_phi_{}_Theta_{}_a0_{}_a1_{}'.format(alpha, gamma, d, k, sigma_str, nonlin_str, rf_str, a_0, a_1)
#         data_dir = '/home/fs1/ts/sobo-rf/data/single_run/alpha_{}_gamma_{}_d_{}_k_{}_sigma_{}_phi_{}_Theta_{}_a0_{}_a1_{}'.format(alpha, gamma, d, k, sigma_str, nonlin_str, rf_str, a_0, a_1)
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)
        import logging
        log_file = data_dir + '/log.log'
        if os.path.isfile(log_file):
            os.remove(log_file)
        targets = logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)
        logging.basicConfig(format='%(message)s', level=logging.INFO, handlers=targets)
        def print(*args, **kwargs):
            logging.info(*args, **kwargs)
    
    print('####################################')
    print('Training random feature models for:')
    print('####################################')
    print('System dimensions')
    print('d      = {}'.format(d))
    print('n      = {}'.format(n))
    print('p      = {}'.format(p))
    print('##')
    print('Ratios')
    print('alpha  = {}'.format(alpha))
    print('gamma  = {}'.format(gamma))
    print('##')
    print('Statistics')
    print('n_stat = {}'.format(n_stat))
    print('n_test = {}'.format(n_test))
    print('##')
    print('Weights for loss function terms')
    print('a_0    = {}'.format(a_0))
    print('a_1    = {}'.format(a_1))
    print('####################################')
    
    seed_varpi = np.random.randint(low = 0, high = 2**32 - 1, size = (args.n_stat))
    ret, varpi, sa, sb, fa, fb, qa, qb, qc = \
                    multiProcessingSampling(generate_and_train, (ratio_pn,),
                                            seed_varpi, nSim = n_stat, nProc = args.nProc)
                                            
                                            
    eps_train_data     = ret[:,0]
    eps_train_h1_data  = ret[:,1]
    eps_train_h1k_data = ret[:,2]
    eps_train_reg_data = ret[:,3]
    eps_gen_data       = ret[:,4]
    eps_gen_h1_data    = ret[:,5]
    eps_gen_h1k_data   = ret[:,6]
    
    print('####################################')
    print('Results for overlap parameters:')
    print('####################################')
    print('Mean varpi = {}'.format(np.mean(varpi, axis = 0)))
    print('Std  varpi = {}'.format(np.std(varpi, axis = 0)))
    print('##')
    print('Mean s_a   = {}'.format(np.mean(sa, axis = 0)))
    print('Std  s_a   = {}'.format(np.std(sa, axis = 0)))
    print('##')
    print('Mean s_b   = {}'.format(np.mean(sb, axis = 0)))
    print('Std  s_b   = {}'.format(np.std(sb, axis = 0)))
    print('##')
    print('Mean f_a   = {}'.format(np.mean(fa, axis = 0)))
    print('Std  f_a   = {}'.format(np.std(fa, axis = 0)))
    print('##')
    print('Mean f_b   = {}'.format(np.mean(fb, axis = 0)))
    print('Std  f_b   = {}'.format(np.std(fb, axis = 0)))
    print('##')
    print('Mean qa   = {}'.format(np.mean(qa, axis = 0)))
    print('Std  qa   = {}'.format(np.std(qa, axis = 0)))
    print('##')
    print('Mean qb   = {}'.format(np.mean(qb, axis = 0)))
    print('Std  qb   = {}'.format(np.std(qb, axis = 0)))
    print('##')
    print('Mean qc   = {}'.format(np.mean(qc, axis = 0)))
    print('Std  qc   = {}'.format(np.std(qc, axis = 0)))
    print('##')
#     data = np.array([np.squeeze(varpi), np.squeeze(sb)]).T
#     print('Cov(varpi, s_b)   = {}'.format(np.cov(data.T)))
    print('####################################')
    print('Results for training and generalization errors:')
    print('####################################')
    print('Mean L2 training error   = {}'.format(np.mean(eps_train_data, axis = 0)))
    print('Std  L2 training error   = {}'.format(np.std(eps_train_data, axis = 0)))
    print('##')
    print('Mean H1k training error  = {}'.format(np.mean(eps_train_h1k_data, axis = 0)))
    print('Std  H1k training error  = {}'.format(np.std(eps_train_h1k_data, axis = 0)))
    print('##')
    print('Mean regularization term = {}'.format(np.mean(eps_train_reg_data, axis = 0)))
    print('Std  regularization term = {}'.format(np.std(eps_train_reg_data, axis = 0)))
    print('##')
    print('Mean L2 generalization error   = {}'.format(np.mean(eps_gen_data, axis = 0)))
    print('Std  L2 generalization error   = {}'.format(np.std(eps_gen_data, axis = 0)))
    print('##')
    print('Mean H1k generalization error  = {}'.format(np.mean(eps_gen_h1k_data, axis = 0)))
    print('Std  H1k generalization error  = {}'.format(np.std(eps_gen_h1k_data, axis = 0)))
    print('##')
    print('####################################')
    print('####################################')
    print('####################################')
    
#     import corner
#     labels = [r'$\varpi$', r'$s_b$']
    plt.figure()
#     corner.corner(data,labels=labels,weights = np.ones(n_stat) / n_stat, bins = 35)
        
    if not store_results:
#         plt.show()
        pass
    
    else:
#         plt.savefig(data_dir + '/varpi-sb-hist.pdf', bbox_inches = 'tight')
#         plt.close()
        
        np.save(data_dir + '/varpi.npy', varpi)
        np.save(data_dir + '/sa.npy', sa)
        np.save(data_dir + '/sb.npy', sb)
        np.save(data_dir + '/fa.npy', fa)
        np.save(data_dir + '/fb.npy', fb)
        np.save(data_dir + '/qa.npy', qa)
        np.save(data_dir + '/qb.npy', qb)
        np.save(data_dir + '/qc.npy', qc)
        
        np.save(data_dir + '/eps-l2-train.npy', eps_train_data)
        np.save(data_dir + '/eps-h1k-train.npy', eps_train_h1k_data)
        np.save(data_dir + '/eps-reg-train.npy', eps_train_reg_data)
        np.save(data_dir + '/eps-l2-gen.npy', eps_gen_data)
        np.save(data_dir + '/eps-h1k-gen.npy', eps_gen_h1k_data)
        
if __name__ == '__main__':
    
#     single_run(args.ratio_pn_start)
#     exit()
    
#     ################################

    now = datetime.now()
    dt_string = now.strftime('%Y_%m_%d_%H_%M_%S')
    data_dir = 'data/{}/{}'.format(base_dir, dt_string)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    if args.seed_varpi is not None:
        seed_varpi = args.seed_varpi * np.ones((args.ratio_pn_num, args.n_stat), dtype='int')
    else:
        seed_varpi = np.random.randint(low = 0, high = 2**32 - 1, size = (args.ratio_pn_num, args.n_stat))
    
    ratio_pn = np.linspace(args.ratio_pn_start, args.ratio_pn_end, args.ratio_pn_num)

    ################################

    eps_train, eps_train_std = np.zeros(args.ratio_pn_num), np.zeros(args.ratio_pn_num) 
    eps_train_data = np.zeros((args.ratio_pn_num, args.n_stat))

    eps_gen, eps_gen_std = np.zeros(args.ratio_pn_num), np.zeros(args.ratio_pn_num) 
    eps_gen_data = np.zeros((args.ratio_pn_num, args.n_stat))

    eps_train_reg, eps_train_reg_std = np.zeros(args.ratio_pn_num), np.zeros(args.ratio_pn_num) 
    eps_train_reg_data = np.zeros((args.ratio_pn_num, args.n_stat))

    eps_train_h1, eps_train_h1_std = np.zeros(args.ratio_pn_num), np.zeros(args.ratio_pn_num) 
    eps_train_h1_data = np.zeros((args.ratio_pn_num, args.n_stat))

    eps_gen_h1, eps_gen_h1_std = np.zeros(args.ratio_pn_num), np.zeros(args.ratio_pn_num) 
    eps_gen_h1_data = np.zeros((args.ratio_pn_num, args.n_stat))

    eps_train_h1k, eps_train_h1k_std = np.zeros(args.ratio_pn_num), np.zeros(args.ratio_pn_num) 
    eps_train_h1k_data = np.zeros((args.ratio_pn_num, args.n_stat))

    eps_gen_h1k, eps_gen_h1k_std = np.zeros(args.ratio_pn_num), np.zeros(args.ratio_pn_num) 
    eps_gen_h1k_data = np.zeros((args.ratio_pn_num, args.n_stat))

    varpi = np.zeros((args.ratio_pn_num, args.n_stat, k))
    sa    = np.zeros((args.ratio_pn_num, args.n_stat))
    sb    = np.zeros((args.ratio_pn_num, args.n_stat, k))
    fa    = np.zeros((args.ratio_pn_num, args.n_stat))
    fb    = np.zeros((args.ratio_pn_num, args.n_stat, k))
    qa    = np.zeros((args.ratio_pn_num, args.n_stat))
    qb    = np.zeros((args.ratio_pn_num, args.n_stat, k))
    qc    = np.zeros((args.ratio_pn_num, args.n_stat, k, k))
    
    ################################

    for i in tqdm.tqdm(range(args.ratio_pn_num)):

        print('ratio_pn = {}'.format(ratio_pn[i]))
        ret, varpi_tmp, sa_tmp, sb_tmp, fa_tmp, fb_tmp, qa_tmp, qb_tmp, qc_tmp = \
                    multiProcessingSampling(generate_and_train, (ratio_pn[i],), \
                    seed_varpi[i], nSim = n_stat, nProc = args.nProc)

        eps_train_data[i]     = ret[:,0]
        eps_train_h1_data[i]  = ret[:,1]
        eps_train_h1k_data[i] = ret[:,2]
        eps_train_reg_data[i] = ret[:,3]
        eps_gen_data[i]       = ret[:,4]
        eps_gen_h1_data[i]    = ret[:,5]
        eps_gen_h1k_data[i]   = ret[:,6]

        eps_train[i]         = np.mean(eps_train_data[i])
        eps_train_std[i]     = np.std(eps_train_data[i])
        eps_train_h1[i]      = np.mean(eps_train_h1_data[i])
        eps_train_h1_std[i]  = np.std(eps_train_h1_data[i])
        eps_train_h1k[i]      = np.mean(eps_train_h1k_data[i])
        eps_train_h1k_std[i]  = np.std(eps_train_h1k_data[i])
        eps_train_reg[i]     = np.mean(eps_train_reg_data[i])
        eps_train_reg_std[i] = np.std(eps_train_reg_data[i])
        eps_gen[i]           = np.mean(eps_gen_data[i])
        eps_gen_std[i]       = np.std(eps_gen_data[i])
        eps_gen_h1[i]        = np.mean(eps_gen_h1_data[i])
        eps_gen_h1_std[i]    = np.std(eps_gen_h1_data[i])
        eps_gen_h1k[i]        = np.mean(eps_gen_h1k_data[i])
        eps_gen_h1k_std[i]    = np.std(eps_gen_h1k_data[i])
        
        varpi[i] = varpi_tmp
        sa[i]    = sa_tmp
        sb[i]    = sb_tmp
        fa[i]    = fa_tmp
        fb[i]    = fb_tmp
        qa[i]    = qa_tmp
        qb[i]    = qb_tmp
        qc[i]    = qc_tmp

        if args.save_constantly or i==(args.ratio_pn_num-1):
            np.save(data_dir + '/dim.npy', d)
            np.save(data_dir + '/dim-{}-a1.npy'.format(d), a_0)
            np.save(data_dir + '/dim-{}-a2.npy'.format(d), a_1)
            np.save(data_dir + '/dim-{}-lbda.npy'.format(d), args.lbda)
            np.save(data_dir + '/dim-{}-Delta.npy'.format(d), Delta)
            np.save(data_dir + '/dim-{}-sigma.npy'.format(d), sigma_str)
            np.save(data_dir + '/dim-{}-nonlin.npy'.format(d), nonlin_str)
            np.save(data_dir + '/dim-{}-theta.npy'.format(d), rf_str)
            np.save(data_dir + '/dim-{}-n-stat.npy'.format(d), n_stat)
            np.save(data_dir + '/dim-{}-ratio-nd.npy'.format(d), ratio_nd)
            np.save(data_dir + '/dim-{}-ratio-pn.npy'.format(d), ratio_pn)
            np.save(data_dir + '/dim-{}-k.npy'.format(d), k)
            np.save(data_dir + '/dim-{}-noise.npy'.format(d), noise_desc)
            np.save(data_dir + '/dim-{}-noise_abbreviation.npy'.format(d), noise_desc_short)
            
            np.save(data_dir + '/dim-{}-eps-train.npy'.format(d), eps_train)
            np.save(data_dir + '/dim-{}-eps-train-std.npy'.format(d), eps_train_std)
            np.save(data_dir + '/dim-{}-eps-train-data.npy'.format(d), eps_train_data)
            np.save(data_dir + '/dim-{}-eps-train-h1.npy'.format(d), eps_train_h1)
            np.save(data_dir + '/dim-{}-eps-train-h1-std.npy'.format(d), eps_train_h1_std)
            np.save(data_dir + '/dim-{}-eps-train-h1-data.npy'.format(d), eps_train_h1_data)
            np.save(data_dir + '/dim-{}-eps-train-h1k.npy'.format(d), eps_train_h1k)
            np.save(data_dir + '/dim-{}-eps-train-h1k-std.npy'.format(d), eps_train_h1k_std)
            np.save(data_dir + '/dim-{}-eps-train-h1k-data.npy'.format(d), eps_train_h1k_data)
            np.save(data_dir + '/dim-{}-eps-train-reg.npy'.format(d), eps_train_reg)
            np.save(data_dir + '/dim-{}-eps-train-reg-std.npy'.format(d), eps_train_reg_std)
            np.save(data_dir + '/dim-{}-eps-train-reg-data.npy'.format(d), eps_train_reg_data)
            np.save(data_dir + '/dim-{}-eps-gen.npy'.format(d), eps_gen)
            np.save(data_dir + '/dim-{}-eps-gen-std.npy'.format(d), eps_gen_std)
            np.save(data_dir + '/dim-{}-eps-gen-data.npy'.format(d), eps_gen_data)
            np.save(data_dir + '/dim-{}-eps-gen-h1.npy'.format(d), eps_gen_h1)
            np.save(data_dir + '/dim-{}-eps-gen-h1-std.npy'.format(d), eps_gen_h1_std)
            np.save(data_dir + '/dim-{}-eps-gen-h1-data.npy'.format(d), eps_gen_h1_data)
            np.save(data_dir + '/dim-{}-eps-gen-h1k.npy'.format(d), eps_gen_h1k)
            np.save(data_dir + '/dim-{}-eps-gen-h1k-std.npy'.format(d), eps_gen_h1k_std)
            np.save(data_dir + '/dim-{}-eps-gen-h1k-data.npy'.format(d), eps_gen_h1k_data)

            np.save(data_dir + '/dim-{}-varpi.npy'.format(d), varpi)
            np.save(data_dir + '/dim-{}-sa.npy'.format(d), sa)
            np.save(data_dir + '/dim-{}-sb.npy'.format(d), sb)
            np.save(data_dir + '/dim-{}-fa.npy'.format(d), fa)
            np.save(data_dir + '/dim-{}-fb.npy'.format(d), fb)
            np.save(data_dir + '/dim-{}-qa.npy'.format(d), qa)
            np.save(data_dir + '/dim-{}-qb.npy'.format(d), qb)
            np.save(data_dir + '/dim-{}-qc.npy'.format(d), qc)

    np.savetxt(data_dir + '/dim-{}-seed-varpi.txt'.format(d), seed_varpi, fmt="%i",)

    plt.figure()
    plt.errorbar(ratio_pn, eps_train, yerr = 1.96 * eps_train_std / np.sqrt(n_stat), label = r'$L^2$ error (train)', color = 'steelblue', capsize = 2, linestyle = 'None', marker = 's')
    plt.errorbar(ratio_pn, eps_train_h1k, yerr = 1.96 * eps_train_h1k_std / np.sqrt(n_stat), label = r'$H^{1,k}$ semi error (train)', color = 'orange', capsize = 2, linestyle = 'None', marker = '^')
    plt.errorbar(ratio_pn, eps_train_reg, yerr = 1.96 * eps_train_reg_std / np.sqrt(n_stat), label = r'$\frac{\lambda}{2} \left \Vert \hat{w} \right \Vert^2$', color = 'navy', capsize = 2, linestyle = 'None', marker = '.')
    plt.errorbar(ratio_pn, eps_gen, yerr = 1.96 * eps_gen_std / np.sqrt(n_stat), label = r'$L^2$ error (gen)', color = 'seagreen', capsize = 2, linestyle = 'None', marker = 'o')
    plt.errorbar(ratio_pn, eps_gen_h1k, yerr = 1.96 * eps_gen_h1k_std / np.sqrt(n_stat), label = r'$H^{1,k}$ semi error (gen)', color = 'firebrick', capsize = 2, linestyle = 'None', marker = 'x')
    plt.grid()
    plt.xlabel(r'$\alpha^{-1} = p / n$')
    plt.ylabel(r'$\varepsilon$')
    plt.xlim(np.amin(ratio_pn), np.amax(ratio_pn))
    #plt.ylim(-0.01, min(1, 1.05 * max(np.amax(eps_gen), np.amax(eps_gen_h1))))
    plt.ylim(0., 3.)
    plt.legend()
    plt.title(r'\begin{{center}}$d = {}$, $\lambda = {}$, $\Delta = {}$, $n/d = {}$, $k = {}$\\  $\sigma =$ {}, $\Theta =$ {}, $\phi =$ {},\\ $a_0 = {}$, $a_1 = {}$, {}, Grad Noise {}, \end{{center}}'.format(d, args.lbda, Delta, ratio_nd, k, sigma_str, rf_str, nonlin_str, a_0, a_1, noise_desc, grad_noise_model, ), wrap = True, fontsize = 10)
    plt.savefig(data_dir + '/dim-{}-lbda-{}-delta-{}-nd-{}-k-{}-sigma-{}-theta-{}-a0-{}-a1-{}-noise-{}-gradnoise-{}.pdf'.format(d, args.lbda, Delta, ratio_nd, k, sigma_str, rf_str, a_0, a_1, noise_desc_short, grad_noise_model, ), bbox_inches = 'tight')
    plt.close()
