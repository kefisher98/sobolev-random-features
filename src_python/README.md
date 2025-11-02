### Multiprocessing and Numpy

In order to take advantage of multiprocessing in python, you must ensure that numpy operates in "single-threaded" mode. This can be achieved by calling the following (which you may want to create an alias for):

`>> MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python rf-reg.py [...]`

### Theory scripts

The saddle point system for $L_2$ training is implemented in `rf-theory.py` and for (projected) Sobolev training in `sobo-rf-theory.py`. These scripts are called by running, e.g.,
```
>> python sobo-rf-theory.py 1e-6 iid_gauss --k 2 --nonlin_str arctan --sigma_str erf --ratio_pn_num 10
```
To compute the statistics only for a single $p/n$, e.g. $p/n = 2.5$, you can specify the flags to `--ratio_pn_start 2.5 --ratio_pn_num 1`.

In the instances where the statistics are random variables of the alignment $\varpi$ --- both the $L_2$ and $H_1^k$ training and generalization errors in Sobolev training, and the $H_1^k$ generalization error in $L_2$ training --- the scripts return the constant coefficient `eps_test_H1k_0` and quadratic coefficient `eps_test_H1k_2` for each ratio $p/n$. The random error can then be sampled using the push-forward map
```
eps_test_H1k = eps_test_H1k_0 + eps_test_H1k_2 * (varpi.T @ varpi)
```
where `varpi` is a standard $k$-dimensional Gaussian random variable.

**Note**: running the Sobolev script with `--k 3` (or possibly higher) seems to require a long time for the root finding algorithm to converge to the optimal overlap parameters. Perhaps this is because our Stieljes transforms use only ~3000 Monte Carlo samples, and thus the variance of our estimator is too large? Or perhaps we need to fix the seed for these samples, similar to how we approached this for our previous Monte Carlo implementation of the theory scripts.

