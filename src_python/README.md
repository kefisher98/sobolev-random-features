# Content

For details on possible hyperparameters, see the argument parsing protocols at the top of each script.

* sobo-rf-theory.py - minimal fixed point system for Sobolev training, implemented asymptotically via operator-valued Cauchy transforms
* rf-theory.py - minimal fixed point system for $L^$ training, implemented asymptotically via operator-valued Cauchy transforms
* rf-reg.py -  Monte Carlo simulations for Sobolev training with uninformed gradient model, with gradient projections "v = N(0, I_d)" for each column of Vk
* texSettings.py - formatting details


# Running scripts


## theory

The saddle point system for $L_2$ training is implemented in `rf-theory.py` and for (projected) Sobolev training in `sobo-rf-theory.py`. These scripts are called by running, e.g.,
```python
>> python sobo-rf-theory.py 1e-6 iid_gauss --k 2 --nonlin_str arctan --sigma_str erf --ratio_pn_num 10
```
The first two argumenst are regularization strength,and distribution for random features. Results and metadata are collected as .npy files in a timestamped data folder. Currently, the scripts are configured to write results only after all experiments are completed.

To compute the statistics only for a single $p/n$, e.g. $p/n = 2.5$, you can specify the flags to `--ratio_pn_start 2.5 --ratio_pn_num 1`.

In the instances where the statistics are random variables of the alignment $\varpi$ --- both the $L_2$ and $H_1^k$ training and generalization errors in Sobolev training, and the $H_1^k$ generalization error in $L_2$ training --- the scripts return the constant coefficient `eps_test_H1k_0` and quadratic coefficient `eps_test_H1k_2` for each ratio $p/n$. The random error can then be sampled using the push-forward map
```python
eps_test_H1k = eps_test_H1k_0 + eps_test_H1k_2 * (varpi.T @ varpi)
```
where `varpi` is a standard $k$-dimensional Gaussian random variable.

## Numerics

Numerical Sobolev training is implemented via `rf-reg.py` and may be called via
```python
>> python rf-reg.py 300 1e-6 iid_gauss --k 2 --nonlin_str arctan --sigma_str erf --ratio_pn_num 10
```
The first three argumenst are input dimension, regularization strength, and distribution for random features.
 
## Multiprocessing and Numpy

In order to take advantage of multiprocessing in python, you must ensure that numpy operates in "single-threaded" mode. This can be achieved by calling the following (which you may want to create an alias for):

`>> MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python rf-reg.py [...]`



