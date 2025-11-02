# Contents

## Scripts for running tests

Edit these scripts to choose details of test hyperparameters---for instance, network size, observation noise level, and activation. The details of possible settings are noted in the scripts.

* run_l2_theory.jl - set up and run a number of tests to produce precise asymptotics of training and generalization error of $L^2$ training
* run_sobo_theory.jl - set up and run a number of tests to produce precise asymptotics of training and generalization error of Sobolev training
* run_rf-reg.jl - set up and run a number of tests where random features models are numerically trained according to either $L^2$ or Sobolev training

## Implementation of tests
* l2_theory.jl - functions used to solve the fixed points system of overlap parameters for $L^2$ training 
* sobo_theory.jl - functions used to solve the fixed points system of overlap parameters for Sobolev training
* phi_moments.csv - precomputed moments related to a number of observations models, used by ```l2_theory.jl``` and ```sobo_theory.jl``` 
* rf-reg.jl - functions used for empirical training of random features model

Note: in ```l2_theory.jl``` and ```sobo_theory.jl```, the fixed points system is solved with the exciting mixing algorithm implemented in ```scipy.optimize```. This functionality is ported to Julia through ```PyCall```.

# How to run scripts

To run in Julia REPL:

```julia
julia --project=</path/to/project> --threads <number of threads>
include("<filename>")
```


To run from command line in serial:

```julia
julia <filename>
```


To run from command line and parallelize over tests:

```julia
julia <filename> <task id>  <total number of tasks>
```

The output will be one .csv file per task with all metadata and results, named according to settings in the run script. Output is recorded after each test for a given set of hyperparameters.
