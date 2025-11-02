# sobolev-random-features

This repostitory retains code which a fixed point system to obtain precise asymptotic representations of the error curves produced by Sobolev training of a random features neural network model. There are also numerical training scripts which verify the theoretical predictions. Implementations are provided in both Julia and Python. Descriptions of how to run code are found in the ```src_julia``` and ```src_python``` directories, respectively. Details on derivations, notation, and results can be found in the paper:

Fisher, K.E., Li M.T.C., Marzouk, Y.M., and Schorlepp, T. (2025). *Precise asymptotic analysis of Sobolev training for random feature models*. [details TBA]

# Brief Definitions

Sobolev training - setting the parameters of a model by minimizing a loss function that matches (1) data labels y to model predictions and (2) gradients of data with respect to input x to model gradients

Random features - a two layer neural network where the features of the first layer are drawn from a prescribed distribution and fixed

