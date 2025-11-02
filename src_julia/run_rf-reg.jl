include("rf-reg.jl")

###############################################################
# user defined parameters
###############################################################

mode     = un_aligned         # approach to projecting gradient observations          (choose from [aligned, un_aligned, Z]) ??
ϖ_vec    = [1.]               # for aligned mode, ϖ values to test                    (array of Floats) 
seed     = 11                 # random seed                                           (positive integer)
n_loops  = 160                # number of indpendent draws of set ups to average over (positive integer)
k        = 1                  # dimension for gradient projection                     (positive integer)
ℓ²       = 1.                 # weight for L² component of objective                  (Float)
H₁ᵏ      = 1.                 # weight for Hₖ¹ component of objective                 (Float)
ϕ        = lingauss           # truth nonlinearity                                    (choose from [id,atan,swish,lingauss,cosh⁻¹,fourier,erf,tanh]) **    
dϕ       = d_lingauss         # derivative of truth nonlinearity                      (choose from [one,d_atan,d_swish,d_lingauss,d_cosh⁻¹,d_fourier,d_erf])
d_vec    = [200]              # vector of input dimensions to test                    (array of positive integers)  
nd_ratio = [8.5]              # vector of n/d ratios to test                          (array of Floats)             ++
pn_ratio = range(0.1,4.,21)   # vector of p/n ratios to test                          (array of Floats)             @@
Δ_vec    = [0.0]              # vector of observation noise standard deviations       (array of Floats)
λ_vec    = [1e-6]             # vector of regularization strengths to test            (array of Floats)
σ        = swish              # activation function                                   (choose from [relu,swish,erf,tanh])
dσ       = d_swish            # derivative of activation functions                    (choose from [d_relu,d_swish,d_erf,d_tanh])


# assign p/n ratios for local process
if length(ARGS) > 0
    my_task_id   = parse(Int64, ARGS[1] ) + 1
    num_tasks    = parse(Int64, ARGS[2] )
    my_pn_ratio  = pn_ratio[ my_task_id:num_tasks:length(pn_ratio) ]
else
    my_task_id   = 1
    my_pn_ratio  = pn_ratio
end
name = string( "rf-reg_k", k, "_", my_task_id, ".csv" )  # name for output file

###############################################################
# run experiments
###############################################################

start_experiment( name )
experiment( mode, d_vec, pn_ratio, nd_ratio, λ_vec, Δ_vec, ϖ_vec, name ; ϕ, dϕ, k, n_loops, seed, σ, dσ, ℓ², H₁ᵏ )

###############################################################
# additional notes
###############################################################
#=

?? unaligned: the projection Vₖ has independent standard Gaussian components
   aligned:   the projection Vₖ has a rank one component aligned with the true feature θ₀ such that Vₖᵀθ₀=ϖ
   Z:         in simulation, replace Vₖᵀθ₀ with a standard Gaussian random variable

** id(ω)=ω,  lingauss(ω) = ω/2-exp(-ω²/2), fourier(ω) = cos(ω)+sin(ω)
   atan(ω)= tan⁻¹(ω), erf(x) = 2/√π ∫ˣ₀ exp(-t²)dt, cosh⁻¹(ω)=cosh⁻¹(ω), swish( x;β=1) = x / ( 1 + exp(-β*x)  )  

++ n = number of training points, d = input dimension

@@ p = number of parameters to train, n = number of training points

=#

