include("l2_theory.jl")

###############################################################
# user defined parameters
###############################################################


k          = 1                        # dimension of gradient projection   (integer)          ??
seed       = 12                       # random seed                        (integer)
σ          = swish                    # activation function                (choose from [erf,tanh,swish,relu])
dσ         = d_swish                  # derivative of activation function  (choose from [d_erf,d_tanh,d_swish,d_relu])
ϕ          = "lingauss"               # truth nonlinearity                 (choose from ["id","arctan","om2","cos+sin","erf","acosh","lingauss"]) ** 
nd_ratio   = [2.345]                  # ratios of n/d to test              (array of Floats)  ++
pn_ratio   = logrange(0.01,1000,171)  # ratios of p/n to test              (array of floats)  @@
λ_vec      = [1e-6]                   # regularization strengths to test   (array of Floats)
Δ_vec      = [0.,0.2,0.4,0.6,0.8,1.0] # noise standard deviations to test  (array of Floats)


# assign p/n ratios for local process
if length(ARGS) > 0
    my_task_id   = parse(Int64, ARGS[1] ) + 1
    num_tasks    = parse(Int64, ARGS[2] )
    my_pn_ratio  = pn_ratio[ my_task_id:num_tasks:length(pn_ratio) ]
else
    my_task_id   = 1
    my_pn_ratio  = pn_ratio
end
name = string( "L2_", k, "_", my_task_id, ".csv" )   # name for saving results


###############################################################
# run experiments
###############################################################

start_experiment(name ; k )
experiment( my_pn_ratio, nd_ratio, λ_vec, Δ_vec, name ; σ, dσ, ϕ, k, seed )

###############################################################
# additional notes
###############################################################
#=

?? though this code runs L² training, k determines the projection size for prediction 

** id(ω)=ω,  om2(ω) = ω² - 1, lingauss(ω) = ω/2-exp(-ω²/2), cos+sin(ω) = cos(ω)+sin(ω)
   arctan(ω)= tan⁻¹(ω), erf(x) = 2/√π ∫ˣ₀ exp(-t²)dt, acosh(ω)=cosh⁻¹(ω)

++ n = number of training points, d = input dimension

@@ p = number of parameters to train, n = number of training points

=#


