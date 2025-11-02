using LinearAlgebra
using Random
using Distributions
using Einsum
using ProgressBars
using Statistics
using CSV
using DataFrames
using SpecialFunctions
using FastGaussQuadrature
using SparseArrays
using Folds
using PyCall

opt = pyimport("scipy.optimize")

# ========================================================================
# Recording
# ========================================================================

function makeFile(headers,name)

    df = DataFrame()
    [df[!,h] = Any[]  for h in headers ]
    addToFile(df,name)
end

function addToFile(df::DataFrame,name)

    if isfile(name)
        CSV.write(name, df, append=true )
    else
        CSV.write(name, df )
    end
end

function addToFile(df::AbstractArray,name)
    addToFile( DataFrame( reshape(df,1,:) , :auto  ) , name )
end

# filter dataframe by column values
function multifilter(df, filCols, selectors; leftover=false)

    function filterRules(cols...)::Bool
        (cols) in selectors
    end

    return leftover ? (filter( filCols => filterRules , df ), filter( filCols => !filterRules , df )) : filter( filCols => filterRules , df )

end

#--------------------------------------------------------------
# Special Functions
# -------------------------------------------------------------

# NOTE: can also use julia functions
#         identity(x) = x
#         one(x)      = 1
#         erf(x)      = error function
#         tanh(x)     = hyperbolic tangent


# change data type
r(x)      = Int64(round(x))
squish(a) = collect(Iterators.flatten(a))

# activation functions
swish( x;β=1) = x / ( 1 + exp(-β*x)  )
relu(x)       = x>0 ? x : 0.

# derivatives of nonlinear functions
d_relu(x)      = x>0 ? 1. : 0.
d_swish(x;β=1) = ( 1 + exp(-β*x) + x*β*exp(-β*x) ) / ( 1 + exp(-β*x)  )^2
d_erf(x)       = ( 2 / sqrt(pi) )* exp( -x^2 )
d_tanh(x)      = 1 - (tanh(x))^2
d_atan(x)      = 1. / ( 1. + x^2)

# numerical estimation of Hermite coefficients
function set_Hermite(σ ; s=31)
    z,w = gausshermite( s ; normalize=true )
    Κ₀  = w' * σ.(z)  
    Κ₁  = w' *( z .* σ.(z) )
    Κₛ  = sqrt( w'*( σ.(z).^2 ) - Κ₀^2 - Κ₁^2 )
    return Κ₀, Κ₁, Κₛ
end

#--------------------------------------------------------------
# Stieltjes transforms
# -------------------------------------------------------------

function gμ(   z ; γ) 
    s = 1/γ
    return ( s*(1-γ) - z - sqrt( (z - s*(1. + γ))^2 - 4γ*s^2 )) / (2γ*z*s)
end

function dgμ( z ; γ) 
    s = 1/γ
    
    f = s*(1-γ) - z - sqrt( (z-s*(1+γ))^2 - 4γ*s^2 )
    g = 2*γ*z*s

    df = -1 - (z - s*(γ+1)) / sqrt( (z - s*(γ+1))^2 - 4γ*s^2 )
    dg = 2γ*s

    return (df*g - f*dg) / g^2
end

#--------------------------------------------------------------
# Solve for V
#--------------------------------------------------------------

V2V̂( V ; settings ) = settings.α/( 1 .+ V) 

function V̂2V( V̂ ; settings )
    κ₁, κₛ, γ = settings.κ₁, settings.κₛ, settings.γ
    z         = ( settings.λ + κₛ^2*V̂ ) / ( κ₁^2 *V̂ )
    
    return  ( γ / V̂ ) * (1 - z*gμ(-z ; γ))  +  ( γ*κₛ^2 /(κ₁^2*V̂) ) * ( 1/z *(1/γ - 1) + gμ(-z ; γ) )
end


function V_system( V, settings )
    V̂   = V2V̂.( V ; settings )
    ret = V̂2V.( V̂ ; settings ) .- V
    return ret
end


function solver( eq_sys ; settings, tol=1e-8, max_steps=100000, η=0.7 )
    state  = 1.
    res    = 1
    steps  = 0
    while res > tol && steps < max_steps
        update = eq_sys( state, settings )
        res    = maximum(abs(update))
        state  = (1-η)*(state+update) + η*state
        steps += 1
    end
    steps == max_steps && error("time out.")

    return state
end


#--------------------------------------------------------------
# Solve for f
#--------------------------------------------------------------

function f̂2f( V̂, f̂ ; settings )
    
    κ₁, κₛ = settings.κ₁, settings.κₛ
    z      = (settings.λ + V̂*κₛ^2 ) / (V̂*κ₁^2)
    
    return (f̂ / V̂) * (1. - z * gμ(-z, γ=settings.γ) )
end


#--------------------------------------------------------------
# Solve for q
#--------------------------------------------------------------


function VV̂f2q( V, V̂, f̂, f ; settings )
    κ₁, κₛ, κ₀             = settings.κ₁, settings.κₛ, settings.κ₀
    γ, k, λ, α, Cₙ11, Cₙ22 = settings.γ, settings.k, settings.λ, settings.α, settings.Cₙ11, settings.Cₙ22

    rhs_q̂  = α / (1. + V)^2 * ( Cₙ11 .- 2f*settings.Edϕ .+ settings.var_ϕ .+ (abs.(κ₀)<1e-8)*settings.Eϕ^2 )
    z      = (λ + V̂*κₛ^2 ) / (V̂*κ₁^2)
    rhs_q  = (f̂^2 / V̂^2) * (1 - 2z*gμ(-z;γ) + z^2*dgμ(-z;γ))   + (f̂^2*κₛ^2/ (κ₁^2 * V̂^2)) * (gμ(-z;γ) - z*dgμ(-z;γ))

    # Mass matrix
    A11  = -α / (1. + V)^2.
    A22  =   (-γ / V̂^2)*(1 - 2*z*gμ(-z;γ) + (z^2)*dgμ(-z;γ))
    A22 +=  (-γ*κₛ^4/(κ₁^4*V̂^2))*(1/z^2 * (1/γ-1) + dgμ(-z;γ))
    A22 +=  (-2*γ*κₛ^2 /( κ₁^2*V̂^2))*(gμ(-z;γ) - z*dgμ(-z;γ))


    Aq⁻¹ = 1/( A11*A22 - 1 ) *  [   A22    -1.  ;
                                    -1.    A11  ]

    q  = getindex.(  [ Aq⁻¹ * [rhs_q̂i; rhs_q] for rhs_q̂i in rhs_q̂ ], 1 )
    q̂  = α * q / (1. + V)^2  + rhs_q̂

    return q̂, q
end



#--------------------------------------------------------------
# Outer loop
#--------------------------------------------------------------

function predict_error( settings )

    # fixed point system
    #V  = solver( V_system ; settings  )
    V  = opt.root( V_system, 0.01, args=(settings),method="excitingmixing", tol=1e-7)["x"]
    V̂  = V2V̂( V ; settings )

    # covariance between truth and y predictions
    f̂  = V̂ .* settings.Edϕ 
    f  = f̂2f( V̂, f̂ ; settings )

     # problem parameters
    κ₁, κₛ, dκ₁, dκₛ       = settings.κ₁, settings.κₛ, settings.dκ₁, settings.dκₛ
    γ, k, λ, α, Cₙ11, Cₙ22 = settings.γ, settings.k, settings.λ, settings.α, settings.Cₙ11, settings.Cₙ22



    # covariance between predictions
    q̂, q = VV̂f2q( V[1], V̂, f̂, f ; settings )
        
    # problem parameters
    κ₁, κₛ, dκ₁, dκₛ, κ₀   = settings.κ₁, settings.κₛ, settings.dκ₁, settings.dκₛ, settings.κ₀
    γ, k, λ, α, Cₙ11, Cₙ22 = settings.γ, settings.k, settings.λ, settings.α, settings.Cₙ11, settings.Cₙ22
    Eϕ, Edϕ, var_ϕ, var_dϕ = settings.Eϕ, settings.Edϕ, settings.var_ϕ, settings.var_dϕ 

    # training error
    
    z        = (λ + V̂*κₛ^2 ) / (V̂*κ₁^2)
    ŵ²       = γ*κₛ^2 / (κ₁^4. * V̂^2) * q̂*((1/z^2)*(1/γ -1) .+ dgμ(-z;γ))  + (f̂^2 .+ γ*q̂) / (κ₁^2 * V̂^2) * (gμ(-z;γ) - z*dgμ(-z;γ))
    ε_tr_ℓ₂  = q̂ / α   # TODO factor of 0.5?
    ε_tr_reg = λ/(2α)*ŵ²

    # ℓ₂ testing error
    ε_gn_ℓ₂    = Cₙ11 + q .+ (var_ϕ  + (abs(κ₀)<1e-8)*Eϕ^2 - 2*Edϕ*f)  
    ε_gn_H₁ᵏ_0 = k*( settings.Cₙ22  +  ŵ²*( dκ₁^2 + dκₛ^2  ))
    ε_gn_H₁ᵏ_2 = [  fill( var_dϕ + Edϕ^2 - 2*Edϕ*f + f^2, length(Cₙ22))    -(Edϕ - f)*sqrt.(q - κₛ^2*ŵ² .- f^2)    q - κₛ^2*ŵ² .- f^2 ]

    return [ε_tr_ℓ₂  ε_tr_reg  ε_gn_ℓ₂  ε_gn_H₁ᵏ_0  ε_gn_H₁ᵏ_2  q  q̂],  [f  f̂  V  V̂]
end

#--------------------------------------------------------------
# Settings
#--------------------------------------------------------------

struct settings
    Cₙ11
    Cₙ22
    λ::Number
    α::Number
    γ::Number
    d::Number
    κ₀::Number
    κ₁::Number
    κₛ::Number
    dκ₀::Number
    dκ₁::Number
    dκₛ::Number
    k::Number
    Eϕ::Number
    Edϕ::Number
    Ed2ϕ::Number
    var_ϕ::Number
    var_dϕ::Number
end

function start_experiment(name ; k )
    makeFile( vcat(  "p/n", "n/d", "lambda", "k", "sigma", "phi", "seed", "Delta_a", "Delta_b",
                     "eps_tr_l2", "eps_tr_reg", "eps_gn_l2", "eps_gn_H1k_0", "eps_gn_H1k_a", "eps_gn_H1k_b", "eps_gn_H1k_c", 
                     "q", "q_hat", "f", "f_hat", "V", "V_hat" ), name )
end

function experiment( pn_ratio::AbstractArray,
                     nd_ratio::AbstractArray,
                     λ_vec   ::AbstractArray,
                     Δ_vec   ::AbstractArray,
                     name    ::String       ;
                     σ          = erf,
                     dσ         = d_erf,
                     ϕ          = "id",
                     seed       = 11,
                     d          = 500,
                     k          = 1  )

    # Hermite coefficients
    Random.seed!(seed)
    κ₀,  κ₁,  κₛ  = set_Hermite(σ)
    dκ₀, dκ₁, dκₛ = set_Hermite(dσ)

    # saved Gaussian integrals
    m = multifilter( CSV.read("phi_moments.csv", DataFrame), ["phi"], [(ϕ,)] )

    # loop test settings
    for ( λ, pn, nd ) in Iterators.product( λ_vec, pn_ratio, nd_ratio )
        α     = 1/pn
        γ     = 1 / (pn*nd)
      
        info  = settings( Δ_vec.^2, Δ_vec.^2, λ, α, γ, d, κ₀,  κ₁,  κₛ, dκ₀, dκ₁, dκₛ, k, 
                          m[1,"E_phi"], m[1,"E_phi_prime"], m[1,"E_phi_prime_prime"], m[1,"var_phi"], m[1,"var_phi_prime"] )
        errs  = predict_error( info  )
        
        for i in 1:length(Δ_vec)
            addToFile( hcat( pn, nd, λ, k, σ, ϕ, seed, Δ_vec[i].^2, Δ_vec[i].^2, errs[1][i,:]', errs[2] ), name )
        end
    end
end
