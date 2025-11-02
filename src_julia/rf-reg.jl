

# how to import:
# julia --project --threads <# of threads>
# include("rf-reg.jl")


using Statistics
using LinearAlgebra
using Distributions
using Random
using CSV
using DataFrames
using SpecialFunctions
using Plots, Plots.Measures
using Combinatorics
using Kronecker
using NPZ

# ========================================================================
# Support
# ========================================================================

# for short
r(x)      = Int64(round(x))
squish(a) = collect(Iterators.flatten(a))
id(x)     = x

# activation functions; from julia, we also have erf(x) and tanh(x)
swish( x;β=1) = x / ( 1 + exp(-β*x)  ) 
relu(x)       = x>0 ? x : 0.
cosh⁻¹(x)     = 1. / cosh(x)
fourier(x)    = cos(x) + sin(x)
lingauss(x)   = x/2 - exp(-x^2/2)

# derivatives of activation functions
d_relu(x)      = x>0 ? 1. : 0.
d_swish(x;β=1) = ( 1 + exp(-β*x) + x*β*exp(-β*x) ) / ( 1 + exp(-β*x)  )^2
d_erf(x)       = ( 2 / sqrt(pi) )* exp( -x^2 )
d_tanh(x)      = 1 - (tanh(x))^2
d_atan(x)      = 1. / ( 1. + x^2)
d_cosh⁻¹(x)    = -sech(x)*tanh(x)
d_fourier(x)   = cos(x) + sin(x)
d_lingauss(x)  = 1/2 + x*exp(-x^2/2)



function sample_haar( dim  )
    A = rand( Normal(), dim, dim )
    F = qr(A)
    L = sign.(diag(F.R))
    return F.Q .* L'
end



# ========================================================================
# Structs 
# ========================================================================

struct un_aligned
    n::Number
    p::Number
    d::Number
    k::Number
    k₀::Number
    λ::Number
    Δ::Number
    σ::Function
    dσ::Function
    ϕ::Function
    dϕ::Function
    ϖ::Number
    ℓ²::Number
    H₁ᵏ::Number
end

struct aligned
    n::Number
    p::Number
    d::Number
    k::Number
    k₀::Number
    λ::Number
    Δ::Number
    σ::Function
    dσ::Function
    ϕ::Function
    dϕ::Function
    ϖ::Number
    ℓ²::Number
    H₁ᵏ::Number
end

struct Z
    n::Number
    p::Number
    d::Number
    k::Number
    k₀::Number
    λ::Number
    Δ::Number
    σ::Function
    dσ::Function
    ϕ::Function
    dϕ::Function
    ϖ::Number
    ℓ²::Number
    H₁ᵏ::Number
end

# ========================================================================
# Numerical Set up
# ========================================================================

function instance( info )
    
    # convenience
    n = info.n
    p = info.p
    d = info.d

    # design
    Θ = rand( Normal(), d, p ) / sqrt(d) 
    X = rand( Normal(), d, n )

    # observation
    #Random.seed!(15)
    θ₀     = rand( Normal(), d, info.k₀ ) / sqrt(d)
    y      = sum( info.ϕ.( X'*θ₀), dims=2 )   +   info.Δ*rand( Normal(), n )
    dy, Vₖ = observe_grad( X, θ₀, info  )

    return X, Θ, θ₀, y, dy, Vₖ 
end

# ------------------------------------------------------------------------
# Gradient Observation Models
# ------------------------------------------------------------------------


function observe_grad( X, θ₀, info::un_aligned )
    #Random.seed!(23)
    Vₖ =  rand( Normal(), info.d, info.k )
    return ( info.dϕ.( X' * θ₀ ) .* θ₀'   +  info.Δ * (rand( Normal(), info.n, info.d ) / sqrt(info.d)  )) * Vₖ,   (Vₖ,)
end
observe_grad( X, θ₀, Vₖ, info::un_aligned )  =  ( info.dϕ.( X' * θ₀ ) .* θ₀'   +  info.Δ * (rand( Normal(), size(X,2), info.d ) / sqrt(info.d)  )) * Vₖ[1]



function observe_grad( X, θ₀, info::aligned )
    d  = info.d
    ϖ  = info.ϖ
    Vₖ = sqrt(d)*ϖ * θ₀  .+  sqrt(1 - (ϖ )^2)*rand( Normal(), d, info.k )
    return ( info.dϕ.( X' * θ₀ ) .* θ₀'   +  info.Δ * rand(Normal(), info.n, d) )  * (Vₖ/sqrt(d)),   (Vₖ,)
end
observe_grad( X, θ₀, Vₖ, info::aligned )  =  ( info.dϕ.( X' * θ₀ ) .* θ₀'   +  info.Δ * rand(Normal(), size(X,2), info.d) ) * ( Vₖ[1] / sqrt(info.d) )


function observe_grad( X, θ₀, info::Z )
    Vₖ = rand( Normal(), info.d, info.k )
    Z  = rand( Normal(), info.k )
    return info.dϕ.( X' * θ₀ ) * Z'   +  info.Δ * rand( Normal(), info.n, info.k ),  (Vₖ,Z)
end
observe_grad( X, θ₀, Vₖ, info::Z )  =  info.dϕ.( X' * θ₀ ) * Vₖ[2]'   +  info.Δ * rand( Normal(), size(X,2), info.k )




# ========================================================================
# Construct ŵ
# ========================================================================


function get_ŵ( Θ::AbstractArray, X::AbstractArray, λ::Number, y::AbstractArray, dy::AbstractArray, Vₖ::AbstractArray, info  )

    # if ridgeless and overparameterized
    λ==0 && info.p>info.n*(1+info.k) && return get_ŵ( Θ, X, y, dy, Vₖ, info )

    M    = Θ'*X
    Z    = info.ℓ²  * info.σ.( M )
    dZ   = info.H₁ᵏ * info.dσ.(M)
    VₖᵀΘ =  Vₖ' * Θ

    ZZᵀ  = Z * Z'   +     (dZ*dZ') .* (VₖᵀΘ' * VₖᵀΘ)
    Zy   = Z * y    +   sum( VₖᵀΘ' .* (dZ    * dy); dims=2   )
        
    return   (  ( ZZᵀ + λ*info.n*I  )   \  Zy )
end

# overparameterized, ridgeless:
function get_ŵ( Θ::AbstractArray, X::AbstractArray, y::AbstractArray, dy::AbstractArray, Vₖ::AbstractArray, info  )
    M    = Θ'*X
    dZ   = info.dσ.(M)
    VₖᵀΘ = Vₖ' * Θ
    Z    = hcat( info.σ.(M),  [dZ .* VₖᵀΘ[i,:] for i=1:info.k ] ... )

    return  Z * ( (Z'*Z) \  vcat( y, squish(dy) )  )
end


# ========================================================================
# Get Error
# ========================================================================


# get errors
function test_risk( ŵ, Θ, θ₀, Vₖ, info ; n_test=1000 )
    #truth
    X  = rand( Normal(), info.d, n_test )
    y  = sum( info.ϕ.(  X'*θ₀), dims=2 )   +   info.Δ*rand( Normal(), n_test )
    dy = observe_grad( X, θ₀, Vₖ, info )

    # predictions
    ŷ  = info.σ.( X' * Θ ) * ŵ
    dŷ = (Vₖ[1]' * Θ) * ( info.dσ.( Θ'*X ) .* ŵ )

    return mean( ( y - ŷ  ).^2 ), sum( ( dy' - dŷ ).^2 )/n_test
end

function train_risk( ŵ, Θ, θ₀, X, y, dy, Vₖ, info )
    # predictions
    ŷ  = info.σ.( X' * Θ ) * ŵ
    dŷ = (Vₖ[1]' * Θ) * ( info.dσ.( Θ'*X ) .* ŵ )

    return (1/2)*mean( ( y - ŷ  ).^2 ),  (1/2)*sum( ( dy'- dŷ ).^2 )/info.n
end


# ========================================================================
# Outer loops
# ========================================================================

function loop_instances( info; n_loops=10, seed=11  )
    ε   = zeros(4, n_loops)

    Threads.@threads for i = 1:n_loops
        Random.seed!( seed+i )
        X, Θ, θ₀, y, dy, Vₖ = instance( info )
        ŵ                   = get_ŵ( Θ, X, info.λ, y, dy, Vₖ[1], info )
        ε[1:2,i]           .= test_risk(   ŵ, Θ, θ₀, Vₖ, info )
        ε[3:4,i]           .= train_risk(  ŵ, Θ, θ₀, X, y, dy, Vₖ, info )   
    end
    return vcat( mean( ε ; dims=2),  std( ε ; dims=2) )  #std( ε ; dims=2)/sqrt(n_loops) )
end


function experiment( mode,
                     d_vec   ::AbstractArray, 
                     pn_ratio::AbstractArray, 
                     nd_ratio::AbstractArray, 
                     λ_vec   ::AbstractArray, 
                     Δ_vec   ::AbstractArray, 
                     ϖ_vec   ::AbstractArray,
                     name    ::String       ; 
                     σ       = erf, 
                     dσ      = d_erf, 
                     ϕ       = id, 
                     dϕ      = one, 
                     ℓ²      = 1,
                     H₁ᵏ     = 1,
                     n_loops = 100, 
                     seed    = 11, 
                     k       = 1,
                     k₀      = 1  )

    if mode != aligned
        ϖ_vec = [false]
    else
        k₀    = 1
    end

    for ( Δ, d, λ, ϖ , pn, nd ) in Iterators.product( Δ_vec, d_vec, λ_vec, ϖ_vec, pn_ratio, nd_ratio )

        info  = mode( r(nd*d), r(pn*nd*d), d, k, k₀, λ, Δ, σ, dσ, ϕ, dϕ, ϖ , ℓ², H₁ᵏ  )
        errs  = loop_instances( info;  n_loops, seed )
        addToFile( vcat( mode, info.n, info.p, d, pn, nd, pn*nd, ϖ , ℓ², H₁ᵏ, seed, n_loops, σ, ϕ, k, k₀,  λ, Δ, errs ), name )
    
    end
end


function start_experiment(name)
    makeFile( [ "setting", "n", "p", "d", "p/n", "n/d", "p/d", "enforce_varpi", "ell2_weight", "H1k_weight", 
                "seed", "n_processes", "sigma", "phi", "k", "k_0", "lambda", "Delta", 
                "eps_gen_l2","eps_gen_H1k","eps_tr_l2","eps_tr_H1k",
                "eps_gen_l2_std","eps_gen_H1k_std","eps_tr_l2_std","eps_tr_H1k_std"  ], name )
                #"eps_gen_l2_stderr","eps_gen_H1k_stderr","eps_tr_l2_stderr","eps_tr_H1k_stderr"  ], name )

end




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
    addToFile( DataFrame( reshape(df,1,:), :auto  ) , name )
end

# filter dataframe by column values
function multifilter(df, filCols, selectors; leftover=false)

    function filterRules(cols...)::Bool
        (cols) in selectors
    end

    return leftover ? (filter( filCols => filterRules , df ), filter( filCols => !filterRules , df )) : filter( filCols => filterRules , df )

end

