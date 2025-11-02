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
cosh⁻¹(x)     = 1. / cosh(x)

# derivatives of nonlinear functions
d_relu(x)      = x>0 ? 1. : 0.
d_swish(x;β=1) = ( 1 + exp(-β*x) + x*β*exp(-β*x) ) / ( 1 + exp(-β*x)  )^2
d_erf(x)       = ( 2 / sqrt(pi) )* exp( -x^2 )
d_tanh(x)      = 1 - (tanh(x))^2
d_atan(x)      = 1. / ( 1. + x^2)
d_cosh⁻¹(x)    = -sech(x)*tanh(x)

# numerical estimation of Hermite coefficients
function set_Hermite(σ ; s=31)
    z,w = gausshermite( s ; normalize=true )
    Κ₀  = w' * σ.(z)  
    Κ₁  = w' *( z .* σ.(z) )
    Κₛ  = sqrt( w'*( σ.(z).^2 ) - Κ₀^2 - Κ₁^2 )
    return Κ₀, Κ₁, Κₛ
end

#--------------------------------------------------------------
# quadrature
# -------------------------------------------------------------

function gauss_quad_1d( degree ; μ=0, σ²=1 )
    ω, weights  = gausshermite( degree ; normalize=true )
    return sqrt(σ²)*ω .+ μ, weights
end

function gauss_quad_2d( degree ; μ=[0. ; 0.], Σ=I(2) )
    L           = cholesky(Σ).L
    ω, weights  = gausshermite( degree ; normalize=true )
    x           = hcat( repeat(ω,       outer=degree), repeat(ω,       inner=degree)  )
    w           =       repeat(weights, inner=degree).*repeat(weights, outer=degree)
    return x, w
end

function get_q_points( k ; n_mc=3000 )
    k == 1 && return gauss_quad_1d(31)
    k == 2 && return gauss_quad_2d(21)
    Random.seed!(1413)
    return rand(Normal(), n_mc, k), ones(n_mc)/n_mc
end

function get_q_compact( a, b ; degree=100 )
    ω, weights  = gausslegendre( degree )
    return 0.5*(ω.+1)*(b-a).+a,  0.5*(b-a)*weights
end

#--------------------------------------------------------------
# densities
# -------------------------------------------------------------

function marchenko_pastur(λ ; c=1 ) 
    a = ( 1 - sqrt(c) )^2
    b = ( 1 + sqrt(c) )^2

    λ<a && return 0.
    λ>b && return 0.
    return (1/(2*π*c*λ)) * sqrt( (b-λ)*(λ-a))
end

function semicircular(λ)
    abs(λ) > 1 && return 0.
    return (2/π) * sqrt(1-λ^2)
end

#--------------------------------------------------------------
# Matrix Operations
# -------------------------------------------------------------

function det_3x3(A)
    return (A[:,1,1] .* (A[:,2,2] .* A[:,3,3] - A[:,2,3] .* A[:,3,2]) -
            A[:,1,2] .* (A[:,2,1] .* A[:,3,3] - A[:,2,3] .* A[:,3,1]) +
            A[:,1,3] .* (A[:,2,1] .* A[:,3,2] - A[:,2,2] .* A[:,3,1]))
end

function det_4x4(A)
    return  A[:,1,1].*det_3x3(A[:,[2,3,4],[2,3,4]]) -
            A[:,1,2].*det_3x3(A[:,[2,3,4],[1,3,4]]) +
            A[:,1,3].*det_3x3(A[:,[2,3,4],[1,2,4]]) -
            A[:,1,4].*det_3x3(A[:,[2,3,4],[1,2,3]])
end

function inv_3x3(A)
    a11, a12, a13 = A[:, 1, 1], A[:, 1, 2], A[:, 1, 3]
    a21, a22, a23 = A[:, 2, 1], A[:, 2, 2], A[:, 2, 3]
    a31, a32, a33 = A[:, 3, 1], A[:, 3, 2], A[:, 3, 3]

    det = (a11 .* (a22 .* a33 - a23 .* a32) -
           a12 .* (a21 .* a33 - a23 .* a31) +
           a13 .* (a21 .* a32 - a22 .* a31))

    # Calculate cofactor matrix elements
    c11 =   a22 .* a33 - a23 .* a32
    c12 = -(a21 .* a33 - a23 .* a31)
    c13 =   a21 .* a32 - a22 .* a31

    c21 = -(a12 .* a33 - a13 .* a32)
    c22 =   a11 .* a33 - a13 .* a31
    c23 = -(a11 .* a32 - a12 .* a31)

    c31 =   a12 .* a23 - a13 .* a22
    c32 = -(a11 .* a23 - a13 .* a21)
    c33 =   a11 .* a22 - a12 .* a21

    # adjugate matrix
    adj = Complex.( zeros(size(A,1), 3, 3) )
    adj[:, 1, 1] = c11
    adj[:, 1, 2] = c21
    adj[:, 1, 3] = c31
    adj[:, 2, 1] = c12
    adj[:, 2, 2] = c22
    adj[:, 2, 3] = c32
    adj[:, 3, 1] = c13
    adj[:, 3, 2] = c23
    adj[:, 3, 3] = c33

    A_inv   = adj ./ det

    return adj ./ det_3x3(A)
end

function inv_4x4(A)

    # adjugate matrix
    adj = Complex.( zeros(size(A,1), 4, 4) )
    adj[:, 1, 1] =  det_3x3( A[:,[2,3,4],[2,3,4]]  )
    adj[:, 1, 2] = -det_3x3( A[:,[1,3,4],[2,3,4]]  )
    adj[:, 1, 3] =  det_3x3( A[:,[1,2,4],[2,3,4]]  )
    adj[:, 1, 4] = -det_3x3( A[:,[1,2,3],[2,3,4]]  )

    adj[:, 2, 1] = -det_3x3( A[:,[2,3,4],[1,3,4]]  )
    adj[:, 2, 2] =  det_3x3( A[:,[1,3,4],[1,3,4]]  )
    adj[:, 2, 3] = -det_3x3( A[:,[1,2,4],[1,3,4]]  )
    adj[:, 2, 4] =  det_3x3( A[:,[1,2,3],[1,3,4]]  )

    adj[:, 3, 1] =  det_3x3( A[:,[2,3,4],[1,2,4]]  )
    adj[:, 3, 2] = -det_3x3( A[:,[1,3,4],[1,2,4]]  )
    adj[:, 3, 3] =  det_3x3( A[:,[1,2,4],[1,2,4]]  )
    adj[:, 3, 4] = -det_3x3( A[:,[1,2,3],[1,2,4]]  )

    adj[:, 4, 1] = -det_3x3( A[:,[2,3,4],[1,2,3]]  )
    adj[:, 4, 2] =  det_3x3( A[:,[1,3,4],[1,2,3]]  )
    adj[:, 4, 3] = -det_3x3( A[:,[1,2,4],[1,2,3]]  )
    adj[:, 4, 4] =  det_3x3( A[:,[1,2,3],[1,2,3]]  )

    return adj ./ det_4x4(A)
end

#--------------------------------------------------------------
#
# Generic Cauchy transform
#
# -------------------------------------------------------------

function operator_Cauchy_transform( k, Z, A, μ, λs, weights ; atoms=nothing  )
    
    result  = sum( [ w*μ(λ)*inv(Z-A*λ)  for (λ,w) in zip(λs,weights)  ])
    isnothing(atoms) && return result

    result += sum( [ a2*inv(Z-A*a1)  for (a1,a2) in zip(atoms[1],atoms[2])  ])
    return result

end

twist(A,B ; dim) = reshape( sum( permutedims( A, [1,3,2] ).*B ; dims=2 ), dim, dim )
function operator_Cauchy_transform_Nd( k, Z, coef_mtx, λs, weights ; atoms=nothing  )
    dim    = size(Z,1)
    result = Folds.sum( [  w*inv( Z - twist( coef_mtx, λs[ℓ,:]' ; dim)) for (ℓ,w) in zip(1:size(λs,1),weights) ] )
    return result
end

#-------------------------------------------------------------------
#
# Specialized Cauchy Transform
#
#-------------------------------------------------------------------

function type1_Cauchy_transform( k, Z, λ::AbstractArray, w ; atoms=nothing  )
    # problem dimensions
    dim      = size(Z,1)
    nw       = size(w,1)

    # extract block components
    A        = Z[2,1]
    B        = Z[2,2:end]
    C        = Z[[1;3:dim],1] # permute rows 1 and 2
    Dinv     = inv( Z[[1;3:dim], 2:dim]  )

    # schur complement
    λC            = zeros(dim-1,nw)
    λC[6:2:end,:] = λ'
    λB            = λC[[3:dim-1;1:2],:]
    schur         = 1 ./ ( A .-  sum( (B .- λB).*(Dinv*(C .- λC));dims=1) )

    # off diagonals + quadrature
    schur_λB  = sum( λB .* schur .* w' ; dims=2 )
    schur_λC  = sum( λC .* schur .* w' ; dims=2 )
    schur_λCB = (λC .* schur) * (λB' .* w )
    schur     = schur*w

    # construct inverse
    results = [ schur                             -reshape( schur.*B - schur_λB, 1, : )*Dinv ;
               -Dinv*( C.*schur - schur_λC  )      Dinv + Dinv*( (C.*schur - schur_λC)*reshape(B,1,:) - C*reshape(schur_λB,1,:) + schur_λCB  )*Dinv ]
    
    # permute columns 1 and 2
    return results[:,[2;1;3:dim]]
end


function type2_Cauchy_transform( k, Z, λs::AbstractArray, weights  )

    nw   = size(weights,1)
    dim  = size(Z,1)
    Zinv = inv(Z)

    tmp1 = Complex.( cat( repeat(         reshape( Zinv[:,2],    1, :),   outer=(nw,1)   ),
                          repeat(         reshape( Zinv[:,4],    1, :),   outer=(nw,1)   ),
                          sum(  [ λs[:,m]*reshape( Zinv[:,6+2m], 1, :) for m in 1:k ] ), dims=3 ))

    wood         = Complex.(zeros(nw,3,3))
    wood[:,1,:]  = reshape( sum( tmp1[:,(3 .+ 2(1:k)),:] .* λs ; dims=2 ), nw, 3 ) + tmp1[:,end,:].*λs[:,1]
    wood[:,2,:]  = tmp1[:,end-2,:] .* λs[:,1]
    wood[:,3,:]  = tmp1[:,1,:]
    wood[:,1,1].-= 1.
    wood[:,2,2].-= 1.
    wood[:,3,3].-= 1.

    wood_inv = inv_3x3( wood )

    tmp2                  = Complex.(zeros(nw,dim,3))
    tmp2[:,2,:]           = wood_inv[:,1,:]
    tmp2[:,4,:]           = wood_inv[:,2,:]
    tmp2[:,6 .+ 2(1:k),:] = wood_inv[:,3:3,:] .* λs

    result                  = Complex.(zeros(nw,dim,dim))
    result[:,:,3 .+ 2(1:k)] = tmp2[:,:,1] .* reshape( λs,:,1,k )
    result[:,:,end]         = tmp2[:,:,1] .* λs[:,1]
    result[:,:,end-2]       = tmp2[:,:,2] .* λs[:,1]
    result[:,:,1]           = tmp2[:,:,3]
    result                  = reshape( sum( result .* weights ; dims=1 ), size(result,2), : )
    result                  = Zinv - Zinv * result * Zinv

    return result
end

function type3_Cauchy_transform( k, Z, λ::AbstractArray, w )

    nw  = size(w,1)
    dim = size(Z,1)
    Z⁻¹ = inv(Z)

    𝘐 = 2:2:2k
    ϰ = 2*(k+1)
    U = [ (ϰ+7).+𝘐,  (5).+𝘐,    2,        (ϰ+4)    ]
    V = [ 1,         (ϰ+2),    (ϰ+5).+𝘐,  (3).+𝘐   ]

    M = cat([     λ*Z⁻¹[ V[1], U[1] ]          λ*Z⁻¹[ V[2], U[1] ]     sum(λ*Z⁻¹[ V[3], U[1] ].*λ ; dims=2)   sum(λ*Z⁻¹[ V[4], U[1] ].*λ ; dims=2) ],
            [     λ*Z⁻¹[ V[1], U[2] ]          λ*Z⁻¹[ V[2], U[2] ]     sum(λ*Z⁻¹[ V[3], U[2] ].*λ ; dims=2)   sum(λ*Z⁻¹[ V[4], U[2] ].*λ ; dims=2) ],
            [ fill( Z⁻¹[ V[1], U[3] ],nw)  fill( Z⁻¹[ V[2], U[3] ],nw)     λ*Z⁻¹[ V[3], U[3] ]                    λ*Z⁻¹[ V[4], U[3] ]              ],
            [ fill( Z⁻¹[ V[1], U[4] ],nw)  fill( Z⁻¹[ V[2], U[4] ],nw)     λ*Z⁻¹[ V[3], U[4] ]                    λ*Z⁻¹[ V[4], U[4] ]              ] ; dims=3 )

    M[:,1,1].-= 1.
    M[:,2,2].-= 1.
    M[:,3,3].-= 1.
    M[:,4,4].-= 1.

    M = inv_4x4(M)

    result = Complex.(zeros(dim,dim))
    result[  U[1],       V[3]  ]       +=  λ'*(w.*λ.*M[:,1,3])
    result[  U[1],       V[4]  ]       +=  λ'*(w.*λ.*M[:,1,4])
    result[  U[2],       V[3]  ]       +=  λ'*(w.*λ.*M[:,2,3])
    result[  U[2],       V[4]  ]       +=  λ'*(w.*λ.*M[:,2,4])
    result[  U[1],      [V[1];V[2]]  ] +=  sum(w.*λ.*M[:,1:1,1:2]; dims=1)[1,:,:]
    result[  U[2],      [V[1];V[2]]  ] +=  sum(w.*λ.*M[:,2:2,1:2]; dims=1)[1,:,:]
    result[  U[3],       V[3] ]        +=  sum(w.*λ.*M[:,3:3,3];   dims=1)[1,:,:]
    result[  U[4],       V[3] ]        +=  sum(w.*λ.*M[:,4:4,3];   dims=1)[1,:,:]
    result[  U[3],       V[4] ]        +=  sum(w.*λ.*M[:,3:3,4];   dims=1)[1,:,:]
    result[  U[4],       V[4] ]        +=  sum(w.*λ.*M[:,4:4,4];   dims=1)[1,:,:]
    result[ [U[3];U[4]], [V[1];V[2]] ] +=  sum(w.*M[:,3:4,1:2];    dims=1)[1,:,:]

    return Z⁻¹ - Z⁻¹*result*Z⁻¹
end


#-------------------------------------------------------------------------------------
#
# Subordinator Calculation
#
#-------------------------------------------------------------------------------------
# original tol: 1e-8
function get_g_sum( Ĝx, Ĝy ; dim, max_iter=1500, tol=1e-8, α=0.2, init_ω=nothing )
    
    Random.seed!(1)
    ω  = isnothing(init_ω) ? im*rand(Normal(), dim, dim) : copy(init_ω)
    ω1 = 1

    for _ in tqdm(1:max_iter)
        

        Ĝxω = Ĝx(ω)
        hxω = try inv(Ĝxω) - ω
              catch err
                error("Matrix inversion failed in h_x computation.")
              end
    
        Ĝya = Ĝy(hxω) 
        hyω = try inv(Ĝya) - hxω
              catch err
                error("Matrix inversion failed in h_y computation.")
              end

        ω1 = α * hyω + (1-α)*ω
        norm(ω1-ω)<tol  &&  return Ĝx(ω1)[1,1]
        ω  = ω1
    end
    println("Warning: ω(Z) iteration did not converge.")
    return Ĝx(ω1)[1,1]
end

#--------------------------------------------------------------
#
# Rational Functions
#
# -------------------------------------------------------------

CI(x) = CartesianIndex(x)

function τ_rational_type1( k, c, z₀, z₁, z₂, z₃, b₀, b₁ )

    # quadrature and y density
    λx, wx  = get_q_points(k) 
    λy, wy  = get_q_compact( (1-sqrt(c))^2, (1+sqrt(c))^2 )
    μy(λ)   = marchenko_pastur( λ ; c )   
    atoms_y = c > 1 ? ([0.], [1 - 1/c]) : nothing

    # Ay
    dim           = 2k+5
    Ay            = Complex.(zeros(dim,dim))
    inds          = [ (4+2m, 2+2m)  for m in 1:k ]
    Ay[CI.(inds)].= z₃
    Ay[2,end]     = b₁
    Ay[4,2]       = z₁
    
    # By
    By              = Complex.(zeros(dim,dim))
    By[1,end-1]     = 1
    By[2,3]         = 1
    By[3,end-1]     = 1
    By[3,end]       = 1
    By[5,1]         = 1
    By[2,end]       = b₀
    By[4,2]         = z₀
    inds            = vcat( [(4,3), (5,2)], [ [(4+2m, 3+2m), (5+2m, 2+2m)] for m in 1:k]...)
    By[CI.(inds)]  .= -1
    inds            = [ (4+2m, 2+2m)  for m in 1:k] # TODO check indices; funny -1 in python
    By[CI.(inds)]  .= z₂ 

    # Cauchy transform functions
    Ĝx(Z) = type1_Cauchy_transform( k, Z, λx, wx )
    Ĝy(Z) = operator_Cauchy_transform( k, Z-By, Ay, μy, λy, wy ; atoms=atoms_y  )

    return -real( get_g_sum( Ĝx, Ĝy ; dim  ) ) 
end


#--------


function τ_rational_type2( k, c, z₀, z₁, z₂, z₃, b₀, b₁ )

    # quadrature and y density
    dim     = 2k+6
    λx, wx  = get_q_points(k) 
    λy, wy  = get_q_compact( (1-sqrt(c))^2, (1+sqrt(c))^2 )
    μy(λ)   = marchenko_pastur( λ ; c )
    atoms_y = c > 1 ? ([0.], [1 - 1/c]) : nothing

    Ay            = Complex.(zeros(dim,dim))
    Ay[3,end-1]   = -b₁
    Ay[5,2]       = z₁
    inds          = [ (5+2m, 2+2m)  for m in 1:k]
    Ay[CI.(inds)].= z₃

    By            = Complex.(zeros(dim,dim))
    inds          = vcat( [(6,2),(5,3)], [ [(5+2m,3+2m), (6+2m,2+2m)]  for m in 1:k]... )
    By[CI.(inds)].= -1
    inds          = [ (5+2m,2+2m)   for m in 1:k]
    By[CI.(inds)].= z₂
    By[5,2]       = z₀
    By[3,end-1]   = -b₀
    By[1,end-2]   = 1
    By[2,3]       = 1
    By[3,end]     = 1
    By[4,end-1]   = 1
    By[6,1]       = 1

    Ĝx(Z) = type2_Cauchy_transform( k, Z, λx, wx )
    Ĝy(Z) = operator_Cauchy_transform( k, Z-By, Ay, μy, λy, wy ; atoms=atoms_y  )

    return -real( get_g_sum( Ĝx,Ĝy ; dim ) )
end

function τ_rational_type_3( k, c, z₀, z₁, z₂, z₃, b₀, b₁, b₂, b₃ )
    
    # block matrix size and locations
    dim    = 4(k+1)+5                                                                                      # number of blocks
    rows   = [ 1, (1).+(1:2),      (3).+(1:2(k+1)),   (3+2(k+1)).+(1:2),      (5+2(k+1)).+(1:2(k+1))  ]    # rows of blocks from multiplying linearizations
    cols   = [ 1, (1).+(1:2(k+1)), (2(k+1)+1).+(1:2), (3+2(k+1)).+(1:2(k+1)), (3+4(k+1)).+(1:2)       ]    # cols of blocks from multiplying linearizations

    # quadrature and densities
    λx, wx  = get_q_points(k) 
    λy, wy  = get_q_compact( (1-sqrt(c))^2, (1+sqrt(c))^2 )
    μy(λ)   = marchenko_pastur( λ ; c )   
    atoms_y = c > 1 ? ([0.], [1 - 1/c]) : nothing

    # Ay: coefficients of m=ΘᵀΘ
    Ay                        = Complex.(spzeros(dim,dim))
    Ay[rows[5],cols[2]]       = spdiagm( vcat([z₁,0], repeat( [z₃,0], k ))  ) 
    Ay[rows[3],cols[4]]       = spdiagm( vcat([z₁,0], repeat( [z₃,0], k ))  )
    Ay[rows[4][1],cols[3][2]] = b₃
    Ay[rows[2][1],cols[5][2]] = b₁

    # By: constant coeffients
    By                        = Complex.(spzeros(dim,dim))
    By[rows[5],    cols[2]]    = spdiagm( 0=>vcat([z₀,0], repeat( [z₂,0], k )), -1=>repeat([0,-1],k+1)[2:end], 1=>repeat([0,-1],k+1)[2:end]  )
    By[rows[3],    cols[4]]    = spdiagm( 0=>vcat([z₀,0], repeat( [z₂,0], k )), -1=>repeat([0,-1],k+1)[2:end], 1=>repeat([0,-1],k+1)[2:end]  )
    By[rows[4],    cols[3]]    = [0. b₂; 1. 1.]
    By[rows[2],    cols[5]]    = [0. b₀; 1. 1.]
    By[rows[1],    cols[5]]    = [1.,0.]
    By[rows[2][1], cols[4][2]] = 1.
    By[rows[3][2], cols[3][1]] = 1.
    By[rows[4][1], cols[2][2]] = 1.
    By[rows[5][2], cols[1]]    = 1.

    # Cauchy transform functions
    Ĝx(Z) =  type3_Cauchy_transform( k, Z, reshape(λx,:,k), wx )
    Ĝy(Z) =  operator_Cauchy_transform(    k, Z-By, Ay,    μy, λy, wy ; atoms=atoms_y  )

    return -real( get_g_sum( Ĝx, Ĝy ; dim, tol=1e-6  ) ) 
end


function τ_rational_type_4( k, c, z₀, z₁, z₂, z₃, b₀, b₁, b₂, b₃ )
    
    # block matrix size and locations
    dim    = 4(k+1)+6                                                                                      # number of blocks
    rows   = [ 1, (1).+(1:2),      (3).+(1:2(k+1)),   (3+2(k+1)).+(1:3),      (6+2(k+1)).+(1:2(k+1))  ]    # rows of blocks from multiplying linearizations
    cols   = [ 1, (1).+(1:2(k+1)), (2(k+1)+1).+(1:3), (4+2(k+1)).+(1:2(k+1)), (4+4(k+1)).+(1:2)       ]    # cols of blocks from multiplying linearizations


    # coefficients of Gaussian diagonals
    coefs                                       = Complex.(zeros(dim,dim,k))
    coefs[rows[5][4:2:end], cols[1],:]          = I(k)
    coefs[rows[4][1],       cols[2][4:2:end],:] = I(k)
    coefs[rows[3][4:2:end], cols[3][1],:]       = I(k)
    coefs[rows[2][1],       cols[4][4:2:end],:] = I(k)
    coefs[rows[4],          cols[3],         1] = [ 0 0 1.; 0 0 0; 1. 0 0 ] 

    # quadrature and densities
    λx, wx  = get_q_points(k) 
    λy, wy  = get_q_compact( (1-sqrt(c))^2, (1+sqrt(c))^2 )
    μy(λ)   = marchenko_pastur( λ ; c )   
    atoms_y = c > 1 ? ([0.], [1 - 1/c]) : nothing

    # Ay: coefficients of m=ΘᵀΘ
    Ay                        = Complex.(spzeros(dim,dim))
    Ay[rows[5],cols[2]]       = spdiagm( vcat([z₁,0], repeat( [z₃,0], k ))  ) 
    Ay[rows[3],cols[4]]       = spdiagm( vcat([z₁,0], repeat( [z₃,0], k ))  )
    Ay[rows[4][2],cols[3][2]] = -b₃
    Ay[rows[2][1],cols[5][2]] =  b₁

    # By: constant coeffients
    By                        = Complex.(spzeros(dim,dim))
    By[rows[5],    cols[2]]    = spdiagm( 0=>vcat([z₀,0], repeat( [z₂,0], k )), -1=>repeat([0,-1],k+1)[2:end], 1=>repeat([0,-1],k+1)[2:end]  )
    By[rows[3],    cols[4]]    = spdiagm( 0=>vcat([z₀,0], repeat( [z₂,0], k )), -1=>repeat([0,-1],k+1)[2:end], 1=>repeat([0,-1],k+1)[2:end]  )
    By[rows[4],    cols[3]]    = [ 0 0 0; 0 -b₂ 1.; 0 1. 0 ]
    By[rows[2],    cols[5]]    = [0. b₀; 1. 1.]
    By[rows[1],    cols[5]]    = [1.,0.]
    By[rows[2][1], cols[4][2]] = 1.
    By[rows[3][2], cols[3][1]] = 1.
    By[rows[4][1], cols[2][2]] = 1.
    By[rows[5][2], cols[1]]    = 1.

    # Cauchy transform functions
    Ĝx(Z) =  operator_Cauchy_transform_Nd( k, Z,    coefs, λx, wx )
    Ĝy(Z) =  operator_Cauchy_transform(    k, Z-By, Ay,    μy, λy, wy ; atoms=atoms_y  )

    return -real( get_g_sum( Ĝx, Ĝy ; dim, tol=1e-6  ) ) 
end

function τ_rational_type_5( k, c, z₀, z₁, z₂, z₃, b₀, b₁, b₂, b₃ ; same_gauss=true )
    
    # block matrix size and locations
    dim    = 4(k+1)+7                                                                                      # number of blocks
    rows   = [ 1, (1).+(1:3),      (4).+(1:2(k+1)),   (4+2(k+1)).+(1:3),      (7+2(k+1)).+(1:2(k+1))  ]    # rows of blocks from multiplying linearizations
    cols   = [ 1, (1).+(1:2(k+1)), (2(k+1)+1).+(1:3), (4+2(k+1)).+(1:2(k+1)), (4+4(k+1)).+(1:3)       ]    # cols of blocks from multiplying linearizations
    j      = same_gauss ? 1 : 2


    # coefficients of Gaussian diagonals
    coefs                                       = Complex.(zeros(dim,dim,k))
    coefs[rows[5][4:2:end], cols[1],:]          = I(k)
    coefs[rows[4][1],       cols[2][4:2:end],:] = I(k)
    coefs[rows[3][4:2:end], cols[3][1],:]       = I(k)
    coefs[rows[2][1],       cols[4][4:2:end],:] = I(k)
    coefs[rows[4],          cols[3],         1] = [ 0 0 1.; 0 0 0; 1. 0 0 ] 
    coefs[rows[2],          cols[5],         j] = [ 0 0 1.; 0 0 0; 1. 0 0 ]

    # quadrature and densities
    λx, wx  = get_q_points(k) 
    λy, wy  = get_q_compact( (1-sqrt(c))^2, (1+sqrt(c))^2 )
    μy(λ)   = marchenko_pastur( λ ; c )   
    atoms_y = c > 1 ? ([0.], [1 - 1/c]) : nothing

    # Ay: coefficients of m=ΘᵀΘ
    Ay                        = Complex.(spzeros(dim,dim))
    Ay[rows[5],cols[2]]       = spdiagm( vcat([z₁,0], repeat( [z₃,0], k ))  ) 
    Ay[rows[3],cols[4]]       = spdiagm( vcat([z₁,0], repeat( [z₃,0], k ))  )
    Ay[rows[4][2],cols[3][2]] = -b₃
    Ay[rows[2][2],cols[5][2]] = -b₁

    # By: constant coeffients
    By                        = Complex.(spzeros(dim,dim))
    By[rows[5],    cols[2]]    = spdiagm( 0=>vcat([z₀,0], repeat( [z₂,0], k )), -1=>repeat([0,-1],k+1)[2:end], 1=>repeat([0,-1],k+1)[2:end]  )
    By[rows[3],    cols[4]]    = spdiagm( 0=>vcat([z₀,0], repeat( [z₂,0], k )), -1=>repeat([0,-1],k+1)[2:end], 1=>repeat([0,-1],k+1)[2:end]  )
    By[rows[4],    cols[3]]    = [ 0 0 0; 0 -b₂ 1.; 0 1. 0 ]
    By[rows[2],    cols[5]]    = [ 0 0 0; 0 -b₀ 1.; 0 1. 0 ]
    By[rows[1],    cols[5]]    = [1.,0.,0.]
    By[rows[2][1], cols[4][2]] = 1.
    By[rows[3][2], cols[3][1]] = 1.
    By[rows[4][1], cols[2][2]] = 1.
    By[rows[5][2], cols[1]]    = 1.

    # Cauchy transform functions
    Ĝx(Z) =  operator_Cauchy_transform_Nd( k, Z,    coefs, λx, wx )
    Ĝy(Z) =  operator_Cauchy_transform(    k, Z-By, Ay,    μy, λy, wy ; atoms=atoms_y  )

    return -real( get_g_sum( Ĝx, Ĝy ; dim, tol=1e-6 ) ) 
end


#--------------------------------------------------------------
# Solve for V
#--------------------------------------------------------------

V2V̂( V ; settings ) = settings.α ./( 1 .+ V) 

function V̂2V( V̂ ; settings )
    κ₁, κₛ, dκ₁, dκₛ = settings.κ₁, settings.κₛ, settings.dκ₁, settings.dκₛ
    
    z₀ = settings.λ + V̂[1]*κₛ^2
    z₁ = V̂[1]*κ₁^2
    z₂ = V̂[2]*dκₛ^2
    z₃ = V̂[2]*dκ₁^2
    
    Va = τ_rational_type1( settings.k, 1/settings.γ, z₀, z₁, z₂, z₃,  κₛ^2,  κ₁^2  )
    Vc = τ_rational_type2( settings.k, 1/settings.γ, z₀, z₁, z₂, z₃, dκₛ^2, dκ₁^2  )
    
    return [Va; Vc]
end

log_transform(V)  =  log(1. + V)
exp_transform(Ṽ)  =  exp(Ṽ) - 1.

function V_system( V, settings )
    V̂   = V2V̂( exp_transform.(V) ; settings )
    ret = log_transform.(V̂2V( V̂ ; settings )) - V

    return ret
end


function solver( eq_sys ; settings, tol=1e-8, max_steps=100000, η=0.7 )
    state  = [1.,1.]
    res    = 1
    steps  = 0
    while res > tol && steps < max_steps
        update = eq_sys( state, settings )
        res    = maximum(abs.(update))
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
    
    κ₁, κₛ, dκ₁, dκₛ  = settings.κ₁, settings.κₛ, settings.dκ₁, settings.dκₛ
    
    z₀ = settings.λ + V̂[1]*κₛ^2
    z₁ = V̂[1]*κ₁^2
    z₂ = V̂[2]*dκₛ^2
    z₃ = V̂[2]*dκ₁^2
    
    fa = τ_rational_type1( settings.k, 1/settings.γ, z₀, z₁, z₂, z₃, 0.,  κ₁^2  )
    f̂[2]==0 && return (1/settings.γ)*[fa*f̂[1]; 0.]

    fb = τ_rational_type2( settings.k, 1/settings.γ, z₀, z₁, z₂, z₃, 0., dκ₁^2  )
    return (1/settings.γ)*[fa*f̂[1]; fb*f̂[2]]
end


#--------------------------------------------------------------
# Solve for q
#--------------------------------------------------------------


function V̂f2q( V̂, f̂, f ; settings )
    κ₀, κ₁, κₛ, dκ₀, dκ₁, dκₛ = settings.κ₀, settings.κ₁, settings.κₛ, settings.dκ₀, settings.dκ₁, settings.dκₛ
    γ, k, λ, α, Cₙ11, Cₙ22    = settings.γ, settings.k, settings.λ, settings.α, settings.Cₙ11, settings.Cₙ22

    z₀ = λ + V̂[1]*κₛ^2
    z₁ = V̂[1]*κ₁^2
    z₂ = V̂[2]*dκₛ^2
    z₃ = V̂[2]*dκ₁^2

    τ_A⁻¹M₀₀A⁻¹M₀₀         =                   τ_rational_type_3( k, 1/γ, z₀, z₁, z₂, z₃,  κₛ^2,   κ₁^2,  κₛ^2,   κ₁^2 )
    τ_A⁻¹D₁M₁₁D₁A⁻¹M₀₀     =                   τ_rational_type_4( k, 1/γ, z₀, z₁, z₂, z₃,  κₛ^2,   κ₁^2, dκₛ^2,  dκ₁^2 )
    τ_A⁻¹D₁M₁₁D₁A⁻¹D₁M₁₁D₁ =                   τ_rational_type_5( k, 1/γ, z₀, z₁, z₂, z₃, dκₛ^2,  dκ₁^2, dκₛ^2,  dκ₁^2 )
    τ_A⁻¹ΘᵀΘA⁻¹M₀₀         =                   τ_rational_type_3( k, 1/γ, z₀, z₁, z₂, z₃,  κₛ^2,   κ₁^2,     0,      1 )
    τ_A⁻¹ΘᵀΘA⁻¹D₁M₁₁D₁     =                   τ_rational_type_4( k, 1/γ, z₀, z₁, z₂, z₃,     0,      1, dκₛ^2,  dκ₁^2 )
    τ_A⁻¹D₁ΘᵀΘD₁A⁻¹M₀₀     = f̂[2]!=0         ? τ_rational_type_4( k, 1/γ, z₀, z₁, z₂, z₃,  κₛ^2,   κ₁^2,     0,      1 )                    : 0.
    τ_A⁻¹D₁ΘᵀΘD₁A⁻¹D₁M₁₁D₁ = f̂[2]!=0         ? τ_rational_type_5( k, 1/γ, z₀, z₁, z₂, z₃,     0,      1, dκₛ^2,  dκ₁^2 )                    : 0.
    τ_A⁻¹D₁M₁₁D₁A⁻¹D₂M₁₁D₂ =  k>1            ? τ_rational_type_5( k, 1/γ, z₀, z₁, z₂, z₃, dκₛ^2,  dκ₁^2, dκₛ^2,  dκ₁^2 ; same_gauss=false ) : τ_A⁻¹D₁M₁₁D₁A⁻¹D₁M₁₁D₁
    τ_A⁻¹D₁ΘᵀΘD₁A⁻¹D₂M₁₁D₂ =  k>1 && f̂[2]!=0 ? τ_rational_type_5( k, 1/γ, z₀, z₁, z₂, z₃,     0,      1, dκₛ^2,  dκ₁^2 ; same_gauss=false ) : τ_A⁻¹D₁ΘᵀΘD₁A⁻¹D₁M₁₁D₁

    # Mass matrix
    Aq = vcat( [  1           0         -τ_A⁻¹M₀₀A⁻¹M₀₀          -τ_A⁻¹D₁M₁₁D₁A⁻¹M₀₀                                          ],
               [  0           1         -k*τ_A⁻¹D₁M₁₁D₁A⁻¹M₀₀    -τ_A⁻¹D₁M₁₁D₁A⁻¹D₁M₁₁D₁ - ((k^2-k)/k)*τ_A⁻¹D₁M₁₁D₁A⁻¹D₂M₁₁D₂   ],
               [ -V̂[1]^2/α    0          1                          0                                                         ],
               [  0          -V̂[2]^2/α   0                          1                                                         ] )


    # RHS
    q0 = [ Aq \ [   ( κ₁^2/γ) * f̂[1]^2 * τ_A⁻¹ΘᵀΘA⁻¹M₀₀,
                  k*( κ₁^2/γ) * f̂[1]^2 * τ_A⁻¹ΘᵀΘA⁻¹D₁M₁₁D₁,
                    V̂[1]^2  * (η1 + settings.var_ϕ + (abs(κ₀)<1e-8)*settings.Eϕ^2   )/α - 2*V̂[1]*f[1]*f̂[1]/α,
                  k*V̂[2]^2  *  η2 / α  ]  for (η1,η2) in zip(Cₙ11, Cₙ22)  ]

    qv =   Aq \ [  (dκ₁^2/γ) * f̂[2]^2 * τ_A⁻¹D₁ΘᵀΘD₁A⁻¹M₀₀,
                   (dκ₁^2/γ) * f̂[2]^2 * ( τ_A⁻¹D₁ΘᵀΘD₁A⁻¹D₁M₁₁D₁ + ((k^2-k)/k)*τ_A⁻¹D₁ΘᵀΘD₁A⁻¹D₂M₁₁D₂),
                    0.,
                    V̂[2]^2 *(settings.var_dϕ + (abs(dκ₀)<1e-8)*settings.Edϕ^2)/α - 2*V̂[2]*f[2]*f̂[2]/α ]

    return [ qv[1:2], [getindex.(q0,1) getindex.(q0,2)] , qv[3:4], [getindex.(q0,3) getindex.(q0,4)] ]
end


#--------------------------------------------------------------
# Outer loop
#--------------------------------------------------------------

function predict_error( settings )

    # fixed point system
    #V  = solver( V_system ; settings  )
    V  = opt.root( V_system, [0.01,0.01], args=(settings),method="excitingmixing",tol=1e-7)["x"]
    V̂  = V2V̂( exp_transform.(V) ; settings )

    # covariance between truth and y predictions
    f̂  = V̂ .* [ settings.Edϕ; settings.Ed2ϕ ]
    f  = f̂2f( V̂, f̂ ; settings )

    # covariance between predictions
    qv, q0, q̂v, q̂0 = V̂f2q( V̂, f̂, f ; settings )
    
    # training error
    ε_tr_ℓ₂_0  = (0.5/settings.α) * q̂0[:,1] 
    ε_tr_ℓ₂_2  = (0.5/settings.α) * q̂v[1]
    ε_tr_H₁ᵏ_0 = (0.5/settings.α) * q̂0[:,2]# * k
    ε_tr_H₁ᵏ_2 = (0.5/settings.α) * q̂v[2]  # * k

    # testing error
    ε_gn_ℓ₂_0  = settings.Cₙ11 + q0[:,1] .+ (settings.var_ϕ  + (abs(settings.κ₀)<1e-8)*settings.Eϕ^2 - 2*settings.Edϕ*f[1])  
    ε_gn_ℓ₂_2  = qv[1]
    ε_gn_H₁ᵏ_0 = k * settings.Cₙ22 + q0[:,2]
    ε_gn_H₁ᵏ_2 = settings.var_dϕ  + (abs(settings.dκ₀)<1e-8)*settings.Edϕ^2 - 2*f[2]*settings.Ed2ϕ + qv[2]


    return [ε_tr_ℓ₂_0  ε_tr_H₁ᵏ_0  ε_gn_ℓ₂_0  ε_gn_H₁ᵏ_0   q0  q̂0],
           [ε_tr_ℓ₂_2  ε_tr_H₁ᵏ_2  ε_gn_ℓ₂_2  ε_gn_H₁ᵏ_2   qv' q̂v'  f'  f̂'  exp_transform.(V')  V̂']
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
                     "eps_tr_l2_0", "eps_tr_H1k_0", "eps_gn_l2_0", "eps_gn_H1k_0", "q0_a", "q0_c", "q0hat_a", "q0hat_c",
                     "eps_tr_l2_2", "eps_tr_H1k_2", "eps_gn_l2_2", "eps_gn_H1k_2", "qv_a", "qv_c", "qvhat_a", "qvhat_c",   
                     "f_a",         "f_b",          "fhat_a",      "fhat_b",       "V_a",  "V_c",  "Vhat_a",   "Vhat_c" ), name )
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
                     k          = 1,
                     d          = 500  )

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
