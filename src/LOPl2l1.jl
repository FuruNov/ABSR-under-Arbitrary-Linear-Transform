using LinearAlgebra
using PartialFunctions
using Parameters
using ProgressMeter
using SparseArrays

include("ProjOp.jl")
include("ProxOp.jl")
include("utils.jl")
include("norm.jl")
include("matDiff.jl")
include("matFilter.jl")

export LOPl2l1Reg

@with_kw struct LOPl2l1Reg{T<:Real}
    λ::T # regularization coefficient
    α::T # non-zero block size
    stopCri::T
    maxIter::UInt64
    @assert all((isless$0).([λ, α, stopCri, maxIter]))
end

function (param::LOPl2l1Reg)(
    y::V, L::Union{Matrix{T},S}, R::S, D::SparseMatrixCSC{Int8,Int64};
)::Tuple{V,V} where {T<:Real,V<:Vector{T},S<:SparseMatrixCSC{T,Int64}}
    @unpack_LOPl2l1Reg param
    γ₁::T, γ₂::T = T(1e0), T(1e0)
    μ₁::T = if isdiag(L)
        _op_norm = maximum(diag(L))
        1 / sqrt(2 * _op_norm^2 + 1) / sqrt(γ₁ * γ₂)
    else
        1 / sqrt(2 * op_norm(L)^2 + 1) / sqrt(γ₁ * γ₂)
    end
    μ₂::T = if isdiag(R)
        _op_norm = maximum(diag(R))
        1 / sqrt(2 * _op_norm^2 + 1) / sqrt(γ₁ * γ₂)
    else
        1 / sqrt(2 * op_norm(R)^2 + 1) / sqrt(γ₁ * γ₂)
    end
    μ₃::T = 1 / sqrt(op_norm(D)^2 + 1) / sqrt(γ₁ * γ₂)

    x::V = if L == I
        copy(y)
    else
        randn(T, size(L, 2))
    end
    σ::V = randn(T, size(D, 2))
    u::V = randn(T, size(L, 1))
    v::V = randn(T, size(R, 1))
    η::V = randn(T, size(D, 1))

    # x, σ, u, v, η, L, R, D = cu.([x, σ, u, v, η, L, R, D])
    r₁, r₂, r₃ = copy.([u, v, η])
    Δr₁, Δr₂, Δr₃, _r₁, _r₂, _r₃ = copy.([r₁, r₂, r₃, r₁, r₂, r₃])

    prog = Progress(maxIter, desc="LOP-l2/l1ALT: λ = $(λ), α = $(round(α, digits=2))")
    for iter = 1:maxIter
        x_prev, σ_prev, u_prev, v_prev, η_prev,
        r₁_prev, r₂_prev, r₃_prev =
            copy([x, σ, u, v, η, r₁, r₂, r₃])

        function primal_params_update!()
            _r₁, _r₂, _r₃ = begin
                r₁ - γ₂ .* Δr₁,
                r₂ - γ₂ .* Δr₂,
                r₃ - γ₂ .* Δr₃
            end

            # Update x, σ, u, η
            x, σ, u, v, η = begin
                x + γ₁ .* (μ₁ .* L' * _r₁ + μ₂ .* R' * _r₂),
                σ + (γ₁ * μ₃) .* D' * _r₃,
                u - (γ₁ * μ₁) .* _r₁,
                v - (γ₁ * μ₂) .* _r₂,
                η - (γ₁ * μ₃) .* _r₃
            end

            # Computation of proximity operators
            σ, u, v, η = Vector.([σ, u, v, η])
            (v, σ), u, η = begin
                proxPersQuad(v, σ, γ₁ * λ),  # proximity operator of phi
                proxQuadError(u, y, γ₁), # proximity operator of quadratic error
                fastl1BallProj(η, α) # l1 ball projection
            end
            x, σ, u, v, η
        end

        function dual_params_update!()
            Δr₁, Δr₂, Δr₃ = begin
                μ₁ .* (L * x - u),
                μ₂ .* (R * x - v),
                μ₃ .* (D * σ - η)
            end
            r₁, r₂, r₃ = begin
                r₁ - γ₂ .* Δr₁,
                r₂ - γ₂ .* Δr₂,
                r₃ - γ₂ .* Δr₃
            end
            r₁, r₂, r₃
        end

        primal_params_update!()
        dual_params_update!()

        prev_params = [
            x_prev, σ_prev,
            u_prev, v_prev, η_prev,
            r₁_prev, r₂_prev, r₃_prev
        ]
        norm_2(params) = sqrt(sum((flip(norm)$1).(params) .^ 2))
        update_residual = norm_2(
            [x, σ, u, v, η, r₁, r₂, r₃] - prev_params
        ) / norm_2(prev_params)
        (update_residual < stopCri) && break
        next!(prog; showvalues=[
            (:iteration, iter),
            (:update_residual, update_residual)
        ])
    end
    finish!(prog)
    return Vector(x), Vector(σ)
end

function (param::LOPl2l1Reg)(
    y::V; mode::Symbol=:l1
)::Tuple{V,V} where {T<:Real,V<:Vector{T}}
    J = N = length(y)
    S = SparseMatrixCSC{T,Int64}
    L::S = sparse(eyes(J)) # identity matrix
    R::S = if (mode == :l1)
        sparse(eyes(N))
    elseif (mode == :tv)
        spmatDiff(N)
    else
        error("mode must be one of :l1, :tv")
    end
    D::SparseMatrixCSC{Int8,Int64} = spmatDiff(size(R, 1))
    x, σ = param(y, L, R, D)
    return x, σ
end

function (param::LOPl2l1Reg)(
    Y::Matrix{T};
    mode::Symbol=:l1
)::Tuple{Matrix{T},Union{Vector{T},Matrix{T}}} where {T}
    y = vec(Y)
    L::SparseMatrixCSC{T,Int64} = sparse(eyes(length(y))) # identity matrix
    R::SparseMatrixCSC{T,Int64} = if (mode == :l1)
        L # identity matrix
    elseif (mode == :tv)
        spmatDiff(size(Y)...) # difference matrix)
    end
    D::SparseMatrixCSC{Int8,Int64} = if (mode == :l1)
        spmatDiff(size(Y)...) # difference matrix
    elseif (mode == :tv)
        sparse(eyes(2)) ⊗ R # difference matrix
    end
    x, σ = param(y, L, R, D)
    X = reshape(x, size(Y))
    if (mode == :nuclear)
        return X, σ
    else
        Σ = if (mode == :l1)
            reshape(σ, size(Y))
        elseif (mode == :tv)
            reshape(σ, (2, 1) .* size(Y))
        else
            error("mode must be one of :l1, :tv, :nuclear")
        end
        return X, Σ
    end
end

function (param::LOPl2l1Reg)(
    Y::Array{T,3};
    mode::Symbol=:l1
)::Tuple{Array{T,3},Array{T,3}} where {T}
    if (mode == :l1)
        y = vec(Y)
        X = zeros(size(Y))
        Σ = zeros(size(Y))
        L::SparseMatrixCSC{T,Int64} = sparse(eyes(length(y))) # identity matrix
        R::SparseMatrixCSC{T,Int64} = L # identity matrix
        D::SparseMatrixCSC{Int8,Int64} = spmatDiff(size(Y)...) # difference matrix
        x, σ = param(y, L, R, D)
        X = reshape(x, size(Y))
        Σ = reshape(σ, size(Y))
        return X, Σ
    else
        error("mode must be :l1")
    end
end