using LinearAlgebra
using PartialFunctions
using Distributed

include("utils.jl")
include("ProjOp.jl")

export proxPersQuad, proxQuadError

function proxQuadError(u::V, y::V, γ::T)::V where {T<:Real,V<:Array{T}}
    @assert size(u) == size(y) "u and y must have the same length"
    x = zeros(T, size(y))
    Threads.@threads for n in eachindex(x)
        x[n] = (γ * y[n] + u[n]) / (γ + 1)
    end
    return x
end

function proxPersQuad(x::T, σ::T, κ::T)::Tuple{T,T} where {T<:Real}
    # Compute positive root of cubic equation
    p, q = begin
        2σ / κ + 1,
        -2abs(x) / κ
    end
    s = findPosRootCubicEq(p, q)

    xₚ, σₚ = if 2κ * σ + x^2 <= κ^2
        (0, 0)
    elseif x == 0 && 2σ > κ
        (0, σ - κ / 2)
    else
        (x - κ * s * sign(x), σ + κ * (s^2 - 1) / 2)
    end
    return xₚ, σₚ
end

function proxPersQuad(
    x::V, σ::V, κ::T
)::Tuple{V,V} where {T<:Real,V<:Array{T}}
    xₚ, σₚ = zeros(T, size(x)), zeros(T, size(σ))
    Threads.@threads for n in eachindex(x)
        xₚ[n], σₚ[n] = proxPersQuad(x[n], σ[n], κ)
    end
    return xₚ, σₚ
end

function findPosRootCubicEq(p::T, q::T)::T where {T<:Real}
    # INPUT
    # p,q: coefficients of depressed cubic equation s^3 + ps + q = 0
    # OUTPUT
    # s: unique positive root of s^3 + ps + q = 0

    s = zero(p)
    D = -q^2 / 4 - p^3 / 27
    s = if D < 0
        sum([(cbrt ∘ op)(-q / 2, sqrt(-D)) for op in [+, -]])
    elseif D == 0
        2cbrt(-q / 2)
    else
        2(cbrt ∘ sqrt)(q^2 / 4 + D) * (cos(atan(-2sqrt(D) / q) / 3))
    end
    return s
end