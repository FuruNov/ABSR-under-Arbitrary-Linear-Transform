using LinearAlgebra
using PartialFunctions

export l2Proj, l1BallProj, fastl1BallProj

l2Proj(u::Vector) = u / max(1, norm(u, 2))

function l1BallProj(η::V, α::S)::V where {S<:Real,V<:Vector{S}}
    M = length(η)
    ηₚ::V = if α == 0
        zeros(S, M)
    elseif norm(η, 1) <= α
        η
    else
        ρ::V = sort(abs.(η), rev=true)
        csρ::V = cumsum(ρ)
        # println("$(norm(η, 1)), $α")
        T::Int64 = maximum((1:M)[@. ρ > (csρ - α) / (1:M)])
        a::V = (max$0).(@. abs.(η) - (csρ[T] - α) / T)
        a .* sign.(η)
    end
    return ηₚ
end

function fastl1BallProj(η::V, α::S)::V where {S<:Real,V<:Vector{S}}
    M = length(η)
    ηₚ::V = if α == 0
        zeros(S, M)
    elseif norm(η, 1) <= α
        η
    else
        sign.(η) .* simplexProj(abs.(η), α)
    end
    return ηₚ
end

function simplexProj(y::V, a::S)::V where {S<:Real,V<:Vector{S}}
    v::typeof(y), v_t::typeof(y) = [y[1]], []
    ρ::typeof(a) = y[1] - a
    N::Int = length(y)

    for n::Int = 2:N
        if y[n] > ρ
            ρ += (y[n] - ρ) / (norm(v, 1) + 1)
            if ρ > y[n] - a
                push!(v, y[n])
            else
                append!(v_t, v)
                v, ρ = [y[n]], y[n] - a
            end
        end
    end

    if !isempty(v_t)
        for _y::typeof(a) in v_t
            if _y > ρ
                push!(v, _y)
                ρ += (_y - ρ) / norm(v, 1)
            end
        end
    end

    while true
        norm_v_prev::S = norm(v, 1)
        for _y::typeof(a) in v
            if _y <= ρ
                filter!(x -> x != _y, v)
                ρ += (ρ - _y) / norm(v, 1)
            end
        end
        if norm_v_prev == norm(v, 1)
            break
        end
    end

    τ::typeof(a), x::typeof(y) = ρ, zeros(N)
    for n::Int = 1:N
        x[n] = max(y[n] - τ, 0)
    end
    x
end