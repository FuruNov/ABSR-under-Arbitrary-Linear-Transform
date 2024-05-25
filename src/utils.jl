..(f, xss) = map(xs -> f.(xs), xss)
join_(xs...) = join(xs, "_")
filename(xs, extension) = "$(join_(xs...)).$(extension)"
logsdir(xs...) = projectdir("logs", xs...)
nameof(x) = string(x)

eyes(n) = LinearMap(x -> x, n, n; issymmetric=true)
eyes(n, m) = LinearMap(x -> x, n, m; issymmetric=true)
minmax_normalize(x) = (x .- minimum(x)) ./ (maximum(x) .- minimum(x))
# mean(x::Array, dims::Int) = sum(x, dims=dims) / size(x, dims)
nmse(X, Y) = norm(X - Y)^2 / norm(X)^2
l0_norm(X) = norm(X, 0)
soft_threshold(x::T, λ::T) where {T} = sign(x) * max(abs(x) - λ, 0)
soft_threshold(X::Array{T}, λ::T) where {T} = @. sign(X) * max(abs(X) - λ, 0)

function conv(
    x, # 信号
    h # カーネル
)
    y = zeros(length(x) + length(h) - 1)
    for i in eachindex(x)
        for j in eachindex(h)
            y[i+j-1] += x[i] * h[j]
        end
    end
    return y
end

function gauss(
    n # カーネルの窓幅
)
    x = -n:n
    return exp.(-x .^ 2 / (2 * n^2))
end

function cantor(x, n=1000)
    if n == 0
        return 0
    elseif x == 0
        return 0
    elseif x < 1 / 3
        return cantor(3x, n - 1) / 2
    elseif x < 2 / 3
        return 1 / 2
    elseif x < 1
        return 1 / 2 + cantor(3x - 2, n - 1) / 2
    else
        return 1
    end
end