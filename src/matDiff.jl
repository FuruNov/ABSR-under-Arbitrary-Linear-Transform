using LinearAlgebra
using LinearMaps
import Kronecker: ⊗
using SparseArrays
using PartialFunctions

export matDiff, spmatDiff

matDiff(X::Array, shift::Tuple) = circshift(X, shift) - X
matDiff(X::Array, dims::Int) = begin
    shift = zeros(ndims(X))
    shift[dims] = -1
    matDiff(X, Tuple(shift))
end

function matDiff(x::AbstractVector)
    N = length(x)
    y = zeros(N)
    @inbounds for i in eachindex(x, y)
        y[i] = x[mod1(i + 1, N)] - x[i]
    end
    return y
end

function matDiff!(y::AbstractVector, x::AbstractVector)
    N = length(x)
    axes(y) == axes(x) || throw(DimensionMismatch())
    @inbounds for i in eachindex(x, y)
        y[i] = x[mod1(i + 1, N)] - x[i]
    end
    return y
end

matDiff(N::Int) = LinearMap(matDiff!, N, N; ismutating=true)
function matDiff(N::Int, M::Int)::LinearMaps.BlockMap
    D(n) = LinearMap(matDiff!, n, n; ismutating=true)
    eyes(n) = LinearMap(x -> x, n, n; issymmetric=true)
    [D(M) ⊗ eyes(N); eyes(M) ⊗ D(N)]
end

function matDiff(X::Matrix)::Matrix
    D(n) = LinearMap(matDiff!, n, n; ismutating=true)
    N, M = size(X)
    [
        Matrix(D(N) * X)
        Matrix((D(M) * Matrix(X')))'
    ]
end

spmatDiff(N::Int)::SparseMatrixCSC{Int8,Int64} = spdiagm(
    0 => -ones(N - 1), 1 => ones(N - 1)
)[1:end-1, :]

spmatDiff(N::Int, M::Int)::SparseMatrixCSC{Int8,Int64} = sparse([
    spmatDiff(M) ⊗ sparse(I, N, N)
    sparse(I, M, M) ⊗ spmatDiff(N)
])

spmatDiff(M::Int, N::Int, L::Int)::SparseMatrixCSC{Int8,Int64} = sparse([
    spmatDiff(M) ⊗ sparse(I, N, N) ⊗ sparse(I, L, L)
    sparse(I, M, M) ⊗ spmatDiff(N) ⊗ sparse(I, L, L)
    sparse(I, M, M) ⊗ sparse(I, N, N) ⊗ spmatDiff(L)
])

function spmatDiff_cyclic(N::Int)::SparseMatrixCSC{Int8,Int64}
    S = spdiagm(
        0 => -ones(N), 1 => ones(N - 1)
    )
    S[N, 1] = 1
    return S
end

function spmatDiff_cyclic(N, M)::SparseMatrixCSC{Int8,Int64}
    sparse([
        spmatDiff_cyclic(M) ⊗ sparse(I, N, N)
        sparse(I, M, M) ⊗ spmatDiff_cyclic(N)
    ])
end

function spmatDiff_cyclic(M, N, L)::SparseMatrixCSC{Int8,Int64}
    sparse([
        spmatDiff_cyclic(M) ⊗ sparse(I, N, N) ⊗ sparse(I, L, L)
        sparse(I, M, M) ⊗ spmatDiff_cyclic(N) ⊗ sparse(I, L, L)
        sparse(I, M, M) ⊗ sparse(I, N, N) ⊗ spmatDiff_cyclic(L)
    ])
end

function spmatDiagmat2Vec(N::Int, M::Int)::SparseMatrixCSC{Int8,Int64}
    # Description: Convert a diagonal matrix to a vector.
    if (N == M)
        e = sparse(I, 1, N + 1)
        blockdiag([e for i in 1:N-1]..., sparse(I, 1, 1))
    else
        error("not implemented")
    end
end