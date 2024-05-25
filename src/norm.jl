import LinearAlgebra
using Arpack
include("matDiff.jl")

export op_norm, tv_norm

op_norm(X::SparseMatrixCSC) = svds(X, nsv=1)[1].S[1]
op_norm(X::Matrix) = svd(X).S[1]
tv_norm(X::Array) = norm(matDiff(size(X)...) * vec(X), 1)

