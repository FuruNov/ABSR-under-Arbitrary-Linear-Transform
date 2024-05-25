using LinearAlgebra, SparseArrays

function spmatFilter(kernel::Vector{T}, width::Int)::SparseMatrixCSC{T,Int64} where {T}
    kernel_width = length(kernel)
    height = width - kernel_width + 1
    return spdiagm([(n - 1) => k * ones(height) for (n, k) in enumerate(kernel)]...)[1:end-(kernel_width-1), :]
end

function spmatMean(kernel_width::Int, width::Int)::SparseMatrixCSC{Int8,Int64}
    spmatFilter(ones(kernel_width), width)
end

function spmat2ndDiff(width::Int)::SparseMatrixCSC{Int8,Int64}
    spmatFilter([1.0, -2.0, 1.0], width)
end

function spmatDiff(width::Int)::SparseMatrixCSC{Int8,Int64}
    spmatFilter([-1.0, 1.0], width)
end

function spmatGauss(kernel_width::Int, width::Int)::SparseMatrixCSC{Float64,Int64}
    kernel = [exp(-((n - 1) - (kernel_width - 1) / 2)^2 / 2) for n in 1:kernel_width]
    spmatFilter(kernel, width)
end