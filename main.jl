# Importing necessary libraries
using Random
using LinearAlgebra
using TOML
using Parameters
using Images: Gray, imresize
using FileIO
using TestImages

# Including source files
include("src/utils.jl")
include("src/norm.jl")
include("src/matDiff.jl")
include("src/LOPl2l1.jl")
include("src/make_data.jl")

# Parsing parameters from a TOML file
toml_params = TOML.parsefile("params.toml")["2d_tv_comp_params"];
# Unpacking parameters from the parsed TOML file
@unpack mode, λs, αs, SNRs = toml_params

# Setting up image name and loading the image
img_name = "peppers_gray";
img = imresize(testimage(img_name), 256 .* (1, 1));
# Converting the image to Float64 and saving it
Y_true = Float64.(Gray.(img));
output_path = joinpath("results", img_name, "true.png")
save(output_path, Gray.(Y_true))

# Printing experimental settings
experimental_settings = (img_name, "$(ndims(Y_true))d", mode)
println(experimental_settings)

# Initializing array to store results:
# SNR values of the recovered images for each combination of λ, α, and SNR
results_arr = zeros(length.([λs, αs, SNRs])...)

for (k, SNR) = enumerate(SNRs)
    Random.seed!(314) # Setting a random seed
    # Adding noise to the image with the specified SNR
    Y_noised = reshape(add_noise(vec(Y_true), SNR), size(Y_true))
    Gray_Y_noised = Gray.(clamp.(Y_noised, 0, 1))
    
    save(joinpath("results", img_name, "noised_SNR_$(SNR).png"), Gray_Y_noised)

    # Running the experiment for each combination of λ and α
    for (i, λ) = enumerate(λs),
        (j, α) = enumerate(αs)

        # Printing the current parameters
        hyper_params, fixed_params = (
            (λ=λ, α=α),
            (stopCri=1e-4, maxIter=1e4)
        )
        params = merge(hyper_params, fixed_params)
        println.(["", params])

        # Creating an instance of the LOPl2l1Reg struct with the specified parameters
        f = LOPl2l1Reg(params...)
        # Running the algorithm and measuring the time taken
        @time X, Σ = f(Y_noised; mode=Symbol(mode))

        # Saving the recovered image and the estimated noise level
        output_path = joinpath("results", img_name, "lambda_$(λ)__alpha_$(α)__SNR_$(SNR)")
        save("$(output_path)_X.png", Gray.(X))
        save("$(output_path)_Sigma.png", Gray.(Σ))
        recovered_SNR = snr(X, Y_true)
        results_arr[i, j, k] = recovered_SNR
    end
end

display(results_arr)