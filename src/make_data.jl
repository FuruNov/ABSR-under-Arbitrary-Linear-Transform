using Distributions
db2mag(db) = 10^(db / 10)
mag2db(mag) = 10 * log10(mag)
snr(x, y) = mag2db(sum(x .^ 2) / sum((x - y) .^ 2))
signal_mag(x) = sum(x .^ 2) / length(x)
noise_mag(x, SNR) = signal_mag(x) / db2mag(SNR)
add_noise(x, SNR) = x + rand(Normal(0, sqrt(noise_mag(x, SNR))), size(x))