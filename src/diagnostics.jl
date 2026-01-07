"""
    compute_ess(MMH_samples::Vector{MMH_sample}; max_lag::Int=100)

Compute the effective sample size (ESS) for each parameter and initial state.

# Arguments
- `MMH_samples`: MMH samples
- `max_lag`: maximum lag for autocorrelation estimation

# Returns
- `ess`: vector of ESS estimates for all variables
"""
function compute_ess(MMH_samples::Vector{MMH_sample}; max_lag::Int=100)
    # Get number of models.
    K = size(MMH_samples, 1)

    # Get number of parameters of the MMH samples.
    n_variables = length(MMH_samples[1].theta) + length(MMH_samples[1].x_init)

    # Fill matrix with the series of the parameters of the MMH samples.
    sample_matrix = Array{Float64}(undef, K, n_variables)
    for i in 1:K
        sample_matrix[i, :] .= vcat(MMH_samples[i].theta, MMH_samples[i].x_init)
    end

    # Calculate the autocorrelation.
    autocorrelation = autocor(sample_matrix, Array(0:max_lag); demean=true)
    ess = zeros(n_variables)

    for i in 1:n_variables
        # Sum autocorrelation of i-th variable until first negative or max_lag.
        autocorrelation_sum = 0.0
        for lag in 1:max_lag
            if autocorrelation[lag+1, i] < 0
                break
            end
            autocorrelation_sum += autocorrelation[lag+1, i]
        end

        # Compute the effective sample size.
        ess[i] = K / (1 + 2 * autocorrelation_sum)
    end

    return ess
end

"""
    compute_gelman_rubin(MMH_chains::Vector{Vector{MMH_sample}})

Compute the Gelman–Rubin statistic for each parameter and initial state from a vector of MMH chains.

# Arguments
- `MMH_chains`: vector of chains, where each chain is a vector of MMH samples

# Returns
- `R_hat`: vector of R̂ values, one for each variable
"""
function compute_gelman_rubin(MMH_chains::Vector{Vector{MMH_sample}})
    N = length(MMH_chains) # Number of chains
    K = length(MMH_chains[1]) # Number of samples per chain

    # Get number of parameters of the MMH samples.
    n_variables = length(MMH_chains[1][1].theta) + length(MMH_chains[1][1].x_init)

    # Extract samples from each chain
    sample_matrices_chains = Array{Float64}[]
    for chain in MMH_chains
        sample_matrix = Array{Float64}(undef, K, n_variables)
        for i in 1:K
            sample_matrix[i, :] .= vcat(chain[i].theta, chain[i].x_init)
        end

        push!(sample_matrices_chains, sample_matrix)
    end

    # Compute means and variances
    means = zeros(N, n_variables)
    variances = zeros(N, n_variables)
    for n in 1:N
        means[n, :] .= vec(mean(sample_matrices_chains[n], dims=1))
        variances[n, :] .= vec(var(sample_matrices_chains[n], dims=1, corrected=true))
    end

    # Between-chain and within-chain variance
    mean_overall = mean(means, dims=1)
    B = K / (N - 1) .* sum((means .- mean_overall) .^ 2, dims=1)
    W = mean(variances, dims=1)

    # Estimated marginal posterior variance and R̂
    V_hat = (K - 1) / K .* W .+ B / K
    R_hat = sqrt.(V_hat ./ W)

    # Clip from below at 1.0 for numerical consistency
    return max.(R_hat, 1.0)
end

"""
    compute_autocorrelation(MMH_samples::Vector{MMH_sample}; max_lag::Int=100)

Compute the autocorrelation function (ACF) of the MMH samples.

# Arguments
- `MMH_samples`: MMH samples
- `max_lag`: maximum lag at which to calculate the ACF

# Returns
- `autocorrelation`: matrix containing the ACF for each variable
"""
function compute_autocorrelation(MMH_samples::Vector{MMH_sample}; max_lag::Int=100)
    # Get number of models.
    K = size(MMH_samples, 1)

    # Get number of parameters of the MMH samples.
    n_variables = length(MMH_samples[1].theta) + length(MMH_samples[1].x_init)

    # Fill matrix with the series of the parameters of the MMH samples.
    sample_matrix = Array{Float64}(undef, K, n_variables)
    for i in 1:K
        sample_matrix[i, :] .= vcat(MMH_samples[i].theta, MMH_samples[i].x_init)
    end

    # Calculate the autocorrelation.
    autocorrelation = autocor(sample_matrix, Array(0:max_lag); demean=true)

    return autocorrelation
end