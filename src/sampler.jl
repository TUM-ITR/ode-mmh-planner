"""
    ODE_MMH(u_t::Function, t_m::AbstractVector{<:AbstractFloat}, y::AbstractMatrix{<:AbstractFloat}, t_span::Tuple{Float64,Float64}, K::Int, K_b::Int, k_d::Int, f_theta!::Function, g_theta!::Function, log_pdf_w_theta::Function, log_pdf_theta::Function, theta_0::AbstractVector{<:AbstractFloat}, log_pdf_x_init::Function, x_init_0::AbstractVector{<:AbstractFloat}, propose_z::Function, log_proposal_ratio_z::Function; ODE_solver=RK4(), ODE_solver_opts=(dt=0.1, adaptive=false), print_progress=true)

Run marginal Metropolis-Hastings (MMH) with ODE-integrated latent trajectory to obtain samples ``\\{\\theta, x(t=0)\\}^{[1:K]}`` from the joint parameter and state posterior distribution ``p(\\theta, x(t=0) \\mid \\mathbb{D}=\\{u(t), y_{1:M}\\})``.

# Arguments
- `u_t`: input trajectory; function of time ``t``
- `t_m`: time points of the output measurements, must lie within `t_span`
- `y`: output measurements
- `t_span`: timespan of the training trajectory
- `K`: number of models/scenarios to be sampled
- `K_b`: length of the burn in period
- `k_d`: number of models/scenarios to be skipped to decrease correlation (thinning)
- `f_theta!`: dynamics function parametrized by theta (mutating); has inputs ``(\\dot{x}, \\theta, x, u, t)``
- `g_theta!`: measurement function parametrized by theta (mutating); has inputs ``(g, \\theta, x, u, t)``
- `log_pdf_w_theta`: function that returns the logarithm of the probability density function of the measurement noise parametrized by theta; has inputs ``(\\theta, w)``
- `log_pdf_theta`: function that returns the logarithm of the probability density function of theta (prior); has input ``(\\theta)``
- `theta_0`: ``\\theta`` used to initialize the MMH sampler
- `log_pdf_x_init`: function that returns the logarithm of probability density function of ``x(t=-T)`` (prior); has input ``x(t=-T)``
- `x_init_0`: ``x(t=-T)`` used to initialize the MMH sampler
- `propose_z`: function that proposes new parameters ``z' = [\\theta'; x'(t=-T)]`` (proposal distribution); has input (``z = [\\theta; x(t=-T)``])
- `log_proposal_ratio_z`: function that computes the logarithm of the ratio of proposal densities for ``z``; has input arguments ``(z, z')``
- `ODE_solver`: ODE solver algorithm to use (we use RK4 as default as this is also used in the optimal control formulation)
- `ODE_solver_opts`: ODE solver setting
- `print_progress`: if true, the progress is printed (default: `true`)

# Returns
- `MMH_samples`: MMH samples
- `acceptance_ratio`: acceptance ratio of the MMH sampler
- `runtime`: runtime of the sampling process
"""
function ODE_MMH(u_t::Function, t_m::AbstractVector{<:AbstractFloat}, y::AbstractMatrix{<:AbstractFloat}, t_span::Tuple{Float64,Float64}, K::Int, K_b::Int, k_d::Int, f_theta!::Function, g_theta!::Function, log_pdf_w_theta::Function, log_pdf_theta::Function, theta_0::AbstractVector{<:AbstractFloat}, log_pdf_x_init::Function, x_init_0::AbstractVector{<:AbstractFloat}, propose_z::Function, log_proposal_ratio_z::Function; ODE_solver=RK4(), ODE_solver_opts=(dt=0.1, adaptive=false), print_progress=true)
    # Total number of samples to be generated
    K_total = K_b + 1 + (K - 1) * (k_d + 1)

    # Get number of parameters, etc.
    n_theta = length(theta_0)
    n_x = length(x_init_0)
    M = size(y, 2)

    # Initialize and pre-allocate.
    MMH_samples = Vector{MMH_sample}(undef, K)
    for k in 1:K
        MMH_samples[k] = MMH_sample(Array{Float64}(undef, n_theta), Array{Float64}(undef, n_x), Array{Float64}(undef, n_x))
    end
    accepted_samples = 0
    current_sample = 1
    n_proposals = 0
    z = vcat(theta_0, x_init_0)
    log_likelihood = -Inf
    log_p_theta = -Inf
    log_p_x_init = -Inf
    t_save = vcat(t_m, t_span[2])

    # Time MMH sampler.
    sampling_timer = time()

    if print_progress
        println("### Started MMH sampling")
    end

    while accepted_samples < K_total
        # Propose new parameters and a new initial state.
        n_proposals += 1
        z_prop = propose_z(z)
        theta_prop = z_prop[1:n_theta]
        x_init_prop = z_prop[n_theta+1:end]
        log_p_theta_prop = log_pdf_theta(theta_prop)
        log_p_x_init_prop = log_pdf_x_init(x_init_prop)
        if !isfinite(log_p_theta_prop) || !isfinite(log_p_x_init_prop)
            continue
        end

        # Update model, i.e., update right hand side of ODE, measurement function, and noise distribution.
        ode_rhs(dx, x, p, t) = f_theta!(dx, theta_prop, x, u_t(t), t)
        g!(g, x, u, t) = g_theta!(g, theta_prop, x, u, t)
        log_pdf_w(w) = log_pdf_w_theta(theta_prop, w)

        # Simulate system forward using a numerical ODE solver to determine likelihood of proposal.
        prob = ODEProblem(ode_rhs, x_init_prop, t_span, theta_prop)
        sol = solve(prob, ODE_solver; ODE_solver_opts..., saveat=t_save)
        x_prop = Array(sol)

        # Likelihood update based on measurement model (logarithms are used for numerical reasons).
        log_likelihood_prop = 0.0
        g_prop = Array{Float64}(undef, size(y, 1))
        w = Array{Float64}(undef, size(y, 1))
        for m in 1:M
            g!(g_prop, x_prop[:, m], u_t(t_m[m]), t_m[m])
            w .= y[:, m] - g_prop
            log_likelihood_prop += log_pdf_w(w)
        end

        # Compute acceptance probability.
        log_acceptance_ratio = log_likelihood_prop - log_likelihood + log_p_theta_prop - log_p_theta + log_p_x_init_prop - log_p_x_init + log_proposal_ratio_z(z, z_prop)

        # Accept or reject the proposal.
        if log(rand()) < log_acceptance_ratio
            accepted_samples += 1
            theta = theta_prop
            x_init = x_init_prop
            z = vcat(theta, x_init)
            log_likelihood = log_likelihood_prop
            log_p_theta = log_p_theta_prop
            log_p_x_init = log_p_x_init_prop

            # Use sample if the burn-in period is reached and the sample is not removed by thinning.
            if (K_b < accepted_samples) && (mod(accepted_samples - (K_b + 1), k_d + 1) == 0)
                MMH_samples[current_sample].theta .= theta_prop
                MMH_samples[current_sample].x_t .= x_prop[:, end]
                MMH_samples[current_sample].x_init .= x_init_prop
                current_sample += 1
            end

            # Print progress.
            if print_progress
                @printf("\e[32m%i/%i samples accepted\e[0m\n", accepted_samples, K_total)
            end
        else
            # Print progress.
            if print_progress
                @printf("\e[31m%i/%i samples accepted\e[0m\n", accepted_samples, K_total)
            end
        end
    end

    runtime = time() - sampling_timer
    acceptance_ratio = ((K_total - 1) / n_proposals) * 100

    # Print runtime and acceptance ratio.
    if print_progress
        @printf("### MMH sampling complete\nRuntime: %.2f s\nAcceptance ratio: %.2f %%\n", runtime, acceptance_ratio)
    end

    return MMH_samples, acceptance_ratio, runtime
end

"""
    staged_ODE_MMH(u_t::Function, t_m::AbstractVector{<:AbstractFloat}, y::AbstractMatrix{<:AbstractFloat}, t_span::Tuple{Float64,Float64}, K::Int, K_b::Int, k_d::Int, f_theta!::Function, g_theta!::Function, log_pdf_w_theta::Function, log_pdf_theta::Function, theta_0::AbstractVector{<:AbstractFloat}, log_pdf_x_init::Function, x_init_0::AbstractVector{<:AbstractFloat}, proposal_z_cov_0::AbstractMatrix{<:AbstractFloat}, M_chunk::Int, K_stage::Int, alpha::Union{AbstractFloat,AbstractVector{<:AbstractFloat}}; regularizer::Float64=1e-8, ODE_solver=RK4(), ODE_solver_opts=(dt=0.1, adaptive=false), print_progress=true)

Run marginal Metropolis-Hastings (MMH) with ODE-integrated latent trajectory with incremental data and adaptive proposal to obtain samples ``\\{\\theta, x(t=0)\\}^{[1:K]}`` from the joint parameter and state posterior distribution ``p(\\theta, x(t=0) \\mid \\mathbb{D}=\\{u(t), y_{1:M}\\})``.
The number of data points used in the likelihood computation is gradually increased by a fixed chunk size. At each stage, the MMH sampler is run on the current data subset, and the proposal distribution is adapted based on the empirical covariance of the collected samples.

# Arguments
- `u_t`: input trajectory; function of time ``t``
- `t_m`: time points of the output measurements, must lie within `t_span`
- `y`: output measurements
- `t_span`: timespan of the training trajectory
- `K`: number of models/scenarios to be sampled
- `K_b`: length of the burn in period
- `k_d`: number of models/scenarios to be skipped to decrease correlation (thinning)
- `f_theta!`: dynamics function parametrized by theta (mutating); has inputs ``(\\dot{x}, \\theta, x, u, t)``
- `g_theta!`: measurement function parametrized by theta (mutating); has inputs ``(g, \\theta, x, u, t)``
- `log_pdf_w_theta`: function that returns the logarithm of the probability density function of the measurement noise parametrized by theta; has inputs ``(\\theta, w)``
- `log_pdf_theta`: function that returns the logarithm of the probability density function of theta (prior); has input ``(\\theta)``
- `theta_0`: ``\\theta`` used to initialize the MMH sampler
- `log_pdf_x_init`: function that returns the logarithm of probability density function of ``x(t=-T)`` (prior); has input ``x(t=-T)``
- `x_init_0`: ``x(t=-T)`` used to initialize the MMH sampler
- `proposal_z_cov_0`: initial covariance of the multivariate normal proposal distribution for ``z = [\\theta; x(t=-T)]`` (e.g., covariance of the prior)
- `M_chunk`: number of data points added at each stage
- `K_stage`: number of samples per stage
- `alpha`: proposal scaling factor (scalar or vector; if a vector its i‐th element is used at stage i)
- `regularizer`: small constant added to the diagonal of the proposal covariance matrix to ensure positive definiteness (default: `1e-8`)
- `ODE_solver`: ODE solver algorithm to use (we use RK4 as default as this is also used in the optimal control formulation)
- `ODE_solver_opts`: ODE solver setting
- `print_progress`: if true, the progress is printed (default: `true`)

# Returns
- `MMH_samples`: final samples from full-data posterior
- `acceptance_ratio`: vector containing the acceptance ratio of each stage
- `runtime`: total runtime of the staged sampling process
"""
function staged_ODE_MMH(u_t::Function, t_m::AbstractVector{<:AbstractFloat}, y::AbstractMatrix{<:AbstractFloat}, t_span::Tuple{Float64,Float64}, K::Int, K_b::Int, k_d::Int, f_theta!::Function, g_theta!::Function, log_pdf_w_theta::Function, log_pdf_theta::Function, theta_0::AbstractVector{<:AbstractFloat}, log_pdf_x_init::Function, x_init_0::AbstractVector{<:AbstractFloat}, proposal_z_cov_0::AbstractMatrix{<:AbstractFloat}, M_chunk::Int, K_stage::Int, alpha::Union{AbstractFloat,AbstractVector{<:AbstractFloat}}; regularizer::Float64=1e-8, ODE_solver=RK4(), ODE_solver_opts=(dt=0.1, adaptive=false), print_progress=true)
    # Get number of parameters, etc.
    n_theta = length(theta_0)
    n_x = length(x_init_0)
    n_z = n_theta + n_x # total number of variables
    M = size(y, 2)

    # Determine the number of stages (each stage adds M_chunk data points)
    N_stages = ceil(Int, M / M_chunk)

    # Initialize current parameter vector
    theta = theta_0
    x_init = x_init_0
    proposal_cov = proposal_z_cov_0

    # Allocate an array to store samples from each stage.
    acceptance_ratio = zeros(N_stages)
    MMH_samples = Vector{MMH_sample}(undef, K)

    sampling_timer = time()

    if print_progress
        println("### Started staged MMH sampling")
    end

    for i in 1:N_stages
        # Select current data chunk - use data from 1 to T_i.
        M_i = min(i * M_chunk, M)
        y_i = y[:, 1:M_i]
        t_m_i = t_m[1:M_i]

        # Define the proposal function as sampling from a multivariate normal.
        propose_z(z) = rand(MvNormal(z, proposal_cov))
        log_ratio_proposal_pdf(z_accepted, z_prop) = 0

        # Call the base MMH sampler.
        if i < N_stages
            # For intermediate stages, sample K_stage samples without thinning.
            MMH_samples_stage, acceptance_ratio_stage = ODE_MMH(u_t, t_m_i, y_i, t_span, K_stage, K_b, 0, f_theta!, g_theta!, log_pdf_w_theta, log_pdf_theta, theta, log_pdf_x_init, x_init, propose_z, log_ratio_proposal_pdf; ODE_solver=ODE_solver, ODE_solver_opts=ODE_solver_opts, print_progress=false)
        else
            # In the final stage, sample K samples with thinning parameter k_d.
            MMH_samples_stage, acceptance_ratio_stage = ODE_MMH(u_t, t_m_i, y_i, t_span, K, K_b, k_d, f_theta!, g_theta!, log_pdf_w_theta, log_pdf_theta, theta, log_pdf_x_init, x_init, propose_z, log_ratio_proposal_pdf; ODE_solver=ODE_solver, ODE_solver_opts=ODE_solver_opts, print_progress=false)
        end

        # Save stage samples and acceptance ratio.
        acceptance_ratio[i] = acceptance_ratio_stage

        if i < N_stages
            # Update proposal covariance based on the empirical covariance
            Z = hcat([vcat(s.theta, s.x_init) for s in MMH_samples_stage]...)

            post_cov_theta = cov(transpose(Z))
            if isa(alpha, Number)
                proposal_cov = alpha * post_cov_theta + regularizer * Matrix(I, n_z, n_z)
            else
                proposal_cov = alpha[i] * post_cov_theta + regularizer * Matrix(I, n_z, n_z)
            end

            # Update the current state to the last sample from the current stage.
            theta = MMH_samples_stage[end].theta
            x_init = MMH_samples_stage[end].x_init
        else
            MMH_samples = MMH_samples_stage
        end
        if print_progress
            @printf("Stage %i/%i complete\nAcceptance ratio: %.2f %%\n", i, N_stages, acceptance_ratio_stage)
        end
    end

    # Print runtime and average acceptance ratio.
    average_acceptance_ratio = mean(acceptance_ratio)
    runtime = time() - sampling_timer
    if print_progress
        @printf("### Staged MMH sampling complete\nRuntime: %.2f s\nAverage acceptance ratio: %.2f %%\n",
            runtime, average_acceptance_ratio)
    end

    return MMH_samples, acceptance_ratio, runtime
end
