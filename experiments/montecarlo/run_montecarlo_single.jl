# This script runs the Monte-Carlo simulation for a single seed.
# It is called by the slurm batch script run_montecarlo_slurm.sh and not intended to be run directly.

using LinearAlgebra
using Random
using Distributions
using DifferentialEquations
using Printf
using OdeMMHPlanner

include(joinpath(@__DIR__, "..", "ExtendedKalmanFilter.jl"))

global hsl_available = false
try
    import HSL_jll
    global hsl_available = true
catch e
    @warn "Optional dependency HSL_jll not available. Falling back to the default solver MUMPS, which may result in increased runtimes or reduced numerical stability." exception = e
    global hsl_available = false
end

# Meal disturbance
struct Meal
    t_meal::Float64   # time of meal     min
    size::Float64     # glucose intake   mg/dL
end

function run_simulation(seed::Int)
    # Specify seed (for reproducible results).
    Random.seed!(seed)

    @printf("Running simulation with seed = %d\n", seed)
    flush(stdout)

    ## Learning parameters
    K = 100 # number of MMH samples in final stage
    k_d = 20 # number of samples to be skipped to decrease correlation (thinning)
    K_b = 200 # length of burn-in period for each stage
    M_chunk = 5 # number of datapoints added at each stage
    K_stage = 500 # number of samples per stage
    alpha = 0.85 # scaling of the proposal covariance
    regularizer = 0.0 # regularizer for proposal covariance
    rk4_step_size = 0.5 # step size for RK4 solver
    ODE_solver = RK4() # ODE solver algorithm
    ODE_solver_opts = (dt=rk4_step_size, adaptive=false) # ODE solver options

    ## State transition function
    # For the simulation, we use a Type-1-diabetes model (Bergman minimal model with insulin infusion and meal disturbance) with the following dynamics and parameters:
    #
    #   dG/dt = -p1 * (G - G_b) - X * G + D(t)
    #   dX/dt = -p2 * X + p3 * (I - I_b)
    #   dI/dt = -n * (I - I_b) + u(t)
    #
    #   States:
    #   x = [
    #           G,       plasma glucose         mg/dL
    #           X,       remote insulin effect  1/min
    #           I        plasma insulin         mU/L
    #       ]
    #
    #   Input:
    #   u(t)     insulin infusion    mU/L/min
    #
    #   Disturbance (assumed known):
    #   D(t)  glucose appearance  mg/dL/min
    #
    #   Parameters:
    #   parameters = [
    #                   p1,     glucose effectiveness       1/min               0.0
    #                   p2,     insulin action decay        1/min               0.015                  
    #                   p3,     insulin action gain         1/((mU/L)·min)      2e-6
    #                   n,      insulin clearance           1/min               0.21
    #                   G_b,    basal glucose               mg/dL               80.0
    #                   I_b,    basal insulin               mU/L                7.0
    #               ]
    #
    # Parameter values taken from:
    #   Ali, Sk Faruque, and Radhakant Padhi. 
    #   "Optimal blood glucose regulation of diabetic patients using single network adaptive critics."
    #   Optimal Control Applications and Methods 32.2 (2011): 196-214.

    # Number of states, etc.
    n_x = 3 # number of states
    n_u = 1 # number of control inputs
    n_y = 1 # number of outputs

    ## Meal disturbance
    # Define meal times and sizes.
    # Note that t = 0 corresponds to 6 pm.
    # The training window is from -T=-12 hours (6 am) to 0 hours (6 pm).
    meals = [
        Meal(-10.0 * 60.0, 60.0),   # breakfast at 8 am
        Meal(-5.0 * 60.0, 90.0),    # lunch at 1 pm
        Meal(1.0 * 60.0, 80.0),     # dinner at 7 pm
    ]

    # Glucose appearance function D(t)
    # We use the dynamics:
    #   dD/dt = -B * D
    function D_t(t; B=0.05)
        D = 0.0
        for m in meals
            if t >= m.t_meal
                delta_t = t - m.t_meal
                D += m.size * B * exp(-B * delta_t)
            end
        end
        return D
    end

    # Glucose-insulin dynamics
    function glucose_insulin_dynamics!(dx, parameters, x, u, t)
        p1, p2, p3, n, G_b, I_b = parameters
        G, X, I = x
        D = D_t(t)

        # Convert u to scalar if needed.
        u_scalar = u isa Number ? u : u[1]

        dx[1] = -p1 * (G - G_b) - X * G + D
        dx[2] = -p2 * X + p3 * (I - I_b)
        dx[3] = -n * (I - I_b) + u_scalar

        return dx
    end

    function glucose_insulin_dynamics(parameters, x, u, t)
        p1, p2, p3, n, G_b, I_b = parameters
        G, X, I = x
        D = D_t(t)

        # Convert u to scalar if needed.
        u_scalar = u isa Number ? u : u[1]

        dx1 = -p1 * (G - G_b) - X * G + D
        dx2 = -p2 * X + p3 * (I - I_b)
        dx3 = -n * (I - I_b) + u_scalar

        return [dx1, dx2, dx3]
    end

    # For the estimation, we assume that p_1, G_b, and I_b are known and only estimate p2, p3, and n as well as the state trajectory.
    # Thus, the unknown parameters are [p2, p3, n].
    # However, to ensure the parameters are positive while keeping the implementation simple, we work with parameters on a log scale.
    # Thus, the actual parameters used in the following are theta = [log(p2), log(p3), log(n)].
    function f_theta!(dx, theta, x, u, t)
        p1 = 0.0
        p2, p3, n = exp.(theta)
        G_b = 80.0
        I_b = 7.0
        theta_full = [p1, p2, p3, n, G_b, I_b]
        glucose_insulin_dynamics!(dx, theta_full, x, u, t)
        return dx
    end

    function f_theta(theta, x, u, t)
        p1 = 0.0
        p2, p3, n = exp.(theta)
        G_b = 80.0
        I_b = 7.0
        theta_full = [p1, p2, p3, n, G_b, I_b]
        return glucose_insulin_dynamics(theta_full, x, u, t)
    end

    ## Measurement model
    # Measurement function
    # In this example, we assume that the measurement function is known and that the glucose level is measured directly with some additive noise.
    const C = [1.0 0.0 0.0]

    function g_theta!(g, theta, x, u, t)
        g .= C * x
        return g
    end

    function g_theta(theta, x, u, t)
        return C * x
    end

    # Measurement noise model
    # We assume zero-mean Gaussian measurement noise with known standard deviation sigma_w.
    # Normalizing factors are omitted as they cancel out in the acceptance ratio.
    const sigma_w = 8.0 # standard deviation of zero-mean Gaussian measurement noise
    sample_w_theta(theta, N) = rand(Normal(0, sigma_w), N) # sample measurement noise
    log_pdf_w_theta(theta, w) = -0.5 * sum((w .^ 2) / (sigma_w^2)) # log pdf of measurement noise, scaling omitted

    ## Prior for the unknown parameters 
    # theta = [log(p2), log(p3), log(n)]
    # The following minimum and maximum values are reported in Ali and Padhi (2011):
    #
    #   p2: [0.01, 0.02]
    #   p3: [1.0e-6, 3.0e-6]
    #   n:  [0.12, 0.30]
    #
    # We use a log-normal prior for the parameters to ensure positivity.
    # We select the mean and variance such that approximately 95% of the values lie between the reported minimum and maximum values.
    const theta_mean = [
        -4.26,      # log(p2)
        -13.27,     # log(p3)
        -1.66       # log(n)
    ]

    const theta_var = [
        0.18^2,
        0.28^2,
        0.23^2
    ]

    # Log pdf of prior
    # Normalizing factors are omitted as they cancel out in the acceptance ratio.
    const theta_cov = Diagonal(theta_var) # covariance matrix of prior
    log_pdf_theta(theta) = -0.5 * sum((theta - theta_mean) .* (theta_cov \ (theta - theta_mean)))

    # Initial guess for model parameters
    theta_0 = theta_mean

    ## Prior for the initial state
    # We assume that the initial state follows a Gaussian distribution.
    # We assume the patient has been fasting before the first meal and initialize the state near the basal equilibrium with some variance.
    const x_init_mean = [80.0, 0.0, 7.0] # mean
    const x_init_var = [8.0^2, 0.001^2, 2.0^2] # variance
    sample_x_init() = rand(MvNormal(x_init_mean, Diagonal(x_init_var)))
    log_pdf_x_init(x_init) = -0.5 * sum((x_init - x_init_mean) .* (Diagonal(x_init_var) \ (x_init - x_init_mean)))

    # Initial guess for the initial state
    x_init_0 = x_init_mean

    # Proposal distribution for z = (theta, x_init)
    # For the staged MMH sampler, the proposal distribution is adapted at each stage based on the samples obtained so far.
    # Here, we only define the initial proposal distribution for the first stage.
    proposal_z_cov_0 = Diagonal(alpha * vcat(theta_var, x_init_var))

    # For a standard MMH sampler, the proposal distribution can be defined as follows:
    # proposal_z_cov = Diagonal(0.01 * vcat(theta_var, x_init_var))
    # propose_z(z) = rand(MvNormal(z, proposal_z_cov))
    # log_proposal_ratio_z(z_accepted, z_prop) = 0

    ## Parameters for data generation
    # We assume we start the inference at t=0 (6 pm).
    # The training window is 12 hours long (from 6 am to 6 pm).
    T_train = 12 * 60.0 # length of training window in minutes
    t_span = (-T_train, 0) # time span for data generation

    M = 200 # total number of measurements
    t_m = sort(-T_train .+ rand(M) .* T_train) # measurement times

    # Draw a patient and an initial state from the prior.
    theta_true = rand(MvNormal(theta_mean, theta_cov)) # [log(p2), log(p3), log(n)]
    x_init_true = rand(MvNormal(x_init_mean, Diagonal(x_init_var)))

    ## Input trajectory
    # To generate reasonable training data in the presence of meal disturbances, we apply a simple bolus
    # insulin input proportional to the meal size. The appropriate proportionality constant k_ins depends
    # on the patient’s insulin sensitivity, which varies between patients drawn from the prior.
    #
    # Because in the Monte-Carlo study each patient is sampled randomly, we avoid hand-tuning k_ins. Instead, 
    # we compute k_ins from the true parameters of the sampled patient so that the bolus input roughly counteracts 
    # the effect of the meal disturbance. This use of the true parameters is solely for generating the training
    # input trajectory.
    #
    # Importantly, the true patient parameters are not available to the MMH sampler or to the optimal
    # control problem; they are only used here to produce realistic training data.
    const p2_nom = 0.015
    const p3_nom = 2e-6
    const n_nom = 0.21

    const k_ins_nom = 0.22
    const S_nom = p3_nom / (p2_nom * n_nom)

    function k_ins_from_theta(theta)
        p2, p3, n = exp.(theta)
        S = p3 / (p2 * n)
        k = k_ins_nom * (S_nom / S)
        return k
    end

    const T_bolus = 60.0 # bolus duration [min]
    k_ins = k_ins_from_theta(theta_true) # scaling factor [mU/L/min per mg/dl]

    function u_t_bolus(t)
        u = zeros(1)
        for m in meals
            if m.t_meal <= t < m.t_meal + T_bolus
                u[1] += k_ins * m.size
            end
        end
        return u
    end

    # Generate state trajectory using ODE solver.
    t_save = vcat(t_m, t_span[2]) # also save the state at time t = 0
    ode_rhs(dx, x, p, t) = f_theta!(dx, theta_true, x, u_t_bolus(t), t)
    prob = ODEProblem(ode_rhs, x_init_true, t_span, theta_true)
    sol = solve(prob, Tsit5(); saveat=t_save)
    x = Array(sol)
    x_0_true = x[:, end-1] # state at time t=0
    x = x[:, 1:end-1]

    # Generate measurements.
    y = zeros(n_y, M)
    for m in 1:M
        y[:, m] = g_theta(theta_true, x[:, m], u_t_bolus(t_m[m]), t_m[m]) + sample_w_theta(theta_true, 1)
    end

    # Save u at measurement times.
    u = [u_t_bolus(t_m[m]) for m in 1:M]

    ## MMH sampling
    # Run a staged MMH sampler with ODE-integrated latent trajectories.
    MMH_samples, acceptance_ratio, runtime_sampling = staged_ODE_MMH(u_t_bolus, t_m, y, (-T_train, 0.0), K, K_b, k_d, f_theta!, g_theta!, log_pdf_w_theta, log_pdf_theta, theta_0, log_pdf_x_init, x_init_0, proposal_z_cov_0, M_chunk, K_stage, alpha; regularizer=regularizer, ODE_solver=ODE_solver, ODE_solver_opts=ODE_solver_opts, print_progress=true)

    ## Formulate and solve optimal control problem
    # Formulate the optimal control problem (OCP) using the MMH samples.
    # Define the cost function.
    const G_REF = 80.0  # reference glucose level in mg/dL
    const U_BASAL = 0.0 # basal level in mU/L/min
    const W_G = 1.0     # weight on glucose deviation
    const W_Gf = 10.0   # weight on glucose deviation at terminal time
    const W_U = 1e-3    # weight on insulin usage
    c(u, x, t) = W_G * (x[1] - G_REF)^2 + W_U * (u[1] - U_BASAL)^2 # running cost
    c_f(x) = W_Gf * (x[1] - G_REF)^2 # terminal cost

    # Define the constraints.
    const G_MIN = 70.0 # minimum glucose level in mg/dL
    const G_MAX = 180.0 # maximum glucose level in mg/dL
    const U_MIN = 0.0 # minimum insulin infusion in mU/L/min
    const U_MAX = 20.0 # maximum insulin infusion in mU/L/min

    h_scenario(u, x, t) = cat(
        x[1] .- G_MAX,
        G_MIN .- x[1],
        dims=1,
    )

    h_u(u, t) = cat(
        u .- U_MAX,
        U_MIN .- u,
        dims=1,
    )

    # Parameters for the OCP
    H = 6 * 60.0 # time horizon in minutes
    N = Int(H / rk4_step_size) # number of discretization points

    # IPOPT options
    # See https://coin-or.github.io/Ipopt/OPTIONS.html for more details.
    Ipopt_options = Dict("max_iter" => 5000, "max_wall_time" => 1800.0, "tol" => 1e-6, "acceptable_tol" => 1e-4, "linear_solver" => "mumps", "hessian_approximation" => "exact", "print_level" => 5)
    if hsl_available
        Ipopt_options["hsllib"] = HSL_jll.libhsl_path
        Ipopt_options["linear_solver"] = "ma57"
        Ipopt_options["ma57_pre_alloc"] = 10.0
    end

    # Run optimization.
    U_MMH, X_MMH, t_grid, J_MMH, solve_successful_MMH, iterations_MMH, runtime_optimization_MMH = solve_MMH_OCP(MMH_samples, n_u, f_theta, g_theta, H, N, c, c_f, h_scenario, h_u; solver_opts=Ipopt_options, rk4_step_size=rk4_step_size)

    # Get optimal input trajectory as a function of time.
    function u_t_MMH(t)
        if t <= t_grid[1]
            return @view U_MMH[:, 1]
        elseif t >= t_grid[end]
            return @view U_MMH[:, end]
        else
            i = searchsortedlast(t_grid, t)
            return @view U_MMH[:, i]
        end
    end

    ## Analyze solution
    # Simulate true system forward with the optimized input trajectory.
    pred_step_size = rk4_step_size
    t_pred = collect(0.0:pred_step_size:H)
    t_span_pred = (0.0, H)
    ode_rhs_MMH(dx, x, p, t) = f_theta!(dx, theta_true, x, u_t_MMH(t), t)
    prob_MMH = ODEProblem(ode_rhs_MMH, x_0_true, t_span_pred, theta_true)
    sol_MMH = solve(prob_MMH, Tsit5(); saveat=t_pred)
    x_true_MMH = Array(sol_MMH)

    # Compute the true cost of the optimized trajectory.
    J_running = sum(c(u_t_MMH(t_pred[i]), x_true_MMH[:, i], t_pred[i]) * pred_step_size for i in 1:length(t_pred)-1)
    J_terminal = c_f(x_true_MMH[:, end])
    J_true_MMH = J_running + J_terminal

    # Check if the constraints are satisfied.
    function check_constraints(t_grid, u_t, x_true, h_scenario, h_u, eps)
        h_scenario_satisfied = true
        h_u_satisfied = true

        for i in 1:length(t_grid)
            h = h_scenario(u_t(t_grid[i]), x_true[:, i], t_grid[i])
            if any(h .> eps)
                h_scenario_satisfied = false
                break
            end
        end

        for i in 1:length(t_grid)
            h = h_u(u_t(t_grid[i]), t_grid[i])
            if any(h .> eps)
                h_u_satisfied = false
                break
            end
        end

        return h_scenario_satisfied, h_u_satisfied
    end

    eps = 1e-4 # small tolerance
    h_scenario_satisfied_MMH, h_u_satisfied_MMH = check_constraints(t_grid, u_t_MMH, x_true_MMH, h_scenario, h_u, eps)

    @printf("Simulation done.\nStarting simulation of comparison method 1: Nominal model based optimal control\n")

    ## Comparison method 1: Nominal model based optimal control
    # In the following we compute the input trajectory using a nominal model with an EKF state estimate.
    # Define nominal model and EKF parameters.
    theta_nom = theta_mean

    f_nominal(x, u, t) = f_theta(theta_nom, x, u, t)
    g_nominal(x, u, t) = g_theta(theta_nom, x, u, t)
    Q_c = Diagonal([1e-4, 2e-8, 1e-3]) # process noise covariance (continuous-time)
    R = Diagonal([sigma_w^2]) # measurement noise covariance

    # Run EKF to estimate the state trajectory over the training data.
    x_hat_hist, runtime_ekf = ExtendedKalmanFilter.run_ekf_offline(x_init_mean, Diagonal(x_init_var), f_nominal, g_nominal, Q_c, R, t_m, y, u_t_bolus; dt=rk4_step_size)

    # Formulate the OCP using the nominal model and EKF state estimate at time t=0.
    model_nom = [MMH_sample(theta_nom, x_hat_hist[:, end], x_init_mean)]
    U_nom, X_nom, t_grid_nom, J_nom, solve_successful_nom, iterations_nom, runtime_optimization_nom = solve_MMH_OCP(model_nom, n_u, f_theta, g_theta, H, N, c, c_f, h_scenario, h_u; solver_opts=Ipopt_options, rk4_step_size=rk4_step_size)

    # Get nominal model based optimal input trajectory as a function of time.
    function u_t_nom(t)
        if t <= t_grid_nom[1]
            return @view U_nom[:, 1]
        elseif t >= t_grid_nom[end]
            return @view U_nom[:, end]
        else
            i = searchsortedlast(t_grid_nom, t)
            return @view U_nom[:, i]
        end
    end

    # Simulate the system forward with the nominal model based optimal input trajectory.
    ode_rhs_nom(dx, x, p, t) = f_theta!(dx, theta_true, x, u_t_nom(t), t)
    prob_nom = ODEProblem(ode_rhs_nom, x_0_true, t_span_pred, theta_true)
    sol_nom = solve(prob_nom, Tsit5(); saveat=t_pred)
    x_true_nom = Array(sol_nom)

    # Compute the true cost of trajectory with nominal model based input.
    J_running = sum(c(u_t_nom(t_pred[i]), x_true_nom[:, i], t_pred[i]) * pred_step_size for i in 1:length(t_pred)-1)
    J_terminal = c_f(x_true_nom[:, end])
    J_true_nom = J_running + J_terminal

    # Check if the constraints are satisfied.
    h_scenario_satisfied_nom, h_u_satisfied_nom = check_constraints(t_grid, u_t_nom, x_true_nom, h_scenario, h_u, eps)

    @printf("Simulation of comparison method 1 done.\nStarting simulation of comparison method 2: Scenario OCP with draws from the prior\n")


    ## Comparison method 2: Scenario OCP with draws from the prior
    # Initially, we intended to compare the performance of the proposed approach to an OCP based on samples drawn
    # directly from the prior distribution. However, in our experiments this approach consistently led to
    # infeasible OCPs as the prior uncertainty is too high to satisfy the state constraints.
    # Therefore, we comment out this section for now.

    #=
    # Draw K samples from the prior.
    MMH_samples_prior = Vector{MMH_sample}(undef, K)
    min_dist = 2 # minimum distance of initial state to constraint boundary
    for k in 1:K
        theta_sample = rand(MvNormal(theta_mean, theta_cov))
        # Sample initial state. Reject samples that are too close to the constraint boundary to improve feasibility of the OCP.
        x_t_sample = zeros(n_x)
        while true
            x_t_sample = sample_x_init()
            if G_MIN + min_dist <= x_t_sample[1] <= G_MAX - min_dist
                break
            end
        end
        x_init_sample = x_t_sample
        MMH_samples_prior[k] = MMH_sample(theta_sample, x_t_sample, x_init_sample)
    end

    # Solve the OCP using the prior samples.
    U_prior, X_prior, t_grid_prior, J_prior, solve_successful_prior, iterations_prior, runtime_optimization_prior = solve_MMH_OCP(MMH_samples_prior, n_u, f_theta, g_theta, H, N, c, c_f, h_scenario, h_u; solver_opts=Ipopt_options, rk4_step_size=rk4_step_size)

    # Get prior based optimal input trajectory as a function of time.
    function u_t_prior(t)
        if t <= t_grid_prior[1]
            return @view U_prior[:, 1]
        elseif t >= t_grid_prior[end]
            return @view U_prior[:, end]
        else
            i = searchsortedlast(t_grid_prior, t)
            return @view U_prior[:, i]
        end
    end

    # Simulate the system forward with prior based input.
    ode_rhs_prior(dx, x, p, t) = f_theta!(dx, theta_true, x, u_t_prior(t), t)
    prob_prior = ODEProblem(ode_rhs_prior, x_0_true, t_span_pred, theta_true)
    sol_prior = solve(prob_prior, Tsit5(); saveat=t_pred)
    x_true_prior = Array(sol_prior)

    # Compute the true cost of trajectory with prior based input.
    J_running = sum(c(u_t_prior(t_pred[i]), x_true_prior[:, i], t_pred[i]) * pred_step_size for i in 1:length(t_pred)-1)
    J_terminal = c_f(x_true_prior[:, end])
    J_true_prior = J_running + J_terminal

    # Check if the constraints are satisfied.
    h_scenario_satisfied_prior, h_u_satisfied_prior = check_constraints(t_grid, u_t_prior, x_true_prior, h_scenario, h_u, eps)

    @printf("Simulation of comparison method 2 done.\nStarting simulation of comparison method 3: No control input\n")
    =#

    ## Comparison method 3: No control input
    u_t_no_control(t) = 0.0

    ode_rhs_no_control(dx, x, p, t) = f_theta!(dx, theta_true, x, u_t_no_control(t), t)
    prob_no_control = ODEProblem(ode_rhs_no_control, x_0_true, t_span_pred, theta_true)
    sol_no_control = solve(prob_no_control, Tsit5(); saveat=t_pred)
    x_true_no_control = Array(sol_no_control)

    # Compute the true cost of trajectory without control.
    J_running = sum(c(u_t_no_control(t_pred[i]), x_true_no_control[:, i], t_pred[i]) * pred_step_size for i in 1:length(t_pred)-1)
    J_terminal = c_f(x_true_no_control[:, end])
    J_true_no_control = J_running + J_terminal

    # Check if the constraints are satisfied.
    h_scenario_satisfied_no_control, h_u_satisfied_no_control = check_constraints(t_grid, u_t_no_control, x_true_no_control, h_scenario, h_u, eps)

    @printf("Simulation of comparison method 3 done.\nStarting simulation of comparison method 4: Simple bolus insulin input\n")

    ## Comparison method 4: Simple bolus insulin input
    # Simulate the system forward using the simple bolus insulin input used for data generation.
    ode_rhs_bolus(dx, x, p, t) = f_theta!(dx, theta_true, x, u_t_bolus(t), t)
    prob_bolus = ODEProblem(ode_rhs_bolus, x_0_true, t_span_pred, theta_true)
    sol_bolus = solve(prob_bolus, Tsit5(); saveat=t_pred)
    x_true_bolus = Array(sol_bolus)

    # Compute the true cost of the trajectory with bolus input.
    J_running = sum(c(u_t_bolus(t_pred[i]), x_true_bolus[:, i], t_pred[i]) * pred_step_size for i in 1:length(t_pred)-1)
    J_terminal = c_f(x_true_bolus[:, end])
    J_true_bolus = J_running + J_terminal

    # Check if the constraints are satisfied.
    h_scenario_satisfied_bolus, h_u_satisfied_bolus = check_constraints(t_grid, u_t_bolus, x_true_bolus, h_scenario, h_u, eps)

    @printf("Simulation of comparison method 4 done.\n")

    ## Return results
    @printf("Finished simulation with seed = %d\n", seed)
    flush(stdout)

    return (
        seed=seed,
        x=x,
        y=y,
        u=u,
        t_m=t_m,
        theta_true=theta_true,
        x_init_true=x_init_true,
        MMH_samples=MMH_samples,
        acceptance_ratio=acceptance_ratio,
        runtime_sampling=runtime_sampling,
        U_MMH=U_MMH,
        X_MMH=X_MMH,
        t_grid=t_grid,
        J_MMH=J_MMH,
        solve_successful_MMH=solve_successful_MMH,
        iterations_MMH=iterations_MMH,
        runtime_optimization_MMH=runtime_optimization_MMH,
        t_pred=t_pred,
        x_true_MMH=x_true_MMH,
        J_true_MMH=J_true_MMH,
        h_scenario_satisfied_MMH=h_scenario_satisfied_MMH,
        h_u_satisfied_MMH=h_u_satisfied_MMH,
        runtime_ekf=runtime_ekf,
        U_nom=U_nom,
        X_nom=X_nom,
        t_grid_nom=t_grid_nom,
        J_nom=J_nom,
        solve_successful_nom=solve_successful_nom,
        iterations_nom=iterations_nom,
        runtime_optimization_nom=runtime_optimization_nom,
        x_true_nom=x_true_nom,
        J_true_nom=J_true_nom,
        h_scenario_satisfied_nom=h_scenario_satisfied_nom,
        h_u_satisfied_nom=h_u_satisfied_nom,
        U_prior=U_prior,
        X_prior=X_prior,
        t_grid_prior=t_grid_prior,
        J_prior=J_prior,
        solve_successful_prior=solve_successful_prior,
        iterations_prior=iterations_prior,
        runtime_optimization_prior=runtime_optimization_prior,
        J_true_prior=J_true_prior,
        h_scenario_satisfied_prior=h_scenario_satisfied_prior,
        h_u_satisfied_prior=h_u_satisfied_prior,
        x_true_prior=x_true_prior,
        x_true_no_control=x_true_no_control,
        J_true_no_control=J_true_no_control,
        h_scenario_satisfied_no_control=h_scenario_satisfied_no_control,
        h_u_satisfied_no_control=h_u_satisfied_no_control,
        x_true_bolus=x_true_bolus,
        J_true_bolus=J_true_bolus,
        h_scenario_satisfied_bolus=h_scenario_satisfied_bolus,
        h_u_satisfied_bolus=h_u_satisfied_bolus,
    )
end

# This function parses command-line arguments.
function parse_args(args::Vector{String})
    seed = nothing
    results_dir = nothing

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--seed"
            i += 1
            seed = parse(Int, args[i])
        elseif arg == "--results-dir"
            i += 1
            results_dir = args[i]
        else
            error("Unknown argument: $arg")
        end
        i += 1
    end

    if seed === nothing || results_dir === nothing
        error("Usage: julia mc_single_run.jl --seed <Int> --results-dir <path>")
    end

    return seed, results_dir
end

# Main program
if abspath(PROGRAM_FILE) == @__FILE__
    # Parse command-line arguments.
    seed, results_dir = parse_args(ARGS)

    # Make sure the results directory exists.
    isdir(results_dir) || mkpath(results_dir)

    # Start simulation.
    result = run_simulation(seed)

    # Save results.
    isdir(results_dir) || mkpath(results_dir)
    results_file = joinpath(results_dir, "results_seed_$(seed).jld2")
    @save results_file result
    println("Results saved to $results_file")
end