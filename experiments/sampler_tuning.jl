# This script demonstrates how to tune and evaluate the performance of the MMH sampler with ODE-integrated latent trajectories using a glucose-insulin dynamics model.
# It includes data generation, sampler execution, and various analyses such as autocorrelation, effective sample size, and trace plots.

using LinearAlgebra
using Random
using Distributions
using DifferentialEquations
using Base.Threads
using Plots
using Printf
using OdeMMHPlanner

# Specify seed (for reproducible results).
Random.seed!(1)

## Learning parameters
K = Int(1e5) # number of MMH samples in final stage
k_d = 0 # number of samples to be skipped to decrease correlation (thinning)
K_b = 200 # length of burn-in period for each stage
M_chunk = 5 # number of datapoints added at each stage
K_stage = 500 # number of samples per stage
alpha = 0.85 # scaling of the proposal covariance - should lead to an acceptance rate of about 25 %
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
struct Meal
    t_meal::Float64   # time of meal     min
    size::Float64     # glucose intake   mg/dL
end

const meals = [
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
const x_init_var = [10.0^2, 0.001^2, 2.0^2] # variance
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
# The test window is 6 hours long (from 6 pm to 12 am).
T_train = 12 * 60.0 # length of learning window in minutes
T_test = 6 * 60.0 # length of test window in minutes
t_span = (-T_train, T_test) # time span for data generation

M = 300 # total number of measurements
M_train = Int(floor(2 / 3 * M)) # training measurements (negative times)
M_test = M - M_train # test measurements (positive times)

t_train = sort(-T_train .+ rand(M_train) .* T_train)
t_test = sort(rand(M_test) .* T_test)
t_m = sort(vcat(t_train, t_test))

# True parameters and initial state for data generation.
theta_true = [
    log(0.015),     # log(p2)
    log(2e-6),      # log(p3)
    log(0.21)       # log(n)
]

x_init_true = [78.0, 5e-4, 8.0]

## Input trajectory
# We use a simple bolus insulin input proportional to the meal size.
const T_bolus = 60.0    # bolus duration [min]
k_ins = 0.22      # scaling factor [mU/L/min per mg/dl]

function u_t_bolus(t)
    u = 0.0
    for m in meals
        if m.t_meal <= t && t < m.t_meal + T_bolus
            u += k_ins * m.size # bolus amplitude proportional to meal size
        end
    end
    return u
end

# Generate state trajectory using ODE solver.
ode_rhs(dx, x, p, t) = f_theta!(dx, theta_true, x, u_t_bolus(t), t)
prob = ODEProblem(ode_rhs, x_init_true, t_span, theta_true)
sol = solve(prob, Tsit5(); saveat=t_m)
x = Array(sol)

# Generate measurements.
y = zeros(n_y, M)
for m in 1:M
    y[:, m] = g_theta(theta_true, x[:, m], u_t_bolus(t_m[m]), t_m[m]) + sample_w_theta(theta_true, 1)
end

# Split data into training and test data.
x_train = x[:, 1:M_train]
y_train = y[:, 1:M_train]

x_test = x[:, M_train+1:end]
y_test = y[:, M_train+1:end]

# Plot data.
p0 = plot()
plot!(t_m, x[1, :], label="\$x_1\$", lw=2)
scatter!(t_m, y[1, :], label="\$y\$", lw=2)
xlabel!("\$t\$")
ylabel!("\$x \\quad | \\quad y\$")
display(p0)

## MMH sampling
# Run an MMH sampler with ODE-integrated latent trajectories.
# MMH_samples, acceptance_ratio, runtime_sampling = ODE_MMH(u_t_bolus, t_train, y_train, (-T_train, 0.0), K, K_b, k_d, f_theta!, g_theta!, log_pdf_w_theta, log_pdf_theta, theta_0, log_pdf_x_init, x_init_0, propose_z, log_proposal_ratio_z; ODE_solver=ODE_solver, ODE_solver_opts=ODE_solver_opts, print_progress=true)

# Run a staged MMH sampler with ODE-integrated latent trajectories.
# The acceptance ratio should be around 25 %. If it is too high, increase alpha; if it is too low, decrease alpha.
MMH_samples, acceptance_ratio, runtime_sampling = staged_ODE_MMH(u_t_bolus, t_train, y_train, (-T_train, 0.0), K, K_b, k_d, f_theta!, g_theta!, log_pdf_w_theta, log_pdf_theta, theta_0, log_pdf_x_init, x_init_0, proposal_z_cov_0, M_chunk, K_stage, alpha; regularizer=regularizer, ODE_solver=ODE_solver, ODE_solver_opts=ODE_solver_opts, print_progress=true)

## Evaluate the obtained model on test data
# Simulate the posterior models forward.
t_pred = t_test
x_pred = Array{Float64}(undef, n_x, length(t_pred), K)
for k in 1:K
    # Get current model.
    x_t = MMH_samples[k].x_t
    ode_rhs_pred(dx, x, p, t) = f_theta!(dx, MMH_samples[k].theta, x, u_t_bolus(t), t)

    # Simulate system forward using a numerical ODE solver.
    prob_pred = ODEProblem(ode_rhs_pred, x_t, t_span, MMH_samples[k].theta)
    sol_pred = solve(prob_pred, ODE_solver; ODE_solver_opts..., saveat=t_pred)
    x_pred[:, :, k] .= Array(sol_pred)
end

# Plot predictions against true test data.
# The predicted trajectories should track the true outputs well.
states_to_plot = [1]
for i in states_to_plot
    # Calculate mean, maximum, and minimum prediction.
    x_pred_mean = mean(x_pred, dims=3)[i, :, 1]
    x_pred_max = maximum(x_pred, dims=3)[i, :, 1]
    x_pred_min = minimum(x_pred, dims=3)[i, :, 1]

    # Plot range of predictions.
    p1 = plot(t_pred, x_pred_min, fillrange=x_pred_max, alpha=0.35, label="all predictions", legend=:topright)

    # Plot true output.
    plot!(t_test, x_test[i, :], label="true trajectory", lw=2)

    # Plot mean prediction.
    plot!(t_pred, x_pred_mean, label="mean prediction", lw=2)

    title!("\$x_{$i}\$: predicted state vs. true state")
    ylabel!("\$x_{$i}\$")
    xlabel!("\$t\$")
    display(p1)
end

## Autocorrelation analysis
# Plot the autocorrelation function (ACF) of the samples.
# A well-mixed chain will show fast decay of autocorrelation. After thinning, the ACF should be near zero even at small lags.
max_lag = 50 # maximum lag for ACF computation
autocorrelation = compute_autocorrelation(MMH_samples; max_lag=max_lag)
n_variables = length(MMH_samples[1].theta) + length(MMH_samples[1].x_init)

p2 = plot(yticks=-1:0.1:1)
for i in 1:n_variables
    if i == 1
        # Plot the ACF of the elements of theta.
        plot!(Array(0:max_lag), autocorrelation[:, i], lc=:red, lw=2, label="\$\\theta\$")
    elseif 1 < i <= length(MMH_samples[1].theta)
        plot!(Array(0:max_lag), autocorrelation[:, i], lc=:red, lw=2, label="")
    elseif i == length(MMH_samples[1].theta) + 1
        # Plot the ACF of the elements of x_init.
        plot!(Array(0:max_lag), autocorrelation[:, i], lc=:green, lw=2, label="\$x(-T_{train})\$")
    elseif length(MMH_samples[1].theta) + 1 < i
        plot!(Array(0:max_lag), autocorrelation[:, i], lc=:green, lw=2, label="")
    end
end

title!("Autocorrelation Function (AFC)")
xlabel!("Lag")
ylabel!("AFC")
display(p2)

## Compute effective sample size (ESS)
# The ESS indicates how many effectively independent samples were drawn. Ideally, after thinning, ESS should approach K.
# The goal of tuning is to maximize the ESS per second.
ess = compute_ess(MMH_samples; max_lag=100)
@printf("Minimum ESS: %.1f (= %.2f / s)\n", minimum(ess), minimum(ess) / runtime_sampling)

## Plot parameter trace
# The trace should appear stationary and show no long-term trends after burn-in. Jump sizes should look reasonable.
sample_matrix = Array{Float64}(undef, K, n_variables)
for i in 1:K
    sample_matrix[i, :] .= vcat(MMH_samples[i].theta, MMH_samples[i].x_init)
end

for i in 1:n_variables
    p3 = plot(Array(0:K-1), sample_matrix[:, i], lw=2, legend=false)
    if i <= length(MMH_samples[1].theta)
        title!("Trace of \$\\theta_{$i}\$")
        ylabel!("\$\\theta_{$i}\$")
    else
        title!("Trace of \$x_{$(i-length(MMH_samples[1].theta))}(-T_{train})\$")
        ylabel!("\$x_{$(i-length(MMH_samples[1].theta))}\$")
    end
    xlabel!("Iteration")
    display(p3)
end

## Plot posterior distribution
# Plot the posterior histogram with overlaid priors and true values.
# If the data is informative, the posterior should be tighter than the prior and centered near the true value.
# However, these individual plots do not show correlation between parameters. 
# The approach might still learn the correlation between parameters and yield a much better prediction than suggested by the individual parameter posteriors.
bins = 100
true_values = [theta_true; x_train[:, 1]]
prior_pdf = Vector{Tuple{Vector{Float64},Vector{Float64}}}()
for i in 1:length(theta_mean)
    prior = Normal(theta_mean[i], sqrt(theta_var[i]))
    values = range(quantile(prior, 0.01), stop=quantile(prior, 0.99), length=500)
    push!(prior_pdf, (values, pdf(prior, values)))
end
for i in 1:length(x_init_mean)
    prior = Normal(x_init_mean[i], sqrt(x_init_var[i]))
    values = range(quantile(prior, 0.01), stop=quantile(prior, 0.99), length=500)
    push!(prior_pdf, (values, pdf(prior, values)))
end

for i in 1:n_variables
    p4 = histogram(sample_matrix[:, i], bins=bins, normalize=:pdf, label="Posterior")

    value, density = prior_pdf[i]
    plot!(value, density, label="Prior", linestyle=:dash, lw=2)

    plot!([true_values[i]], seriestype=:vline, label="True", lw=2)

    if i <= length(MMH_samples[1].theta)
        title!("Sample PDF of \$\\theta_{$i}\$")
        xlabel!("\$\\theta_{$i}\$")
    else
        title!("Sample PDF of \$x_{$(i-length(MMH_samples[1].theta))}(-T_{train})\$")
        xlabel!("\$x_{$(i-length(MMH_samples[1].theta))}\$")
    end
    ylabel!("Density")
    display(p4)
end

#=
## Optional: run multiple independent MMH chains and compute the Gelman–Rubin statistic R̂.
# R̂ quantifies convergence by comparing within-chain to between-chain variance.
# R̂ close to 1 (typically R̂ < 1.05) indicates good convergence across chains.
N = 10 # number of independent chains
MMH_chains = Vector{Vector{MMH_sample}}(undef, N)
@threads for n in 1:N
    theta_0 = rand(MvNormal(theta_mean, Diagonal(theta_var)))
    x_init_0 = rand(MvNormal(x_init_mean, Diagonal(x_init_var)))
    MMH_chains[n] = staged_ODE_MMH(u_t_bolus, t_train, y_train, (-T_train, 0.0), K, K_b, k_d, f_theta!, g_theta!, log_pdf_w_theta, log_pdf_theta, theta_0, log_pdf_x_init, x_init_0, proposal_z_cov_0, M_chunk, K_stage, alpha; regularizer=regularizer, ODE_solver=ODE_solver, ODE_solver_opts=ODE_solver_opts, print_progress=true)[1]
end

R_hat = compute_gelman_rubin(MMH_chains)
@printf("Maximum R̂: %.2f\n", maximum(R_hat))
=#

#=
## Save data to text file for external plotting.
include("data2txt.jl")
lag = collect(0:50)
data2txt("autocorrelation.txt", 
    "lag", lag, 
    "theta1", autocorrelation[:, 1], 
    "theta2", autocorrelation[:, 2], 
    "theta3", autocorrelation[:, 3], 
    "x1", autocorrelation[:, 4], 
    "x2", autocorrelation[:, 5], 
    "x3", autocorrelation[:, 6]
    )
=#