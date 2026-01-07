"""
    solve_MMH_OCP(MMH_samples::Vector{MMH_sample}, n_u::Int, f_theta::Function, g_theta::Function, H::Float64, N::Int, c::Function, c_f::Function, h_scenario::Function, h_u::Function; U_init=nothing, MMH_samples_pre_solve=nothing, K_warmup=0, solver_opts=nothing, rk4_step_size=0.1, print_progress=true)

Solve the continous time scenario optimal control problem of the following form:

``\\min_{u(\\cdot)} J_{sc}(u(\\cdot)) := \\frac{1}{K} \\sum_{k=1}^{K} [ c_f(x^{[k]}(H)) + \\int_0^H c(u(t), x^{[k]}(t), t) dt],``

subject to: 
```math
\\begin{aligned}
\\forall t \\in [0,H], \\forall k  \\in \\mathbb{N}_{\\leq K}, \\\\
x^{[k]}(t) = \\Phi(t; \\theta^{[k]}, x^{[k]}(0), u(\\cdot)), \\\\
h_{scenario}(u(t), x^{[k]}(t), t) \\leq 0.
\\end{aligned}
```
Here, ``\\Phi(t; \\theta^{[k]}, x^{[k]}(0), u(\\cdot))`` denotes the solution at time ``t`` of the ODE ``\\dot{x}(t) = f_{\\theta}(x(t), u(t))`` with initial condition ``x(0) = x^{[k]}(0)`` and parameter ``\\theta`` under the input trajectory ``u(\\cdot)``.

# Arguments
- `MMH_samples`: MMH samples
- `n_u`: number of control inputs
- `f_theta`: dynamics function parametrized by theta (non mutating); has inputs ``(\\theta, x, u, t)``
- `g_theta`: measurement function parametrized by theta (non mutating); has inputs ``(\\theta, x, u, t)``
- `H`: horizon of the OCP (as time)
- `N`: number of discretization steps in the horizon
- `c`: function with input arguments ``(u(t), x^{[k]}(t), t)`` that returns the running cost of scenario k at time t
- `c_f`: function with input argument ``x^{[k]}(H)`` that returns the terminal cost of scenario k
- `h_scenario`: function with input arguments ``(u(t), x^{[k]}(t), t)`` that returns the constraint vector for scenario k at time t; a feasible solution must satisfy ``h_{\\mathrm{scenario}} \\leq 0`` for all scenarios and all discretization points
- `h_u`: function with input arguments ``(u(t), t)`` that returns the constraint vector for the control inputs; a feasible solution satisfy ``h_u \\leq 0`` at all discretization points
- `U_init`: initial guess for the input trajectory - either a `n_u x N` array, a function with input argument t, or nothing (default: nothing)
- `MMH_samples_pre_solve`: if provided, an initial guess for the input trajectory is obtained by solving an OCP with the samples in `MMH_samples_pre_solve` only
- `K_warmup`: if `K_warmup > 0` and `MMH_samples_pre_solve` is provided, an initial guess for the the input trajectory is obtained in a two stage process: first, an OCP with only `K_warmup` samples from `MMH_samples_pre_solve` is solved and then an OCP with all samples in `MMH_samples_pre_solve`
- `solver_opts`: SolverOptions struct containing options of the solver
- `rk4_step_size`: step size of the RK4 integrator used to simulate the system dynamics
- `print_progress`: if set to true, the progress is printed

# Returns
- `U_opt`: piecewise constant optimal input trajectory, array of dimension `n_u x N`
- `X_opt`: states at the discretization points for all scenarios, array of dimension `n_x x N x K`
- `t_grid`: time grid of the discretization points, array of dimension `N + 1`
- `J_sc_opt`: optimal cost ``J_{sc}``
- `solve_successful`: true if the optimization was successful, false otherwise
- `iterations`: number of iterations of the solver
- `runtime`: runtime of the optimization
"""
function solve_MMH_OCP(MMH_samples::Vector{MMH_sample}, n_u::Int, f_theta::Function, g_theta::Function, H::Float64, N::Int, c::Function, c_f::Function, h_scenario::Function, h_u::Function; U_init=nothing, MMH_samples_pre_solve=nothing, K_warmup=0, solver_opts=nothing, rk4_step_size=0.1, print_progress=true)
    # Time optimization.
    optimization_timer = time()

    # Get number of states, etc.
    K = length(MMH_samples)
    n_x = size(MMH_samples[1].x_t, 1)

    # Get discretization points.
    dt = H / N
    t_grid = collect(0.0:dt:H)

    # Determine initialization.
    # With a good initialization the runtime of the optimization can be reduced significantly.
    # If a set of samples `MMH_samples_pre_solve` is provided, the OCP is solved with the samples in `MMH_samples_pre_solve` first to obtain an initialization for the problem.
    # If additionally `K_warmup` is provided, a problem that only considers `K_warmup` randomly selected scenarios of `MMH_samples_pre_solve` is solved first to obtain an initialization for the problem with all scenarios in `MMH_samples_pre_solve`.
    if !(MMH_samples_pre_solve === nothing)
        if K_warmup > 0
            # Sample the K_warmup scenarios that are considered for the initialization.
            warmup_samples = sample(1:size(MMH_samples_pre_solve, 1), K_warmup)

            if print_progress
                println("###### Started pre-solving step")
            end

            # Solve OCP with K_warmup samples from MMH_samples_pre_solve.
            U_init = solve_MMH_OCP(MMH_samples_pre_solve[warmup_samples], f_theta, g_theta, sample_v_theta, sample_w_theta, H, J, h_scenario, h_u; J_u=J_u, U_init=U_init, K_warmup=0, solver_opts=solver_opts, print_progress=print_progress)[1]

            # Solve OCP with all samples from MMH_samples_pre_solve.
            U_init = solve_MMH_OCP(MMH_samples_pre_solve, f_theta, g_theta, sample_v_theta, sample_w_theta, H, J, h_scenario, h_u; J_u=J_u, U_init=U_init, K_warmup=0, solver_opts=solver_opts, print_progress=print_progress)[1]

            if print_progress
                println("###### Pre-solving step complete, switching back to the original problem")
            end
        else
            if print_progress
                println("###### Started pre-solving step")
            end

            # Solve OCP with all samples from MMH_samples_pre_solve.
            U_init = solve_MMH_OCP(MMH_samples_pre_solve, f_theta, g_theta, sample_v_theta, sample_w_theta, H, J, h_scenario, h_u; J_u=J_u, U_init=U_init, K_warmup=0, solver_opts=solver_opts, print_progress=print_progress)[1]

            if print_progress
                println("###### Pre-solving step complete, switching back to the original problem")
            end
        end
    elseif U_init isa Function
        # Evaluate function at discretization points.
        U_init_array = zeros(n_u, N)
        for n in 1:N
            t = t_grid[n]
            U_init_array[:, n] = U_init(t)
        end
        U_init = U_init_array
    elseif U_init === nothing
        U_init = zeros(n_u, N)
    end

    # Piecewise-constant control from U_init on uniform grid
    U_t_init(t) = begin
        n = floor(Int, t / dt) + 1
        n = clamp(n, 1, N)
        @view U_init[:, n]
    end

    # Determine initial guess for X.
    X_init = Array{Float64}(undef, n_x, N + 1, K)
    for k in 1:K
        f(x, u, t) = f_theta(MMH_samples[k].theta, x, u, t)
        X_init[:, 1, k] .= MMH_samples[k].x_t

        # Propagate from t=0 to t=H using RK4 with input U_t_init.
        for n in 1:N
            X_init[:, n+1, k] .= rk4_interval(f, X_init[:, n, k], U_t_init, (t_grid[n], t_grid[n+1]), rk4_step_size)
        end
    end

    # Set up OCP.
    OCP = Model(Ipopt.Optimizer)
    @variable(OCP, U[i=1:n_u, n=1:N], start = U_init[i, n])
    @variable(OCP, X[i=1:n_x, n=1:N+1, k=1:K], start = X_init[i, n, k])

    # Add dynamic constraints for the states.
    # The runtime could likely be improved by defining the intermediate RK4 steps as optimization variables as well.
    # However, this is not done here to keep the implementation simple.
    for k in 1:K
        f(x, u, t) = f_theta(MMH_samples[k].theta, x, u, t)
        for n in 1:N
            @constraint(OCP, X[:, n+1, k] .== rk4_interval_const_u(f, X[:, n, k], U[:, n], (t_grid[n], t_grid[n+1]), rk4_step_size))
        end
    end

    # Set the initial state.
    for k in 1:K
        for i in 1:n_x
            fix(X[i, 1, k], MMH_samples[k].x_t[i]; force=true)
        end
    end

    # Add scenario constraints.
    for k in 1:K
        for n in 1:N
            @constraint(OCP, h_scenario(U[:, n], X[:, n, k], (n - 1) * dt) .<= 0.0)
        end
    end

    # Add input constraints.
    for n in 1:N
        @constraint(OCP, h_u(U[:, n], (n - 1) * dt) .<= 0.0)
    end

    # Define objective.
    J_sc = (1.0 / K) * sum(c_f(X[:, N+1, k]) + sum(c(U[:, n], X[:, n, k], (n - 1) * dt) * (t_grid[n+1] - t_grid[n]) for n in 1:N) for k in 1:K)
    @objective(OCP, Min, J_sc)

    # Set options.
    if !(solver_opts === nothing)
        for opt in solver_opts
            set_attributes(OCP, opt)
        end
    end

    if !print_progress
        set_silent(OCP)
    end

    # Solve OCP.
    if print_progress
        println("### Started optimization algorithm")
    end

    optimize!(OCP)
    runtime = time() - optimization_timer

    if print_progress
        @printf("### Optimization complete\nRuntime: %.2f s\n", runtime)
    end

    U_opt = value.(U)
    X_opt = value.(X)
    J_sc_opt = objective_value(OCP)
    solve_successful = is_solved_and_feasible(OCP)
    iterations = MOI.get(OCP, MOI.BarrierIterations())

    if !solve_successful
        @warn("The optimization did not converge to an optimal and feasible solution. The solver returned: $(termination_status(OCP))")
    end

    return U_opt, X_opt, t_grid, J_sc_opt, solve_successful, iterations, runtime
end