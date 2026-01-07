# This file contains an RK4 ODE solver implementation, which is used to formulate constraints in the optimal control problem.
# For inference, we use the DifferentialEquations.jl package to be flexible with the choice of ODE solver.
# However, to ensure consistency when the inferred ODE model is used in an optimal control problem, the RK4 solver from DifferentialEquations.jl should be used for inference.

"""
    rk4_step(f, x, u, t, h)

Perform a single classical 4th-order Runge–Kutta (RK4) integration step.

# Arguments
- `f`: dynamics function with inputs `(x, u, t)`
- `x`: state at time t
- `u`: input at time t
- `t`: current time
- `h`: time step

# Returns
- `x_next`: state at time t + h
"""
function rk4_step(f, x, u, t, h)
    k1 = f(x, u, t)
    k2 = f(x .+ 0.5h .* k1, u, t + 0.5h)
    k3 = f(x .+ 0.5h .* k2, u, t + 0.5h)
    k4 = f(x .+ h .* k3, u, t + h)
    x_next = x .+ (h / 6.0) .* (k1 .+ 2.0 .* k2 .+ 2.0 .* k3 .+ k4)
    return x_next
end

"""
    rk4_interval(f, x0, u_t, t_span, dt)

Integrate an ODE with input u_t(t) over a time interval using fixed-step RK4.

# Arguments
- `f`: dynamics function with inputs ``(x, u, t)``
- `x0`: state at time `t_span[1]`
- `u_t`: input trajectory; function of time ``t``
- `t_span`: tuple `(t_start, t_end)` specifying the integration interval
- `dt`: step size

# Returns
- `x`: state at time `t_span[2]`
"""
function rk4_interval(f, x0, u_t, t_span, dt)
    t_start, t_end = t_span
    N_steps = Int(ceil((t_end - t_start) / dt))
    x = x0
    t = t_start

    for i in 1:N_steps
        h = min(dt, t_end - t)
        u = u_t(t)
        x = rk4_step(f, x, u, t, h)
        t += h
    end

    return x
end

"""
    rk4_interval_const_u(f, x0, u, t_span, dt)

Integrate an ODE with a constant input u over a time interval using fixed-step RK4.

# Arguments
- `f`: dynamics function with inputs ``(x, u, t)``
- `x0`: state at time `t_span[1]`
- `u`: constant input applied for the whole integration interval
- `t_span`: tuple `(t_start, t_end)` specifying the integration interval
- `dt`: step size

# Returns
- `x`: state at time `t_span[2]`
"""
function rk4_interval_const_u(f, x0, u, t_span, dt)
    t_start, t_end = t_span
    N_steps = Int(ceil((t_end - t_start) / dt))
    x = x0
    t = t_start

    for i in 1:N_steps
        h = min(dt, t_end - t)
        x = rk4_step(f, x, u, t, h)
        t += h
    end

    return x
end