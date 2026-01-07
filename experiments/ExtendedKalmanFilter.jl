# This module implements an extended Kalman filter (EKF) for state estimation of nonlinear state-space models.
# The EKF is used in the experiments to provide state estimates for a comparison method that uses a nominal model to formulate an optimal control problem.

module ExtendedKalmanFilter
using LinearAlgebra
using ForwardDiff
using Printf
using OdeMMHPlanner

mutable struct EKF
    x_hat::Vector{Float64} # current state estimate
    P::Matrix{Float64} # state covariance
    f::Function # dynamics function
    g::Function # measurement function
    Q::Matrix{Float64} # process noise covariance
    R::Matrix{Float64} # measurement noise covariance
end

"""
    jacobian_f(kf::EKF, x_prev::AbstractVector, u_t, t_prev::Real, t_k::Real, dt::Real)

This function computes the Jacobian of the discretized state transition function. The discretized state transition function is defined by integrating the dynamics from t_prev to t_k using RK4.

# Arguments
- `kf`: EKF instance
- `x_prev`: previous state estimate
- `u_t`: input trajectory; function of time t
- `t_prev`: previous time
- `t_k`: current time
- `dt`: step size for RK4 integration

# Returns
- Jacobian matrix df/dx evaluated at x_prev
"""
function jacobian_f(kf::EKF, x_prev::AbstractVector, u_t, t_prev::Real, t_k::Real, dt::Real)
    # Define discretized state transition function
    Phi(x) = OdeMMHPlanner.rk4_interval(kf.f, x, u_t, (t_prev, t_k), dt)

    # Return Jacobian
    return ForwardDiff.jacobian(Phi, x_prev)
end

"""
    jacobian_g(kf::EKF, x_hat::AbstractVector, u::AbstractVector, t::Real)

This function computes the Jacobian of the measurement function with respect to x.

# Arguments
- `kf`: EKF instance
- `x_hat`: state estimate
- `u`: input
- `t`: time

# Returns
- Jacobian matrix dg/dx evaluated at x_hat
"""
function jacobian_g(kf::EKF, x_hat::AbstractVector, u::AbstractVector, t::Real)
    g(x) = kf.g(x, u, t)
    return ForwardDiff.jacobian(g, x_hat)
end

"""
    ekf_predict!(kf, u_t, t_prev, t_k, dt)

This function performs the EKF prediction step from t_prev to t_k.

# Arguments
- `kf`: EKF instance
- `u_t`: input trajectory; function of time t
- `t_prev`: previous time
- `t_k`: current time
- `dt`: step size for RK4 integration

# Returns
- updated EKF instance after prediction step
"""
function ekf_predict!(kf::EKF, u_t, t_prev::Real, t_k::Real, dt::Real)
    # Nonlinear state propagation
    x_hat_prev = kf.x_hat
    x_hat_pred = OdeMMHPlanner.rk4_interval(kf.f, x_hat_prev, u_t, (t_prev, t_k), dt)

    # Linearization A = df/dx
    F = jacobian_f(kf, x_hat_prev, u_t, t_prev, t_k, dt)

    # Covariance prediction
    P_pred = F * kf.P * F' .+ kf.Q * (t_k - t_prev)

    kf.x_hat .= x_hat_pred
    kf.P .= P_pred
    return kf
end

"""
    ekf_update!(kf, y_meas)

This function performs the EKF measurement update step.

# Arguments
- `kf`: EKF instance
- `y_meas`: output measurement
- `u_k`: current input
- `t_k`: current time

# Returns
- updated EKF instance after measurement update
"""
function ekf_update!(kf::EKF, y_meas::AbstractVector, u_k::AbstractVector, t_k::Real)
    # Predicted measurement
    y_pred = kf.g(kf.x_hat, u_k, t_k)

    # Innovation
    innov = y_meas .- y_pred

    # Jacobian H = dg/dx
    H = jacobian_g(kf, kf.x_hat, u_k, t_k)

    # Innovation covariance
    S = H * kf.P * H' .+ kf.R

    # Kalman gain
    K = (kf.P * H') / S

    # State update
    kf.x_hat .+= K * innov

    # Covariance update
    kf.P .= (I - K * H) * kf.P

    return kf
end

"""
    run_ekf_offline(x0::AbstractVector,P0::AbstractMatrix,f::Function,g::Function,Q::AbstractMatrix,R::AbstractMatrix,t_m::AbstractVector,y::AbstractMatrix,u_t::Function;dt::Real)

Run EKF over a sequence of measurements at times `t_m` with values `y`.

# Arguments
- `x0`: initial state estimate
- `P0`: initial state covariance
- `f`: dynamics function
- `g`: measurement function
- `Q`: process noise covariance
- `R`: measurement noise covariance
- `t_m`: vector of measurement times
- `y`: matrix of measurements, each column corresponds to a measurement at time `t_m[i]`
- `u_t`: input trajectory; function of time t
- `dt`: step size for RK4 integration

# Returns
- `x_hat_hist`: matrix of state estimates at each measurement time, each column corresponds to the estimate at time `t_m[i]`
- `runtime_ekf`: total runtime of the EKF
"""
function run_ekf_offline(x0::AbstractVector, P0::AbstractMatrix, f::Function, g::Function, Q::AbstractMatrix, R::AbstractMatrix, t_m::AbstractVector, y::AbstractMatrix, u_t::Function; dt::Real)
    # Initialize.
    n_x = length(x0)
    M = length(t_m)
    x_hat_hist = zeros(n_x, M)
    kf = EKF(collect(x0), Matrix(P0), f, g, Matrix(Q), Matrix(R))

    # Time EKF.
    ekf_timer = time()

    # First measurement
    t_prev = t_m[1]
    ekf_update!(kf, y[:, 1], u_t(t_prev), t_prev)
    x_hat_hist[:, 1] .= kf.x_hat

    for k in 2:M
        t_k = t_m[k]

        # Prediction from t_prev to t_k
        ekf_predict!(kf, u_t, t_prev, t_k, dt)

        # Perform measurement update at t_k.
        ekf_update!(kf, y[:, k], u_t(t_k), t_k)
        x_hat_hist[:, k] .= kf.x_hat
        t_prev = t_k
    end

    runtime_ekf = time() - ekf_timer
    @printf("EKF runtime: %.4f seconds\n", runtime_ekf)

    return x_hat_hist, runtime_ekf
end

end