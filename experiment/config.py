import os

def init(nx, ny, flow, base_dir_path, output_dir_path, dt_var, rho_var, u_max_var, viscosity_var, fx_ext_var, fy_ext_var, max_ts_var=10000, epsilon_var=5e-10, L_var=1.0, W_var=1.0, stopping_bc_conditions_var=False, stopping_bc_time_var=None, relaxation_time_var=0.0):
    global Nx, Ny, dx, dy, dt_fixed, trial, base_dir, output_dir, rho, u_max, viscosity, max_ts, epsilon, L, W, Fx_ext, Fy_ext, stopping_bc_conditions, stopping_bc_time, relaxation_time
    Nx = nx
    Ny = ny
    L= L_var
    W = W_var
    dx = L/Nx
    dy = W/Ny
    dt_fixed = dt_var
    trial = flow
    base_dir = base_dir_path
    output_dir = output_dir_path
    os.makedirs(output_dir, exist_ok=True)
    # Parameters
    rho = rho_var
    u_max = u_max_var
    viscosity = viscosity_var
    Fx_ext = fx_ext_var
    Fy_ext = fy_ext_var
    max_ts = max_ts_var
    epsilon = epsilon_var
    stopping_bc_conditions = stopping_bc_conditions_var
    stopping_bc_time = stopping_bc_time_var
    relaxation_time = relaxation_time_var



