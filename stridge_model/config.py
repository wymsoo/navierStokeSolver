import argparse
import os

def initialize_fluid_config():
    """Initialize shared parameters used by the STRidge model and reports."""
    global Nx, Ny, dx, dy, epsilon, G, rho, D, viscosity, nu
    global c_advective, c_pressure, c_viscous
    global dt, u_max, v_max, Re, Fx_ext, Fy_ext

    Nx = 81
    Ny = 81
    dx = 1.0 / Nx
    dy = 1.0 / Ny
    epsilon = 5e-10
    G = 10.0
    rho = 10
    D = 1.0
    viscosity = 10
    nu = viscosity / rho
    c_advective = 1
    c_pressure = 1 / rho
    c_viscous = nu

    # Report metadata retained for compatibility with the report formatter.
    dt = 0.0001
    u_max = 1
    v_max = 1
    Re = (rho * u_max * D) / viscosity
    Fx_ext = 0.0
    Fy_ext = 0.0


def initialize_model_config(argv=None):
    parser = argparse.ArgumentParser(description="Identify Navier-Stokes terms from saved field snapshots.")
    parser.add_argument("--lam", type=float, default=1e-6)
    parser.add_argument("--tol_min", type=float, default=1e-2)
    parser.add_argument("--tol_max", type=float, default=0.15)
    parser.add_argument("--num_tol", type=int, default=10)
    parser.add_argument("--l0_penalty", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_start", type=int, default=1)
    parser.add_argument(
        "--input_directories",
        nargs="+",
        default=[
            os.path.join(os.path.dirname(__file__), "output_npz"),
        ],
    )
    parser.add_argument("--crop_t", type=int, default=2000, help="discard this many time layers from the end")
    parser.add_argument("--crop_y", type=int, default=2, help="discard this many spatial points near each wall")
    parser.add_argument("--crop_x", type=int, default=2, help="discard this many spatial points near each wall")
    parser.add_argument(
        "--report",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "stridge_report.md"),
        help="path for the generated Markdown report",
    )
    args = parser.parse_args(argv)
    if args.lam < 0 or args.l0_penalty < 0:
        parser.error("--lam and --l0_penalty must be non-negative")
    if args.tol_min <= 0 or args.tol_max <= args.tol_min:
        parser.error("--tol_min must be positive and smaller than --tol_max")
    if args.num_tol < 1 or args.data_start < 0:
        parser.error("--num_tol must be positive and --data_start must be non-negative")
    if min(args.crop_t, args.crop_x, args.crop_y) < 0:
        parser.error("crop values must be non-negative")
    return args



# def init(nx, ny, flow, base_dir_path, output_dir_path, dt_var, rho_var, u_max_var, viscosity_var, fx_ext_var, fy_ext_var, max_ts_var=10000, epsilon_var=5e-10, L_var=1.0, W_var=1.0, stopping_bc_conditions_var=False, stopping_bc_time_var=None, relaxation_time_var=0.0):
#     global Nx, Ny, dx, dy, dt_fixed, trial, base_dir, output_dir, rho, u_max, viscosity, max_ts, epsilon, L, W, Fx_ext, Fy_ext, stopping_bc_conditions, stopping_bc_time, relaxation_time, nu
#     Nx = nx
#     Ny = ny
#     L= L_var
#     W = W_var
#     dx = L/Nx
#     dy = W/Ny
#     dt_fixed = dt_var
#     trial = flow
#     base_dir = base_dir_path
#     output_dir = output_dir_path
#     os.makedirs(output_dir, exist_ok=True)
#     # Parameters
#     rho = rho_var
#     u_max = u_max_var
#     viscosity = viscosity_var
#     Fx_ext = fx_ext_var
#     Fy_ext = fy_ext_var
#     max_ts = max_ts_var
#     epsilon = epsilon_var
#     stopping_bc_conditions = stopping_bc_conditions_var
#     stopping_bc_time = stopping_bc_time_var
#     relaxation_time = relaxation_time_var
#     nu = viscosity_var/rho_var



