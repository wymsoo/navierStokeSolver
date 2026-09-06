import numpy as np

def setup_solver_parameters(
    rho: float,           # Density (kg/m^3)
    viscosity: float,     # Dynamic viscosity (Pa·s)
    L: float,             # Domain length (m)
    D: float,             # Domain height / pipe diameter (m)
    Nx: int,              # Number of grid cells in x
    Ny: int,              # Number of grid cells in y
    G: float = 9.81,      # Body force / gravity (m/s^2)
    safety_factor: float = 0.1  # Requested factor of 0.1
):
    # --- 1. Grid Discretization ---
    dx = L / Nx
    dy = D / Ny
    
    # --- 2. Kinematic Viscosity ---
    nu = viscosity / rho  # m^2/s
    
    # --- 3. Characteristic Velocity & Reynolds Number ---
    # Poiseuille flow driven by body force G: u_max = (rho * G * D^2) / (8 * viscosity)
    u_max = (rho * G * (D**2)) / (8.0 * viscosity)
    Re_max = (rho * u_max * D) / viscosity
    
    # --- 4. Mathematical Stability Limits ---
    # A) Viscous Diffusion Limit (Von Neumann Condition in 2D)
    dt_viscous_limit = 1.0 / (2.0 * nu * ((1.0 / dx**2) + (1.0 / dy**2)))
    
    # B) Advective CFL Limit (Courant-Friedrichs-Lewy Condition)
    # Avoid division by zero if velocity is zero initially
    dt_cfl_limit = min(dx, dy) / (u_max + 1e-12)
    
    # --- 5. Determine Governing Limit & Operational dt ---
    # The actual physical limit is whichever constraint is stricter (smaller)
    dt_absolute_limit = min(dt_viscous_limit, dt_cfl_limit)
    
    # Scale down by factor of 0.1 as requested
    dt_operational = safety_factor * dt_absolute_limit
    
    # --- Print Summary Report ---
    print("=" * 55)
    print("        NUMERICAL SOLVER PARAMETER REPORT        ")
    print("=" * 55)
    print(f"Fluid Density (rho)      : {rho} kg/m^3")
    print(f"Dynamic Viscosity (mu)   : {viscosity} Pa·s")
    print(f"Kinematic Viscosity (nu) : {nu:.6e} m^2/s")
    print(f"Domain Size (L x D)      : {L} m x {D} m")
    print(f"Grid Resolution          : {Nx} x {Ny} cells")
    print(f"Grid Spacing (dx, dy)    : {dx:.6f} m, {dy:.6f} m")
    print("-" * 55)
    print(f"Est. Peak Velocity (u_max): {u_max:.4f} m/s")
    print(f"Max Reynolds Number (Re) : {Re_max:.2f}")
    print("-" * 55)
    print(f"Viscous Limit (dt_visc)  : {dt_viscous_limit:.6e} s")
    print(f"Advective Limit (dt_cfl) : {dt_cfl_limit:.6e} s")
    print(f"Governing Strict Limit   : {dt_absolute_limit:.6e} s")
    print(f"--> OPERATIONAL dt (0.1x): {dt_operational:.6e} s")
    print("=" * 55)

    dt = dt_operational

    return rho, viscosity,L,D,Nx,Ny,dx,dy,u_max,Re_max,dt
    
    # return {
    #     "rho": rho,
    #     "viscosity": viscosity,
    #     "L": L,
    #     "D": D,
    #     "Nx": Nx,
    #     "Ny": Ny,
    #     "dx": dx,
    #     "dy": dy,
    #     "u_max": u_max,
    #     "Re": Re_max,
    #     "dt_limit": dt_absolute_limit,
    #     "dt": dt_operational
    # }
G = 9.81
max_ts = 10000
H = 0.005
epsilon = 1e-15

rho,viscosity,L,D,Nx,Ny,dx,dy,u_max,Re ,dt = setup_solver_parameters(
    rho=920.0,
    viscosity=0.081,
    L=0.05,
    D=0.005,  # Shrunk by factor of 10
    Nx=31,
    Ny=31,
    G=9.81,
    safety_factor=0.9
)
