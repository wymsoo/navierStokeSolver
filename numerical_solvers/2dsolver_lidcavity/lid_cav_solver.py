import numpy as np
import matplotlib.pyplot as plt
from advective import advective
from set_dirichlet_bc_1 import set_Dirichlet_BC
from viscous import viscous
from solve_poisson import Solve_Poisson
from velocityfieldplot import velocityField
from pressurefieldplot import PressureField
from plot_loss import plot_loss
import os

def compute_timestep(u, v, dx, dy, rho, mu, dt_prev, u_wall, cfl=0.3):
    """
    Computes adaptive timestep for Navier-Stokes solver.
    """
    # 1. Compute local velocity magnitudes
    u_mag = np.sqrt(u[:,:-1]**2 + v[:-1,:]**2)
    u_max = np.max(u_mag)
    
    # Fallback to boundary wall velocity at step 0 if interior flow is at rest
    if u_max < 1e-6:
        u_max = u_wall
    # Compute minimum grid spacing
    ds_min = min(dx, dy)
    # Compute Advective CFL Limit
    dt_cfl = cfl * (ds_min / u_max)
    # 4. Compute Viscous / Diffusion Limit
    dt_visc = 0.25 * (rho * (ds_min**2) / mu)
    # 5. Take strict minimum between CFL and Viscous criteria
    dt_target = min(dt_cfl, dt_visc)
    # 6. Apply 10% maximum growth cap from previous timestep (if available)
    if dt_prev is not None:
        dt_new = min(dt_target, 1.1 * dt_prev)
    else:
        dt_new = dt_target

    return dt_new



def main():
    iteration = []
    loss = []
    Nx = 51
    Ny = 51
    trial = f'lidcavity_{Nx}'
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(os.path.dirname(base_dir), f"stridge_model/output_npz/{trial}")
    os.makedirs(output_dir,exist_ok=True)
    # Parameters
    dt = 0.00001
    epsilon = 7e-11
    # Grid size
    dx = 1.0 / Nx
    dy = 1.0 / Ny
    G = 9.81
    rho = 100
    u_max = 1
    # D = 0.005
    D = 1.0
    viscosity = 10
    Re = (rho * u_max * D) / viscosity
    print("Reynolds number=", Re)

    # Velocity fields (staggered)
    U = np.zeros((Nx - 1, Ny))      # u-velocity at x-faces
    V = np.zeros((Nx, Ny - 1))      # v-velocity at y-faces
    
    # Number of iterations
    timesteps = 10000
    H = 5  # Subsampling for visualization
    
    # Time iteration loop
    time = dt
    
    for i in range(1, timesteps + 1):
        dt = compute_timestep(U, V, dx, dy, rho, viscosity, dt_prev=dt, u_wall=1.0, cfl=0.3) # cfl 40% safety buffer below the theoretical 2D limit ($0.5$) 
        # Set Boundary Conditions
        Ubc, Vbc = set_Dirichlet_BC(U, V)
        
        # Non-linear terms
        advectU, advectV = advective(Ubc, Vbc, dx, dy)
        viscousU, viscousV = viscous(Ubc, Vbc, Re, dx, dy)
        
        # Compute intermediate velocities
        Ustar = U - advectU * dt + viscousU * dt
        Vstar = V - advectV * dt + viscousV * dt 
        
        # Solve Poisson's equation for pressure
        P = Solve_Poisson(Ustar, Vstar, dx, dy, Nx, Ny, dt)
        # Compute pressure gradients
        Px = np.diff(P, axis=0) / dx  # x-direction
        Py = np.diff(P, axis=1) / dy  # y-direction
        
        # Apply pressure correction
        U_new = Ustar - dt * Px / rho
        V_new = Vstar - dt * Py / rho

        U_loss = np.sum((U_new-U)**2/(Nx*Ny)) #MSE
        V_loss = np.sum((V_new-V)**2/(Nx*Ny))
        total_loss = U_loss+V_loss
        print(f"iteration{i} total loss:", total_loss)
        U = U_new
        V = V_new

        if ((U_loss+V_loss)<epsilon):
            break

        iteration.append(i)
        loss.append(total_loss)
    
        # velocityField(U, V, P, Nx, Ny, time, H)
        # PressureField(P, Nx, Ny, time)
        if (i % 1 == 0):

            # velocityField(U,V,P, Nx, Ny, time, H)
            # np.savetxt(os.path.join(u_dir, f"u_velocity_t={i}.txt"), U, delimiter="\t")
            # np.savetxt(os.path.join(v_dir, f"v_velocity_t={i}.txt"), V, delimiter="\t")
            # np.savetxt(os.path.join(p_dir, f"pressure_field_t={i}.txt"), P, delimiter="\t")
            np.savez_compressed(
                                os.path.join(output_dir,f"data_t={i}.npz"),
                                u=U,
                                v=V,
                                p=P,
                                dt=dt,
                                dx=dx,
                                dy=dy
                            )

        # print(f"Iteration: {i}")
        time += dt
    # Final time adjustment
    plot_loss(iteration,loss)
    velocityField(U,V,P, Nx, Ny, time)
    u_max = np.max(U)
    v_max = np.max(V)
    max_v = max(u_max, v_max)
    print("max v:",max_v)
    Re = (rho * max_v * D) / viscosity
    print("Reynolds number=", Re)
    print("Simulation completed.")




if __name__ == "__main__":
    main()