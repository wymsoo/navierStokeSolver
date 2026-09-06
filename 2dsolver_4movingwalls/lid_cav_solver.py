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


def main():
    iteration = []
    loss = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "output")
    u_dir = os.path.join(output_dir, "u_velocity_field")
    v_dir = os.path.join(output_dir, "v_velocity_field")
    p_dir = os.path.join(output_dir, "pressure_field")
    os.makedirs(u_dir, exist_ok=True)
    os.makedirs(v_dir, exist_ok=True)
    os.makedirs(p_dir, exist_ok=True)

    # Parameters
    dt = 0.0001
    epsilon = 5e-10
    # Grid size
    Nx = 31
    Ny = 31
    dx = 1.0 / Nx
    dy = 1.0 / Ny
    G = 9.81
    rho = 1000
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

        time += dt
    
        # velocityField(U, V, P, Nx, Ny, time, H)
        # PressureField(P, Nx, Ny, time)
        if (i % 1 == 0):
            # velocityField(U,V,P, Nx, Ny, time, H)
            np.savetxt(os.path.join(u_dir, f"u_velocity_t={i}.txt"), U, delimiter="\t")
            np.savetxt(os.path.join(v_dir, f"v_velocity_t={i}.txt"), V, delimiter="\t")
            np.savetxt(os.path.join(p_dir, f"pressure_field_t={i}.txt"), P, delimiter="\t")

        # print(f"Iteration: {i}")
        time += dt
    # Final time adjustment
    plot_loss(iteration,loss)
    velocityField(U,V,P, Nx, Ny, time)
    print("U_max = ", np.max(U))
    print("V_max = ", np.max(V))
    print("Simulation completed.")




if __name__ == "__main__":
    main()