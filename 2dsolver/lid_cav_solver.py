import numpy as np
import matplotlib.pyplot as plt
from advective import advective
from set_dirichlet_bc_1 import set_Dirichlet_BC
from viscous import viscous
from solve_poisson import Solve_Poisson
from velocityfieldplot import velocityField
from pressurefieldplot import PressureField
from stagger import stagger_back
from global_var import Nx, Ny, Re, R, G ,dx, dy, dt, timesteps, H, L


def main():

    # Velocity fields (staggered)
    U = np.zeros((Nx - 1, Ny))      # u-velocity at x-faces
    V = np.zeros((Nx, Ny - 1))      # v-velocity at y-faces
    
    # Time iteration loop
    time = dt
    
    for i in range(1, timesteps + 1):
        # Set Boundary Conditions
        Ubc, Vbc = set_Dirichlet_BC(U, V)
        
        # Non-linear terms
        advectU, advectV = advective(Ubc, Vbc, dx, dy)
        # advective terms are zero
        
        # Viscous terms
        viscousU, viscousV = viscous(Ubc, Vbc, Re, dx, dy)
        

        Ustar = U + advectU * dt + viscousU * dt
        Vstar = V + advectV * dt + viscousV * dt

        # Solve Poisson's equation for pressure
        P = Solve_Poisson(Ustar, Vstar, dx, dy, Nx, Ny, dt)
        
        # Compute pressure gradients
        Px = np.diff(P, axis=0) / dx  # x-direction
        Py = np.diff(P, axis=1) / dy  # y-direction
        # Pz = np.diff(P, axis=2) / dz  # z-direction
        
        # Apply pressure correction
        U = Ustar - dt * Px
        V = Vstar - dt * Py
        # W = Wstar - dt * Pz
        V[0,0] = 0
        
        # Visualization
        if i % 1 == 0:
            velocityField(U, V, P, Nx, Ny, time, H)
            PressureField(P, Nx, Ny, time)
            # plt.tight_layout()
            # plt.draw()
            # plt.pause(0.01) 
        
        print(f"Iteration: {i}")
        time += dt
    
    # Final time adjustment
    time -= dt

    y = np.linspace(-R/2,R/2,Ny)
    stag_U, stag_V = stagger_back(U, V)
    print(stag_U)
    V_mag = np.sqrt(stag_U**2+stag_V**2)
    V_avg = np.mean(V_mag,axis=0)
    print(V_avg.shape)

    plt.plot(y,V_avg)
    plt.title("Velocity Magnitude plotted against radius")
    plt.xlabel('y')
    plt.ylabel('velocity magnitude')
    plt.show()

    print("Simulation completed.")




if __name__ == "__main__":
    main()