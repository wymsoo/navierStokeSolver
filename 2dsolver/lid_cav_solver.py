import numpy as np
import matplotlib.pyplot as plt
from advective import advective
from set_dirichlet_bc_1 import set_Dirichlet_BC
from viscous import viscous
from solve_poisson import Solve_Poisson
from velocityfieldplot import velocityField
from pressurefieldplot import PressureField
from stagger import stagger_back
from global_var import Nx, Ny, Re, D, G ,dx, dy, dt, max_ts, H, L, rho, viscosity, epsilon

def main():

    # Velocity fields (staggered)
    U = np.zeros((Nx - 1, Ny))      # u-velocity at x-faces
    V = np.zeros((Nx, Ny - 1))      # v-velocity at y-faces
    iteration = []
    loss = []
    
    # Time iteration loop
    time = dt
    
    for i in range(1, max_ts + 1):

        Ubc, Vbc = set_Dirichlet_BC(U, V)

        #Euler's method
        advectU, advectV = advective(Ubc, Vbc, dx, dy)
        viscousU, viscousV = viscous(Ubc, Vbc, Re, dx, dy, viscosity)

        Ustar = U + advectU * dt + viscosity/rho * viscousU * dt
        Vstar = V + advectV * dt + viscosity/rho * viscousV * dt


        # Solve Poisson's equation for pressure
        P = Solve_Poisson(Ustar, Vstar, dx, dy, Nx, Ny, dt)    
        # Compute pressure gradients
        Px = np.diff(P, axis=0) / dx
        Py = np.diff(P, axis=1) / dy 
        
        # Apply pressure correction
        U_new = Ustar - dt * Px/rho
        V_new = Vstar - dt * Py/rho
        
        # Visualization
        # if i % 1 == 0:
            # velocityField(U_new, V_new, P, Nx, Ny, time, H)
            # PressureField(P, Nx, Ny, time)
        
        U_loss = np.mean(abs(U_new-U))
        V_loss = np.mean(abs(V_new-V))
        total_loss = U_loss+V_loss
        U = U_new
        V = V_new

        if ((U_loss+V_loss)<epsilon):
            break

            
        print(f"Iteration: {i}")
        time += dt
        iteration.append(i)
        loss.append(total_loss)
    
    # Final time adjustment
    time -= dt

    y = np.linspace(-D/2,D/2,Ny)
    stag_U, stag_V = stagger_back(U, V)
    V_mag = np.sqrt(stag_U**2+stag_V**2)
    # V_avg = np.mean(V_mag,axis=0)
    V_avg = np.max(V_mag,axis=0)

    #benchmark
    print("R:",D,"Y:",y)
    V_theory = -1/viscosity/2*rho*G*(y**2-(D/2)**2)


    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)  # First subplot in a 2x1 grid

    ax1.plot(y,V_avg, label='V by solver')
    ax1.legend()
    ax1.set_title("Velocity Magnitude plotted against radius")
    ax1.set_xlabel('y')
    ax1.set_ylabel('velocity magnitude')

    ax2 = fig.add_subplot(2, 1, 2)  # First subplot in a 2x1 grid
    ax2.plot(y,V_theory, label='V theory')
    ax2.legend()
    ax2.set_title("Theoretical velocity Magnitude plotted against radius")
    ax2.set_xlabel('y')
    ax2.set_ylabel('velocity magnitude')

    plt.show()

    plt.plot(iteration,loss,label='total loss')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.show()

    print("Simulation completed.")




if __name__ == "__main__":
    main()