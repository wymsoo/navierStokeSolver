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
    stag_Ux = []
    stag_Uy = []
    stag_Vx = []
    stag_Vy = []
    
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

        
        U_loss = np.mean(abs(U_new-U))
        V_loss = np.mean(abs(V_new-V))
        total_loss = U_loss+V_loss
        U = U_new
        V = V_new

        if ((U_loss+V_loss)<epsilon):
            break

        stag_U, stag_V = stagger_back(U, V)
        stag_Ux.append(np.mean(np.mean(stag_U,axis=0)))
        stag_Vx.append(np.mean(np.mean(stag_V,axis=0)))
        stag_Uy.append(np.mean(np.mean(stag_U,axis=1)))
        stag_Vy.append(np.mean(np.mean(stag_V,axis=1)))

            
        print(f"Iteration: {i}")
        time += dt
        iteration.append(i)
        loss.append(total_loss)
    
    # Final time adjustment
    time -= dt

    velocityField(U, V, P, Nx, Ny, time, H)
    PressureField(P, Nx, Ny, time)

    y = np.linspace(-D/2,D/2,Ny)
    x = np.linspace(0,D,Nx)
    stag_U, stag_V = stagger_back(U, V)
    V_mag = np.sqrt(stag_U**2+stag_V**2)
    V_avg = np.mean(V_mag,axis=0)
    V_max = np.max(V_mag,axis=0)

    #benchmark
    print("R:",D,"Y:",y)
    V_theory = -1/viscosity/2*rho*G*(y**2-(D/2)**2)


    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)  # First subplot in a 2x1 grid

    ax1.plot(y,V_avg, label='V by solver')
    ax1.legend()
    ax1.set_title("Mean Velocity plotted against radius")
    ax1.set_xlabel('y')
    ax1.set_ylabel('Velocity (Using V_mean)')

    ax9 = fig.add_subplot(3, 1, 2)
    ax9.plot(y,V_max, label='V by solver')
    ax9.legend()
    ax9.set_title("Mean Velocity plotted against radius")
    ax9.set_xlabel('y')
    ax9.set_ylabel('Velocity (Using V_max)')

    ax2 = fig.add_subplot(3, 1, 3)  # First subplot in a 2x1 grid
    ax2.plot(y,V_theory, label='V theory')
    ax2.legend()
    ax2.set_title("Theoretical velocity Magnitude plotted against radius")
    ax2.set_xlabel('y')
    ax2.set_ylabel('velocity magnitude')

    plt.show()

    fig3 = plt.figure()
    ax7 = fig3.add_subplot(2, 1, 1)  # First subplot in a 2x1 grid

    ax7.plot(y,np.transpose(stag_U))
    ax7.set_title("U at different x values plotted against y")
    ax7.set_xlabel('y')
    ax7.set_ylabel('U along y-axis')

    ax8 = fig3.add_subplot(2, 1, 2)  # First subplot in a 2x1 grid
    ax8.plot(y,np.transpose(stag_V))
    ax8.set_title("V at different x values plotted against y")
    ax8.set_xlabel('y')
    ax8.set_ylabel('V along y-axis')

    plt.show()

    plt.plot(iteration,loss,label='total loss')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('Loss plotted against iterations')
    plt.show()

    fig2 = plt.figure()
    ax3 = fig2.add_subplot(2,2,1)
    ax3.plot(iteration, stag_Ux)
    ax3.set_xlabel('iterations')
    ax3.set_ylabel('Mean U along x-axis')
    ax4 = fig2.add_subplot(2,2,2)
    ax4.plot(iteration, stag_Vx)
    ax4.set_xlabel('iterations')
    ax4.set_ylabel('Mean V along x-axis')
    ax5 = fig2.add_subplot(2,2,3)
    ax5.plot(iteration, stag_Uy)
    ax5.set_xlabel('iterations')
    ax5.set_ylabel('Mean U along y-axis')
    ax6 = fig2.add_subplot(2,2,4)
    ax6.plot(iteration, stag_Vy)
    ax6.set_xlabel('iterations')
    ax6.set_ylabel('Mean V along y-axis')

    plt.show()


    print("Simulation completed.")




if __name__ == "__main__":
    main()