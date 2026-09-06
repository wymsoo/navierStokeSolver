import numpy as np
import matplotlib.pyplot as plt
from stagger import stagger_back
from global_var import D, Nx, Ny, viscosity, rho, G

def compare_with_theory(U,V):
    y = np.linspace(-D/2,D/2,Ny)
    x = np.linspace(0,D,Nx)
    stag_U, stag_V = stagger_back(U, V)
    V_mag = np.sqrt(stag_U**2+stag_V**2)
    V_avg = np.mean(V_mag,axis=0)
    V_max = np.max(V_mag,axis=0)

    #benchmark: rho=1000 viscosity=10 G=9.81 D=0.05
    V_theory = -1/viscosity/2*rho*G*(y**2-(D/2)**2)

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)  # First subplot in a 2x1 grid

    ax1.plot(y,V_avg, label='V by solver')
    ax1.legend()
    ax1.set_title("Mean Velocity plotted against radius")
    ax1.set_xlabel('y')
    ax1.set_ylabel('Velocity (Using V_mean)')

    ax2 = fig.add_subplot(2, 1, 2)  # second subplot in a 2x1 grid
    ax2.plot(y,V_theory, label='V theory')
    ax2.legend()
    ax2.set_title("Theoretical velocity Magnitude plotted against radius")
    ax2.set_xlabel('y')
    ax2.set_ylabel('velocity magnitude')

    plt.show()

