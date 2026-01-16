import numpy as np 
import matplotlib.pyplot as plt
from stagger import stagger_back

def velocityField(U, V, P, Nx, Ny, time, H):

    fig = plt.figure(figsize=(18, 12))
    Uplot, Vplot = stagger_back(U, V)

    #2D XY PLANE
    ax_xy = fig.add_subplot(1, 1, 1)
    
    x_2d = np.arange(0, Nx)
    y_2d = np.arange(0, Ny)
    X_xy, Y_xy = np.meshgrid(x_2d, y_2d, indexing='ij')
    
    U_xy = Uplot[::, ::]
    V_xy = Vplot[::, ::]
    
    vel_mag_xy = np.sqrt(U_xy**2 + V_xy**2)
    normxy = plt.Normalize(vmin=0, vmax=np.round(np.max(vel_mag_xy),6))
    colormapxy = plt.cm.plasma(normxy(vel_mag_xy))

    for i in range(0, Nx):
        for j in range(0, Ny):
            quiver_xy = ax_xy.quiver(X_xy, Y_xy, U_xy, V_xy, color=colormapxy[i,j])


    ax_xy.set_aspect('equal', adjustable='box')
    ax_xy.set_xlabel('X')
    ax_xy.set_ylabel('Y')
    ax_xy.set_title(f'XY Plane (t = {time:.3f})')
    ax_xy.set_xlim(0, Nx)
    ax_xy.set_ylim(0, Ny)
    cbar_xy = fig.colorbar(plt.cm.ScalarMappable(norm=normxy, cmap='plasma'), ax=ax_xy, boundaries=np.linspace(0,np.round(np.max(vel_mag_xy),6),5))
    cbar_xy.set_label('Velocity Magnitude')

    plt.suptitle(f'Velocity and Pressure Field at Time = {time:.2f}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'../velocity/graph_time_{int(time*1000000)}.png')
    plt.close()
