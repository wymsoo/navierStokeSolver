import numpy as np 
import matplotlib.pyplot as plt
from stagger import stagger_back

def velocityField(U, V, P, Nx, Ny, time, H):

    fig = plt.figure(figsize=(18, 12))
    ax2d = fig.add_subplot(2, 2, 1)
    
    # Convert staggered to collocated for visualization
    Uplot, Vplot = stagger_back(U, V)
    # Uplot = np.transpose(Uplot, (1, 0, 2))
    # Vplot = np.transpose(Vplot, (1, 0, 2))
    # Wplot = np.transpose(Wplot, (1, 0, 2))
    
    # Create meshgrid for quiver
    x = np.arange(0, Nx, H)
    y = np.arange(0, Ny, H)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Subsample for visualization
    U_subsampled = Uplot[::H, ::H]
    V_subsampled = Vplot[::H, ::H]

    ax2d.quiver(X, Y, U_subsampled, V_subsampled,alpha=0.6)
    
    ax2d.set_xlabel('X')
    ax2d.set_ylabel('Y')
    ax2d.set_title(f'3D Flow field at time = {time:.3f}')
    ax2d.set_xlim(0, Nx)
    ax2d.set_ylim(0, Ny)

    mid_x = Nx //2 
    mid_y = Ny // 2
    

    #2D XY PLANE
    ax_xy = fig.add_subplot(2, 2, 2)
    
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
    plt.savefig(f'../velocity/graph_time_{int(time*1000)}.png')
    plt.close()
