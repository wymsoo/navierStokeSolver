import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
import os
from models.operators.stagger import stagger_back
import config


def velocityField(U, V, P, Nx, Ny, time, path_dir):

    fig = plt.figure(figsize=(18, 12))
    Uplot, Vplot = stagger_back(U, V)
    ax_xy = fig.add_subplot(1, 1, 1)
    
    x_2d = np.arange(0, Nx)
    y_2d = np.arange(0, Ny)
    X_xy, Y_xy = np.meshgrid(x_2d, y_2d, indexing='ij')
    
    U_xy = Uplot[:, :]
    V_xy = Vplot[:, :]
    
    vel_mag_xy = np.sqrt(U_xy**2 + V_xy**2)
    normxy = plt.Normalize(vmin=0, vmax=np.round(np.max(vel_mag_xy),4))
    colormap = plt.colormaps['plasma']
    colormapxy = colormap(normxy(vel_mag_xy))

    for i in range(0, Nx):
        for j in range(0, Ny):
            quiver_xy = ax_xy.quiver(X_xy, Y_xy, U_xy, V_xy, vel_mag_xy, color=colormapxy[i,j])


    ax_xy.set_aspect('equal', adjustable='box')
    ax_xy.set_xlabel('X')
    ax_xy.set_ylabel('Y')
    ax_xy.set_title(f'XY Plane (t = {time:.3f})')
    ax_xy.set_xlim(-1, Nx)
    ax_xy.set_ylim(-1, Ny)
    cbar_xy = fig.colorbar(quiver_xy, ax=ax_xy)
    cbar_xy.set_label('Velocity Magnitude')

    plt.suptitle(f'Velocity Field at Time = {time*1000:.2f}ms', fontsize=14)
    plt.tight_layout()
    os.makedirs(path_dir,exist_ok=True)
    filename = os.path.join(
        path_dir,
        'experiment',
        'images',
        'velocity_field',
        f'{config.trial}_velocity_time_{int(round(time * 1000))}ms_Fx={config.Fx_ext}_Fy={config.Fy_ext}.png',
    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename)
    plt.close(fig)
    print(f"Velocity Plot saved to {filename}.")
