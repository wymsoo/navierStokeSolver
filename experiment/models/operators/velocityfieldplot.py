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
    # Keep colors comparable between snapshots: 1.0 is the reference speed.
    normxy = plt.Normalize(vmin=0, vmax=1.0, clip=True)
    colormap = plt.colormaps['plasma']

    quiver_xy = ax_xy.quiver(
        X_xy,
        Y_xy,
        U_xy,
        V_xy,
        vel_mag_xy,
        cmap=colormap,
        norm=normxy,
        angles='xy',
        scale_units='xy',
        scale=1.0,
    )


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
    if (config.stopping_bc_conditions):
        filename = os.path.join(
            path_dir,
            'experiment',
            'images',
            f'{config.trial}',
            'velocity_field',
            f'flow_stopped_at_{config.stopping_bc_time}',
            f't={int(round(time * 1000))}ms_Fx={config.Fx_ext}_Fy={config.Fy_ext}.png',
        )
    else:
        filename = os.path.join(
            path_dir,
            'experiment',
            'images',
            f'{config.trial}',
            'velocity_field',
            'no_flow_stop',
            f't={int(round(time * 1000))}ms_Fx={config.Fx_ext}_Fy={config.Fy_ext}.png',
        )

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename)
    plt.close(fig)
    print(f"Velocity Plot saved to {filename}.")

