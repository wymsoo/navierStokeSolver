import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import config

PLOTS_DIR = Path(__file__).resolve().parent / 'plots'

def PressureField(P, Nx, Ny, time):

    fig = plt.figure(figsize=(12, 8))
    
    # Create slices of pressure field
    # Middle slice in each direction
    mid_x = Nx // 2
    mid_y = Ny // 2
    
    # XY plane at mid Z
    ax1 = fig.add_subplot(111)
    im1 = ax1.imshow(P[:, :].T, origin='lower', 
                    extent=[0, 1, 0, 1], cmap='RdBu_r')
    ax1.set_title(f'Pressure Field at Time = {time*1000:.2f}ms')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1)
    
    plt.tight_layout()
    if (config.stopping_bc_conditions):
        filename = os.path.join(
            config.base_dir,
            'experiment',
            'images',
            f'{config.trial}',
            'pressure_field',
            f'flow_stopped_at_{config.stopping_bc_time}',
            f't={int(round(time * 1000))}ms_Fx={config.Fx_ext}_Fy={config.Fy_ext}.png',
        )
    else:
        filename = os.path.join(
            config.base_dir,
            'experiment',
            'images',
            f'{config.trial}',
            'pressure_field',
            'no_flow_stop',
            f't={int(round(time * 1000))}ms_Fx={config.Fx_ext}_Fy={config.Fy_ext}.png',
        )

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename)
    plt.close(fig)
    print(f"Pressure Plot saved to {filename}.")
