import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PLOTS_DIR = Path(__file__).resolve().parent / 'plots'

def PressureField(P, Nx, Ny, time):

    fig2 = plt.figure(figsize=(12, 8))

    ax1 = fig2.add_subplot(111)
    im1 = ax1.imshow(P.T, origin='lower', extent=[0, 1, 0, 1], cmap='RdBu_r')
    ax1.set_title(f'XY Plane ')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1)
    plt.suptitle(f'Pressure Field at Time = {time:.2f}')
    plt.tight_layout()
    PLOTS_DIR.mkdir(exist_ok=True)
    fig2.savefig(PLOTS_DIR / f'pressure_time_{int(time*1000)}.png')
    plt.close(fig2)
