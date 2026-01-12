import numpy as np
import matplotlib.pyplot as plt

def PressureField(P, Nx, Ny, time):

    fig2 = plt.figure(figsize=(12, 8))
    
    # Create slices of pressure field
    # Middle slice in each direction
    mid_x = Nx // 2
    mid_y = Ny // 2
    
    # XY plane at mid Z
    ax1 = fig2.add_subplot(111)
    im1 = ax1.imshow(P[:, :].T, origin='lower', 
                    extent=[0, 1, 0, 1], cmap='RdBu_r')
    ax1.set_title(f'XY Plane')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1)
    
    plt.tight_layout()
    plt.savefig(f'../pressure/graph_time_{int(time*1000)}.png')
    plt.close()
