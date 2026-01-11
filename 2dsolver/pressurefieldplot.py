import numpy as np
import matplotlib.pyplot as plt

def PressureField(P, Nx, Ny, time):

    fig2 = plt.figure(figsize=(12, 8))
    
    # Create slices of pressure field
    # Middle slice in each direction
    mid_x = Nx // 2
    mid_y = Ny // 2
    # mid_z = Nz // 2
    
    # XY plane at mid Z
    ax1 = fig2.add_subplot(111)
    im1 = ax1.imshow(P[:, :].T, origin='lower', 
                    extent=[0, 1, 0, 1], cmap='RdBu_r')
    ax1.set_title(f'XY Plane ')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1)
    
    # # XZ plane at mid Y
    # ax2 = fig2.add_subplot(222)
    # im2 = ax2.imshow(P[:, mid_y, :].T, origin='lower',extent=[0, 1, 0, 1], cmap='RdBu_r')
    # ax2.set_title(f'XZ Plane at Y={mid_y/Ny:.2f}')
    # ax2.set_xlabel('X')
    # ax2.set_ylabel('Z')
    # plt.colorbar(im2, ax=ax2)
    
    # # YZ plane at mid X
    # ax3 = fig2.add_subplot(223)
    # im3 = ax3.imshow(P[mid_x, :, :].T, origin='lower', extent=[0, 1, 0, 1], cmap='RdBu_r')
    # ax3.set_title(f'YZ Plane at X={mid_x/Nx:.2f}')
    # ax3.set_xlabel('Y')
    # ax3.set_ylabel('Z')
    # plt.colorbar(im3, ax=ax3)
    
    # # 3D isosurface
    # ax4 = fig2.add_subplot(224, projection='3d')
    
    # # Create isosurface (simplified - using scatter for demonstration)
    # # For proper isosurface, you'd need mayavi or plotly
    # X, Y, Z = np.meshgrid(np.linspace(0, 1, Nx),
    #                      np.linspace(0, 1, Ny),
    #                      np.linspace(0, 1, Nz), indexing='ij')
    
    # # Flatten arrays for scatter plot
    # x_flat = X.flatten()[::10]  # Subsample for performance
    # y_flat = Y.flatten()[::10]
    # z_flat = Z.flatten()[::10]
    # p_flat = P.flatten()[::10]
    
    # scatter = ax4.scatter(x_flat, y_flat, z_flat, c=p_flat, 
    #                      cmap='RdBu_r', alpha=0.6, s=10)
    # ax4.set_title('3D Pressure Field')
    # ax4.set_xlabel('X')
    # ax4.set_ylabel('Y')
    # ax4.set_zlabel('Z')
    
    plt.suptitle(f'Pressure Field at Time = {time:.2f}')
    plt.tight_layout()
    plt.savefig(f'../pressure/graph_time_{int(time*1000)}.png')
    plt.close()
