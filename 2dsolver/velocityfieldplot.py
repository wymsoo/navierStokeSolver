import numpy as np 
import matplotlib.pyplot as plt
from stagger import stagger_back

def velocityField(U, V, P, Nx, Ny, time, H):

    fig = plt.figure(figsize=(18, 12))
    # ax3d = fig.add_subplot(2, 2, 1, projection='3d')
    
    # Convert staggered to collocated for visualization
    Uplot, Vplot = stagger_back(U, V)
    # Uplot = np.transpose(Uplot, (1, 0, 2))
    # Vplot = np.transpose(Vplot, (1, 0, 2))
    # Wplot = np.transpose(Wplot, (1, 0, 2))
    
    # # Create meshgrid for quiver
    # x = np.arange(0, Nx, H)
    # y = np.arange(0, Ny, H)
    # z = np.arange(0, Nz, H)
    # X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # # Subsample for visualization
    # U_subsampled = Uplot[::H, ::H, ::H]
    # V_subsampled = Vplot[::H, ::H, ::H]
    # W_subsampled = Wplot[::H, ::H, ::H]

    # ax3d.quiver(X, Y, Z, U_subsampled, V_subsampled, W_subsampled,
    #             length=2.0, normalize=True, alpha=0.6)
    
    # ax3d.set_xlabel('X')
    # ax3d.set_ylabel('Y')
    # ax3d.set_zlabel('Z')
    # ax3d.set_title(f'3D Flow field at time = {time:.3f}')
    # ax3d.view_init(elev=20, azim=-60)
    # ax3d.set_xlim(0, Nx)
    # ax3d.set_ylim(0, Ny)
    # ax3d.set_zlim(0, Nz)

    # mid_x = Nx //2 
    # mid_z = Nz // 2
    # mid_y = Ny // 2
    

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

    # # 2D XZ PLANE
    # ax_xz = fig.add_subplot(2, 2, 3)
    
    # z_2d = np.arange(0, Nz)
    # X_xz, Z_xz = np.meshgrid(x_2d, z_2d, indexing='ij')
    
    # U_xz = Uplot[::, mid_y, ::]
    # W_xz = Wplot[::, mid_y, ::]
    
    # vel_mag_xz = np.sqrt(U_xz**2 + W_xz**2)
    # normxz = plt.Normalize(vmin=0, vmax=np.round(np.max(vel_mag_xz),6))
    # colormapxz = plt.cm.plasma(normxz(vel_mag_xz))

    # for i in range(0, Nx):
    #     for j in range(0, Ny):
    #         quiver_xz = ax_xz.quiver(X_xz, Z_xz, U_xz, W_xz, color=colormapxz[i,j])

    # ax_xz.set_aspect('equal', adjustable='box')
    # ax_xz.set_xlabel('X')
    # ax_xz.set_ylabel('Z')
    # ax_xz.set_title(f'XZ Plane at Y={mid_y/Ny:.2f} (t = {time:.3f})')
    # ax_xz.set_xlim(0, Nx)
    # ax_xz.set_ylim(0, Nz)
    # cbar_xz = fig.colorbar(plt.cm.ScalarMappable(norm=normxz, cmap='plasma'), ax=ax_xz, boundaries=np.linspace(0,np.round(np.max(vel_mag_xz),6),5))
    # cbar_xz.set_label('Velocity Magnitude')

    # #2D YZ PLANE
    # ax_yz = fig.add_subplot(2, 2, 4)
    
    # z_2d = np.arange(0, Nz)
    # Y_yz, Z_yz = np.meshgrid(y_2d, z_2d, indexing='ij')
    
    # V_yz = Uplot[mid_x,::, ::]
    # W_yz = Wplot[mid_x,::, ::]
    
    # # Calculate velocity magnitude for colormap
    # vel_mag_yz = np.sqrt(V_yz**2 + W_yz**2)
    # normyz = plt.Normalize(vmin=0, vmax=np.round(np.max(vel_mag_yz),6))
    # colormapyz = plt.cm.plasma(normyz(vel_mag_yz))

    # for i in range(0, Ny):
    #     for j in range(0, Ny):
    #         quiver_yz = ax_yz.quiver(Y_yz, Z_yz, V_yz, W_yz, color=colormapyz[i,j])

    # ax_yz.set_aspect('equal', adjustable='box')
    # ax_yz.set_xlabel('Y')
    # ax_yz.set_ylabel('Z')
    # ax_yz.set_title(f'YZ Plane at Y={mid_y/Ny:.2f} (t = {time:.3f})')
    # ax_yz.set_xlim(0, Ny)
    # ax_yz.set_ylim(0, Nz)
    # cbar_yz = fig.colorbar(plt.cm.ScalarMappable(norm=normyz, cmap='plasma'), ax=ax_yz, boundaries=np.linspace(0,np.round(np.max(vel_mag_yz),6),5))
    # cbar_yz.set_label('Velocity Magnitude')

    plt.suptitle(f'Velocity and Pressure Field at Time = {time:.2f}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'../velocity/graph_time_{int(time*1000)}.png')
    plt.close()
