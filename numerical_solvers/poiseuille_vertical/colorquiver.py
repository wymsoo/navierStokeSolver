import numpy as np 
import matplotlib.pyplot as plt 

def colorquiver(ax, Uplot, Vplot, Wplot, Nx, Ny, Nz, H):
    # Compute the magnitude of the vectors
    mags = np.sqrt(Uplot**2 + Vplot**2 + Wplot**2)

    # Get the current colormap
    current_colormap = plt.get_cmap('jet')

    # Determine the color for each arrow using the colormap
    norm_mags = (mags.flatten() - np.min(mags.flatten())) / (np.max(mags.flatten()) - np.min(mags.flatten())) #normalize the colors
    colors = current_colormap(norm_mags)[:, :3]  # RGB values, excluding alpha

    # Create a new quiver plot
    X, Y, Z = np.meshgrid(np.arange(1, Nx+1), np.arange(1, Ny+1), np.arange(1, Nz+1))
    
    # Generate quiver plot with colors determined by magnitudes
    for i in range(0, Nx, H):
        for j in range(0, Ny, H):
            for k in range(0, Nz, H):
                ax.quiver(X[i, j, k], Y[i, j, k], Z[i, j, k], Uplot[i, j, k], Vplot[i, j, k], Wplot[i, j, k], 
                           color=colors[i * Ny + j], length=1, normalize=True)

    # Adjust plot settings
    ax.set_box_aspect([1, 1, 1])  # Aspect ratio
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.grid(True)
