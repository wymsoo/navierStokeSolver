import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.io import loadmat

# Load and extract data
mat_data = loadmat('2d_stridge_model/burgers_shock.mat')
U = mat_data['usol']  # (256, 100) - spatial points x time steps
x = mat_data['x']     # (256, 1) - spatial coordinates
t = mat_data['t']     # (100, 1) - time coordinates

x_coords = np.asarray(x).flatten()
t_coords = np.asarray(t).flatten()

print(f"Data shapes: x={x_coords.shape}, t={t_coords.shape}, U={U.shape}")
print(f"Spatial domain: [{x_coords.min():.4f}, {x_coords.max():.4f}]")
print(f"Time domain: [{t_coords.min():.4f}, {t_coords.max():.4f}]")
print(f"Velocity range: [{U.min():.4f}, {U.max():.4f}]")

def create_line_plots(x_coords, t_coords, U, num_frames=8, output_dir="line_frames"):
    # Create output directory
    output_path = os.path.join(os.path.dirname(__file__), output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    # Select time indices evenly spaced across the domain
    time_indices = np.linspace(0, U.shape[1] - 1, num_frames, dtype=int)
    
    # Get consistent y-axis limits
    v_min, v_max = U.min(), U.max()
    
    for frame_idx, t_idx in enumerate(time_indices):
        current_time = t_coords[t_idx]
        u_velocities = U[:, t_idx]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot velocity profile
        ax.plot(x_coords, u_velocities, linewidth=1.5, color='steelblue')
        
        # Set consistent axis limits
        ax.set_xlim([x_coords.min() - 0.05, x_coords.max() + 0.05])
        ax.set_ylim([v_min - 0.1 * (v_max - v_min), v_max + 0.1 * (v_max - v_min)])
        
        ax.set_xlabel('Spatial Position (x)', fontsize=12)
        ax.set_ylabel('Velocity (u)', fontsize=12)
        ax.set_title(f'Burgers Equation - Velocity Profile at t = {current_time:.4f}', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Save the figure
        output_filename = f"burgers_line_t{frame_idx:03d}_t{current_time:.4f}.png"
        output_filepath = os.path.join(output_path, output_filename)
        plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_filepath}")
        
        plt.close(fig)
    
    print(f"\nAll frames saved to: {output_path}")


if __name__ == "__main__":
    # Also create line plots as an alternative visualization
    print("\n=== Generating Line Profile Plots ===")
    create_line_plots(x_coords, t_coords, U, num_frames=8, output_dir="line_frames")
