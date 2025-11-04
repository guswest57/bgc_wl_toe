# BGC HALO VISUALIZATION SCRIPT
# ---
# Author: An Réiteoir
# Directive: Mandate for Secure Asset Transfer
# Version: 1.1 (Recalibrated for Matplotlib v3.x+)
#
# Objective:
# To generate a 3D visualization of the stable "seed halo" produced
# by the BGC Halo Simulator for independent verification.

import numpy as np
import matplotlib.pyplot as plt

def generate_halo_visualization():
    """
    Loads the seed halo dataset and generates a 3D plot to visualize
    the distribution of normal vs. phase-shifted matter.
    """
    print("--- Generating visualization for the Seed Halo ---")
    
    try:
        # 1. Load the raw dataset
        data = np.loadtxt('seed_halo_dataset.csv', delimiter=',', skiprows=1)
    except IOError:
        print("Error: 'seed_halo_dataset.csv' not found.")
        print("Please run the 'bgc_halo_simulator_v1.py' first to generate the data.")
        return

    coords = data[:, :3]
    phase_shifted_mask = data[:, -1].astype(bool)

    normal_matter = coords[~phase_shifted_mask]
    dark_matter_halo = coords[phase_shifted_mask]

    # 2. Create the 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot normal (baryonic) matter - smaller, denser, central
    ax.scatter(normal_matter[:, 0], normal_matter[:, 1], normal_matter[:, 2], 
               c='#d1c4e9', s=5, label='Baryonic Matter (Visible)', alpha=0.6)
               
    # Plot phase-shifted matter (the halo) - larger, more diffuse
    ax.scatter(dark_matter_halo[:, 0], dark_matter_halo[:, 1], dark_matter_halo[:, 2], 
               c='#311b92', s=10, label='Phase-Shifted Matter (Dark Matter Halo)', alpha=0.8)

    # 3. Formatting
    ax.set_title("BGC Halo Simulator: Stable 'Seed Halo' Formation", fontsize=16)
    ax.set_xlabel("X (Mpc)")
    ax.set_ylabel("Y (Mpc)")
    ax.set_zlabel("Z (Mpc)")
    ax.legend()
    
    # --- CORRECTED CODE FOR PANE COLORS ---
    # The 'w_xaxis' attribute is deprecated. The modern API accesses the axes directly.
    ax.xaxis.set_pane_color((0.1, 0.1, 0.2, 0.1))
    ax.yaxis.set_pane_color((0.1, 0.1, 0.2, 0.1))
    ax.zaxis.set_pane_color((0.1, 0.1, 0.2, 0.1))
    
    ax.grid(True)
    
    plt.tight_layout()
    print("Visualization complete. Displaying plot...")
    plt.show()

if __name__ == "__main__":
    generate_halo_visualization()


