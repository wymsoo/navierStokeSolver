import numpy as np

def set_Dirichlet_BC(U, V, u_wall=1.0):
    """
    Applies boundary conditions for a 4-lid moving cavity (Counter-Clockwise):
      - South Wall (Bottom) : Moves RIGHT  (U_south_wall = +u_wall, V_south = 0)
      - North Wall (Top)    : Moves LEFT   (U_north_wall = -u_wall, V_north = 0)
      - East Wall  (Right)  : Moves UP     (V_east_wall  = +u_wall, U_east  = 0)
      - West Wall  (Left)   : Moves DOWN   (V_west_wall  = -u_wall, U_west  = 0)
    """

    # =========================================================
    # 1. HORIZONTAL VELOCITY (U)
    # =========================================================
    # --- Top/Bottom Boundaries (Tangential Shear) ---
    # Linear interpolation across wall: (U_ghost + U_interior)/2 = U_wall
    lenU = U.shape
    Unorth = -2.0 * u_wall - U[:, -1].reshape((lenU[0], 1))  # Moving Left (-u_wall)
    Usouth =  2.0 * u_wall - U[:,  0].reshape((lenU[0], 1))  # Moving Right (+u_wall)
    
    U_padded = np.concatenate((Usouth, U, Unorth), axis=1)

    # --- Left/Right Boundaries (Normal Impermeable Walls) ---
    # U faces sit directly ON the East/West walls -> U_wall = 0
    lenU_p = U_padded.shape
    Uwest = np.zeros((1, lenU_p[1]))
    Ueast = np.zeros((1, lenU_p[1]))
    
    U_padded = np.concatenate((Uwest, U_padded, Ueast), axis=0)


    # =========================================================
    # 2. VERTICAL VELOCITY (V)
    # =========================================================
    # --- Top/Bottom Boundaries (Normal Impermeable Walls) ---
    # V faces sit directly ON the North/South walls -> V_wall = 0
    lenV = V.shape
    Vsouth = np.zeros((lenV[0], 1))
    Vnorth = np.zeros((lenV[0], 1))
    
    V_padded = np.concatenate((Vsouth, V, Vnorth), axis=1)

    # --- Left/Right Boundaries (Tangential Shear) ---
    # Linear interpolation across wall: (V_ghost + V_interior)/2 = V_wall
    lenV_p = V_padded.shape
    Vwest = -2.0 * u_wall - V_padded[ 0, :].reshape((1, lenV_p[1])) # Moving Down (-u_wall)
    Veast =  2.0 * u_wall - V_padded[-1, :].reshape((1, lenV_p[1])) # Moving Up (+u_wall)
    
    V_padded = np.concatenate((Vwest, V_padded, Veast), axis=0)

    return U_padded, V_padded