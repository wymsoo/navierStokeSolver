import numpy as np
from global_var import Nx, Ny, Re, D, G ,dx, dy, rho, dt, viscosity

def set_Dirichlet_BC(U, V):

    # MAC staggered grid: axis 0 is west-to-east (x), axis 1 is south-to-north (y).
    # U stores horizontal velocity on vertical faces; V stores vertical velocity on horizontal faces.
    # Each block below adds one ghost layer on the named sides. Ghosts are used by
    # finite differences near boundaries; they are not additional physical cells.

    lenU = U.shape

    # East/west walls: odd reflection of the tangential U component makes
    # the interpolated velocity at each wall zero (no slip).
    Uwest = -U[0,:].reshape((1,lenU[1]))
    Ueast = -U[-1,:].reshape((1,lenU[1]))
    U = np.concatenate((Uwest, U), axis=0)
    U = np.concatenate((U, Ueast), axis=0)

    lenU = U.shape

    # North/south are periodic: copy the opposite edge into each ghost layer.
    Unorth = U[:,0].reshape((lenU[0],1))
    Usouth = U[:,-1].reshape((lenU[0],1))
    U = np.concatenate((Usouth, U), axis=1)
    U = np.concatenate((U, Unorth), axis=1)

    lenV = V.shape

    # U and V are both zero at the east/west walls. For V, the odd
    # reflection enforces no slip for the vertical (tangential) component.
    Veast = -V[-1,:].reshape((1, lenV[1])) 
    Vwest = -V[0,:].reshape((1, lenV[1]))
    V = np.concatenate((Vwest, V), axis=0)
    V = np.concatenate((V, Veast), axis=0)

    lenV = V.shape
    
    # North/south periodicity for V, with one ghost column on each side.
    Vnorth = V[:, 0].reshape((lenV[0], 1))
    Vsouth = V[:, -1].reshape((lenV[0], 1))
    V = np.concatenate((Vsouth, V), axis=1)
    V = np.concatenate((V, Vnorth), axis=1)
    

    return U,V



def set_Dirichlet_BC_lidCavity(U, V):

    lenU = U.shape

    # Apply north and south boundaries for U
    lenU = U.shape
    Unorth = 2*np.ones((lenU[0], 1)) - U[:,-1].reshape((lenU[0],1))  # set the last row of velocity field to be north boundary
    Usouth = -U[:, 0].reshape((lenU[0],1)) # set the first row of velocity field to be south boundary
    U = np.concatenate((Usouth, U), axis=1)
    U = np.concatenate((U, Unorth), axis=1) #u_1/2 = c1

    
    # Apply front and back boundaries for U
    lenU = U.shape
    Ueast = np.zeros((1, lenU[1]))
    Uwest = np.zeros((1, lenU[1]))
    U = np.concatenate((Ueast, U), axis=0)
    U = np.concatenate((U, Uwest), axis=0)

    
    # Same thing for V
    lenV = V.shape
    
    # Apply north and south boundaries for V
    # Vnorth = 2*np.ones((lenV[0], lenV[1], 1)) - V[:, :, -1].reshape((lenV[0],lenV[1],1)) 
    Vnorth= -V[:, -1].reshape((lenV[0],1))
    Vsouth= -V[:, 0].reshape((lenV[0],1))
    V = np.concatenate((Vsouth, V), axis=1)
    V = np.concatenate((V, Vnorth), axis=1)
    
    lenV = V.shape
    # Apply front and back boundaries for V
    Veast = -V[-1, :].reshape((1, lenV[1]))
    Vwest = -V[0, :].reshape((1, lenV[1]))
    V = np.concatenate((Vwest, V), axis=0)
    V = np.concatenate((V, Veast), axis=0)

    
    return U, V

