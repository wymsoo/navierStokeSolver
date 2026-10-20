import numpy as np

def div(U, V, dx, dy):

    # Compute derivatives
    dUdx = np.diff(U, axis=0) / dx
    dVdy = np.diff(V, axis=1) / dy
    # dWdz = np.diff(W, axis=2) / dz

    divergence = dUdx[:,1:-1]+ dVdy[1:-1,:]
    
    return divergence