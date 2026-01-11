from scipy.sparse import diags, eye, kron, csr_matrix
import numpy as np


# Grid size
Nx = 4
Ny = 4
dx = 1.0 / Nx
dy = 1.0 / Ny

dt = 0.01
Re = 1000 # Reynolds number
rho = 900
G = 9.81

# Velocity fields (staggered)
U = np.zeros((Nx - 1, Ny))      # u-velocity at x-faces
V = np.zeros((Nx, Ny - 1))      # v-velocity at y-faces

# Number of iterations

# Apply front and back boundaries for U
lenU = U.shape
# Unorth = 2*np.ones((lenU[0], 1)) - U[:, -1].reshape((lenU[0],1)) 
# Unorth = -U[:, :, -1].reshape((lenU[0],lenU[1],1))  
Unorth = -U[:,-1].reshape((lenU[0],1))  # for poiseuille flow, there is no slip in the top and bottom boundary
Usouth = -U[:,0].reshape((lenU[0],1))
U = np.concatenate((Usouth, U), axis=1)
U = np.concatenate((U, Unorth), axis=1)

print("1:",U)

midU = Ny//2
lenU = U.shape
# Ueast = np.zeros((1, lenU[1])) # set the last row of velocity field to be north boundary
# Uwest = np.zeros((1, lenU[1])) # set the first row of velocity field to be south boundary
Uwest = np.zeros((1, lenU[1])) + G*dt #periodic boundary conditions
Ueast = U[-1,:].reshape((1, lenU[1])) + G*dt #neumann boundary condition: boundary = previous last cell
U = np.concatenate((Uwest, U), axis=0)
U = np.concatenate((U, Ueast), axis=0) #u_1/2 = c1

print("2U:",U)


U[:,midU+1] = U[:,midU]
U[:,midU-1] = U[:,midU] # dUdx at middle = 0

# Same thing for V
lenV = V.shape

# Apply north and south boundaries for V
# Vnorth = 2*np.ones((lenV[0], lenV[1], 1)) - V[:, :, -1].reshape((lenV[0],lenV[1],1)) 
Vnorth= np.zeros((lenV[0], 1))
Vsouth= np.zeros((lenV[0], 1))
V = np.concatenate((Vsouth, V), axis=1)
V = np.concatenate((V, Vnorth), axis=1)
print("V",V)

lenV = V.shape
# Apply front and back boundaries for V
Veast = 2*V[-2,:]-V[-1,:].reshape((1, lenV[1])) #boundary at the end of V-field
Vwest = -V[0,:].reshape((1, lenV[1]))
V = np.concatenate((Vwest, V), axis=0)
V = np.concatenate((V, Veast), axis=0)

print("V2", V)