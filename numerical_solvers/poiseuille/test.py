from scipy.sparse import diags, eye, kron, csr_matrix
import numpy as np
from global_var import rho, viscosity, G, D, Ny

A = np.array([[1,1,1],
     [1,1,1],
     [1,1,1],
     [1,2,1]])

B = np.array([[1],
     [2],
     [3],
     [1]])

zero = np.zeros((A.shape[0],1))
y = np.linspace(-D/2,D/2,Ny)
V_theory = -1/viscosity/4*rho*G*(y**2-(D/2)**2)
print(V_theory)



# mass = 0.002
# ball_diam = 0.04
# rho = 1.225
# g = 9.81
# l = 2.74
# w = 1.52
# delta_h = 0.05
# h = 0.16 + delta_h
# d_cross = np.sqrt(l**2+w**2) # 3.13
# u_km_h_x = 80
# u_x = u_km_h_x/3600*1000 # 16.6
# t = d_cross/u_x #0.18
# a_y = 8*h/t**2
# u_y = u_x*a_y/2/d_cross
# a_magnus = a_y - g
# spin = 3*a_magnus/(16*np.pi**2*ball_diam**3*rho*u_x)
# spin_rps = spin/2/np.pi
# print(spin_rps)



