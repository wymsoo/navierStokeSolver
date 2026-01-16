global dt, Nx, Ny, dx, dy, Re, rho, G, D, timesteps, H, L
Re = 1000 # Reynolds number
rho = 1000
G = 9.81
D = 0.001 # radius
L = 0.001 # length
# Parameters
# converges at around 100
dt = 0.000001
H = 1
timesteps = 120
Nx = 11
Ny = 11
dx = L / Nx
dy = D / Ny


