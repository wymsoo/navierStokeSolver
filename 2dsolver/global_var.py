global dt, Nx, Ny, dx, dy, Re, rho, G, D, max_ts, H, L, viscosity, epsilon
rho = 1400
viscosity = 10 #viscosity of honey Pa s^-1
Re = rho/viscosity*0.1*0.05 #= 14*0.05 = 0.7
G = 9.81
D = 0.05 # radius: 1cm
L = 0.05 # length: 1m
# Parameters
dt = 0.0001 #vdt ~0.1*0.01 = 0.001
H = 1
max_ts = 10000
Nx = 11
Ny = 11
dx = L / Nx # 0.005
dy = D / Ny
epsilon = 1e-8


