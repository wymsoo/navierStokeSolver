import argparse
import numpy as np


def simulate_startup_poiseuille(
    H=1.0,
    Lx=1.0,
    Ny=129,
    Nx=16,
    dt=5e-4,
    t_end=1.0,
    nu=0.05,
    rho=1.0,
    dpdx=-1.0,
    noise_level=0.0,
    seed=0,
):
    """
    Solve startup plane Poiseuille flow in a channel.

    Governing equation:
        u_t = nu * u_yy - (1/rho) * dp/dx
    with no-slip walls:
        u(t,0) = u(t,H) = 0,
    and initial condition:
        u(0,y) = 0.

    Parameters
    ----------
    H : float
        Channel height.
    Lx : float
        Dummy streamwise length used only to build a repeated x-grid.
    Ny : int
        Number of collocated points in y, including the two walls.
    Nx : int
        Number of repeated x-points. The physics is 1D in y, but we also save
        a repeated 2D field u(t, x, y) for convenience.
    dt : float
        Time step.
    t_end : float
        Final time.
    nu : float
        Kinematic viscosity.
    rho : float
        Density.
    dpdx : float
        Constant pressure gradient along x.
        For pressure-driven flow from left to right, dpdx < 0.
    noise_level : float
        Relative Gaussian noise level, measured against the steady-state max velocity.
    seed : int
        Random seed for reproducibility.
    """
    if Ny < 5:
        raise ValueError("Ny must be at least 5.")

    y = np.linspace(0.0, H, Ny)
    x = np.linspace(0.0, Lx, Nx)
    dy = y[1] - y[0]

    Nt = int(np.round(t_end / dt)) + 1
    t = np.linspace(0.0, dt * (Nt - 1), Nt)

    g = -(1.0 / rho) * dpdx  # constant body-force equivalent of the pressure gradient

    # Backward-Euler step for the interior unknowns.
    n_int = Ny - 2
    main = -2.0 * np.ones(n_int)
    off = np.ones(n_int - 1)
    D2 = (np.diag(main) + np.diag(off, 1) + np.diag(off, -1)) / (dy ** 2)

    A = np.eye(n_int) - dt * nu * D2
    rhs_const = dt * g * np.ones(n_int)

    u = np.zeros((Nt, Ny), dtype=float)

    for n in range(Nt - 1):
        rhs = u[n, 1:-1] + rhs_const
        u[n + 1, 1:-1] = np.linalg.solve(A, rhs)

    # Optional measurement noise.
    umax_steady = g * H ** 2 / (8.0 * nu)
    if noise_level > 0.0:
        rng = np.random.default_rng(seed)
        sigma = noise_level * max(abs(umax_steady), 1e-12)
        u[:, 1:-1] += rng.normal(0.0, sigma, size=u[:, 1:-1].shape)
        u[:, 0] = 0.0
        u[:, -1] = 0.0

    # Repeated 2D field for later extension to u(t, x, y).
    u2d = np.repeat(u[:, None, :], Nx, axis=1)

    # Steady analytical profile for reference.
    u_steady = g / (2.0 * nu) * y * (H - y)

    return {
        "u": u,
        "u2d": u2d,
        "t": t,
        "x": x,
        "y": y,
        "dt": dt,
        "dy": dy,
        "dx": x[1] - x[0] if Nx > 1 else Lx,
        "nu": nu,
        "rho": rho,
        "dpdx": dpdx,
        "g": g,
        "H": H,
        "Lx": Lx,
        "u_steady": u_steady,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate startup Poiseuille flow data.")
    parser.add_argument("--output", type=str, default="startup_poiseuille_data.npz")
    parser.add_argument("--H", type=float, default=1.0)
    parser.add_argument("--Lx", type=float, default=1.0)
    parser.add_argument("--Ny", type=int, default=129)
    parser.add_argument("--Nx", type=int, default=16)
    parser.add_argument("--dt", type=float, default=5e-4)
    parser.add_argument("--t_end", type=float, default=1.0)
    parser.add_argument("--nu", type=float, default=0.05)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--dpdx", type=float, default=-1.0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = simulate_startup_poiseuille(
        H=args.H,
        Lx=args.Lx,
        Ny=args.Ny,
        Nx=args.Nx,
        dt=args.dt,
        t_end=args.t_end,
        nu=args.nu,
        rho=args.rho,
        dpdx=args.dpdx,
        noise_level=args.noise,
        seed=args.seed,
    )

    np.savez(args.output, **data)
    print(f"Saved data to {args.output}")
    print(f"u shape   : {data['u'].shape}   (Nt, Ny)")
    print(f"u2d shape : {data['u2d'].shape} (Nt, Nx, Ny)")
    print(f"True PDE  : u_t = {data['g']:.8f} + {data['nu']:.8f} * u_yy")
