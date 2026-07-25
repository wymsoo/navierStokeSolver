import argparse
import numpy as np

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import sys; sys.path.append('../')
from global_var import Nx, Ny, Re, D, G ,dx, dy, dt, max_ts, H, L, rho, viscosity, epsilon, endtime, time_res
import os


class STRidgeResult:
    def __init__(self, coefficients, descriptions, best_tol, train_error, val_error):
        self.coefficients = coefficients
        self.descriptions = descriptions
        self.best_tol = best_tol
        self.train_error = train_error
        self.val_error = val_error


def compute_derivatives(u, dt, dy):
    ut = np.gradient(u, dt, axis=0, edge_order=2)
    uy = np.gradient(u, dy, axis=1, edge_order=2)
    uyy = np.gradient(uy, dy, axis=1, edge_order=2)
    return ut, uy, uyy


def build_library(u, uy, uyy):
    terms = [
        np.ones_like(u),
        u,
        u ** 2,
        u ** 3,
        uy,
        uyy,
        u * uy,
        u * uyy,
    ]
    descriptions = [
        "1",
        "u",
        "u^2",
        "u^3",
        "u_y",
        "u_yy",
        "u*u_y",
        "u*u_yy",
    ]
    Theta = np.column_stack([term.reshape(-1) for term in terms])
    return Theta, descriptions


def normalize_columns(Theta):
    norms = np.linalg.norm(Theta, axis=0)
    norms[norms < 1e-14] = 1.0
    Theta_norm = Theta / norms
    return Theta_norm, norms


def stridge(Theta, y, lam=1e-8, tol=1e-6, maxit=100):
    """
    Sequential Threshold Ridge Regression.
    Thresholding is applied to the coefficients of the column-normalized matrix.
    The returned coefficients are converted back to the original scaling.
    """
    Theta_n, norms = normalize_columns(Theta)
    d = Theta_n.shape[1]

    w = np.linalg.lstsq(Theta_n.T @ Theta_n + lam * np.eye(d), Theta_n.T @ y, rcond=None)[0]
    support_prev = np.arange(d)

    # recalculate weights until the support does not change or no more weights
    for _ in range(maxit):
        support = np.where(np.abs(w) >= tol)[0] # all the weights with weights larger than tolerance
        if support.size == 0: 
            w[:] = 0.0
            break
        if np.array_equal(support, support_prev): # number of weights do not change
            break
        support_prev = support.copy() # save support
        w[:] = 0.0 # set weights to zero
        w[support] = np.linalg.lstsq(
            Theta_n[:, support].T @ Theta_n[:, support] + lam * np.eye(support.size),
            Theta_n[:, support].T @ y,
            rcond=None,
        )[0] # calculate new weights with supports

    support = np.where(np.abs(w) >= tol)[0] # 
    if support.size > 0:
        w[:] = 0.0
        w[support] = np.linalg.lstsq(Theta_n[:, support], y, rcond=None)[0]
    else:
        w[:] = 0.0

    return w / norms


def train_stridge(Theta, y, tol_values, lam=1e-8, l0_penalty=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    n = Theta.shape[0]
    idx = rng.permutation(n)
    n_train = int(0.8 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    Theta_train = Theta[train_idx]
    y_train = y[train_idx]
    Theta_val = Theta[val_idx]
    y_val = y[val_idx]

    best_score = np.inf
    best_coef = None
    best_tol = None
    best_train_error = None
    best_val_error = None

    for tol in tol_values:
        coef = stridge(Theta_train, y_train, lam=lam, tol=tol)
        train_error = np.linalg.norm(y_train - Theta_train @ coef, 2) / np.sqrt(y_train.size)
        val_error = np.linalg.norm(y_val - Theta_val @ coef, 2) / np.sqrt(y_val.size)
        score = val_error + l0_penalty * np.count_nonzero(np.abs(coef) > 1e-12)
        if score < best_score:
            best_score = score
            best_coef = coef.copy()
            best_tol = tol
            best_train_error = train_error
            best_val_error = val_error

    return best_coef, best_tol, best_train_error, best_val_error


def format_equation(coefficients, descriptions, lhs="u_t", coef_cutoff=1e-10):
    # save active coeficcients as tuples in an array
    active = []
    for c, name in zip(coefficients, descriptions):
        if abs(c) > coef_cutoff:
            active.append((c, name))
    if not active:
        return f"{lhs} = 0"

    pieces = []
    for i, (c, name) in enumerate(active):
        sign = "-" if c < 0 else "+"
        mag = abs(c)
        if name == "1":
            term = f"{mag:.8e}"
        else:
            term = f"{mag:.8e}*{name}"
        if i == 0:
            pieces.append(term if c >= 0 else f"- {term}")
        else:
            pieces.append(f" {sign} {term}")
    return f"{lhs} =" + "".join(pieces)

def count_files_scandir(directory):
    count = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file():
                count += 1
    return count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learn the startup Poiseuille PDE from saved data.")
    parser.add_argument("--input", type=str, default=os.path.join(os.path.dirname(__file__), "startup_poiseuille_data.npz"))
    parser.add_argument("--lam", type=float, default=1e-8)
    parser.add_argument("--tol_min", type=float, default=1e-8)
    parser.add_argument("--tol_max", type=float, default=1e-1)
    parser.add_argument("--num_tol", type=int, default=40)
    parser.add_argument("--l0_penalty", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--crop_t", type=int, default=1, help="discard this many time layers near each temporal boundary")
    parser.add_argument("--crop_y", type=int, default=2, help="discard this many spatial points near each wall")
    args = parser.parse_args()




    # time_len = count_files_scandir('output/u_velocity_field')
    # t = np.arange(0,endtime,dt)
    # x_size = 31
    # y_size = 31
    # y = np.linspace(-D/2,D/2,Ny)
    # x = np.linspace(0,D,Nx)
    # V = np.zeros((time_len,x_size,y_size-1)) #(time,x,y-1)
    # P = np.zeros((time_len,x_size,y_size)) #(time,x,y)
    # u = np.zeros((time_len,x_size-1,y_size)) #(time,x-1,y)
    # starttime = 1
    # for i in range(time_len-1):
    #     u[i] = np.loadtxt(f"output_startup/u_velocity_field/u_velocity_t={starttime}.txt")
    #     V[i] = np.loadtxt(f"output_startup/v_velocity_field/v_velocity_t={starttime}.txt")
    #     P[i] = np.loadtxt(f"output_startup/pressure_field/pressure_field_t={starttime}.txt")
    #     starttime+=1



    data = np.load(args.input)
    u = data["u2d"]
    print(u.shape)
    t = data["t"]
    y = data["y"]
    dt = float(data["dt"])
    dy = float(data["dy"])
    nu_true = float(data["nu"])
    g_true = float(data["g"])


    ut, uy, uyy = compute_derivatives(u, dt, dy)

    

    t_slice = slice(args.crop_t, u.shape[0] - args.crop_t)
    y_slice = slice(args.crop_y, u.shape[1] - args.crop_y)

    u_crop = u[t_slice, y_slice]
    ut_crop = ut[t_slice, y_slice]
    uy_crop = uy[t_slice, y_slice]
    uyy_crop = uyy[t_slice, y_slice]

    Theta, descriptions = build_library(u_crop, uy_crop, uyy_crop)
    y_target = ut_crop.reshape(-1)

    tol_values = np.logspace(np.log10(args.tol_min), np.log10(args.tol_max), args.num_tol)
    coef, best_tol, train_error, val_error = train_stridge(
        Theta,
        y_target,
        tol_values=tol_values,
        lam=args.lam,
        l0_penalty=args.l0_penalty,
        seed=args.seed,
    )

    print(descriptions)

    # print("=" * 72)
    # print("Discovered equation")
    # print(format_equation(coef, descriptions, lhs="u_t"))
    # print("=" * 72)
    # print(f"Best tolerance      : {best_tol:.8e}")
    # print(f"Train RMSE          : {train_error:.8e}")
    # print(f"Validation RMSE     : {val_error:.8e}")
    # print(f"True equation       : u_t = {g_true:.8e} + {nu_true:.8e}*u_yy")
    # print("Estimated coefficients")
    # for c, name in zip(coef, descriptions):
    #     print(f"  {name:8s} : {c:+.8e}")
