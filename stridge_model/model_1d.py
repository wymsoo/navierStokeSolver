import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from global_var import dt, Nx, Ny, dx, dy, Re, rho, G, D, max_ts, H, L, viscosity, epsilon, nu
import sys; sys.path.append('../')
import scipy.io as sio
import itertools
import os
import argparse
import linecache
import matplotlib.pyplot as plt
from format_report import write_markdown_report, format_equation

u_max = 1.0

def count_files_scandir(directory):
    count = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file():
                count += 1
    return count


def gradient(u, dt, dy):
    ut = np.gradient(u, dt, axis=0, edge_order=2)
    uy = np.gradient(u, dy, axis=1, edge_order=2)
    uyy = np.gradient(uy, dy, axis=1, edge_order=2)
    return ut, uy, uyy

# Construct Operator Dictionary
import itertools

def build_dict(u, uy, u_yy):

    all_differentials = [u, uy, u_yy]
    all_labels = ['u','u_y','u_yy']

    # Dictionary: for display
    dictionary = {}
    combinations = itertools.combinations(all_differentials, 2)
    label_combinations = itertools.combinations(all_labels, 2)

    # compute all product of combinations
    oper_dict = [(a*b).reshape(-1) for a, b in combinations]


    # Use string keys for combined labels (e.g. 'u*u_y') instead of tuples
    for label, oper in zip(label_combinations, oper_dict):
        if isinstance(label, tuple):
            key = f"{label[0]}*{label[1]}"
        else:
            key = label
        dictionary[key] = oper

    # add back the single terms
    for diff, label in zip(all_differentials, all_labels):
        oper_dict.append(diff.reshape(-1))
        dictionary[label] = diff

    dictionary['1'] = np.ones_like(u)
    # dictionary['u**2'] = (u**2).reshape(-1)
    oper_dict.append(np.ones_like(u).reshape(-1))
    # oper_dict.append((u**2).reshape(-1))

    dictionary_keys = []
    for key in dictionary:
        dictionary_keys.append(key)


    # oper_dict = np.transpose(np.array(oper_dict),(1,0,2,3))
    oper_dict = np.transpose(np.array(oper_dict))
    return oper_dict, dictionary_keys



def stridge1(X, Y, tol, lam, max_iter = 100):
    d = X.shape[1]
    biginds = np.arange(d)  # initialise array for big indices
    biginds_prev = biginds

    #normalise theta
    norm = np.linalg.norm(X, axis=0)
    norm[norm<1e-16] = 1.0
    X_norm = X/norm

    # Calculate initial weights
    weights = np.linalg.lstsq(X_norm.T @ X_norm + lam * np.eye(d), X_norm.T @ Y, rcond=None)[0]
    # print("weights:",weights)

    if biginds.size == 0:
        return weights

    for iter in range(max_iter):
        biginds = np.where(abs(weights) >= tol)[0]
        # If the weights do not change, break the loop
        if np.array_equal(biginds, biginds_prev):
            break
        if biginds.size==0:
            weights[:]=0
            break
        biginds_prev = biginds.copy()
        weights[:] = 0
        weights[biginds] = np.linalg.lstsq(
            X_norm[:, biginds].T.dot(X[:, biginds]) + lam * np.eye(len(biginds)),
            X_norm[:, biginds].T.dot(Y),
            rcond=None
        )[0]

    biginds = np.where(abs(weights)>= tol)[0]
    if biginds.size > 0:
        weights[:] = 0.0
        weights[biginds] = np.linalg.lstsq(X_norm[:, biginds], Y, rcond=None)[0]
    else:
        weights[:] = 0.0

    return np.array(weights/norm)

def trainStridge(X, y, tol_values, lam, l0_penalty, seed=0, print_best_tol=False):
    rng = np.random.default_rng(seed)

    n_samples = X.shape[0]
    train_size = int(n_samples * 0.8)
    X_train, X_validate = X[:train_size], X[train_size:]
    y_train, y_validate = y[:train_size], y[train_size:]

    # n = X.shape[0]
    # rand_indices = rng.permutation(n)
    # X_train = X[rand_indices[:int(n*0.8)]]
    # X_validate = X[rand_indices[int(n*0.8):]]
    # y_train = y[rand_indices[:int(n*0.8)]]
    # y_validate = y[rand_indices[int(n*0.8):]]
    score_best = np.inf
    w_best = np.linalg.lstsq(X, y, rcond=None)[0]
    train_err_best = np.linalg.norm(y_train - X_train @ w_best, 2) + l0_penalty * np.count_nonzero(w_best)
    val_err_best = np.linalg.norm(y_validate - X_validate @ w_best, 2) + l0_penalty * np.count_nonzero(w_best)
    tol_best = 0

    for tolerance in tol_values:
        weights = stridge1(X_train, y_train, tolerance, args.lam)
        train_err = np.linalg.norm(y_train - X_train @ weights, 2) / np.sqrt(y_train.size)
        val_err = np.linalg.norm(y_validate - X_validate @ weights, 2)/np.sqrt(y_validate.size)
        score = val_err + l0_penalty * np.count_nonzero(np.abs(weights) > 1e-12)
        if score <= score_best:
            # Improved: keep direction
            # print("error improved", train_err)
            score_best = score
            w_best = weights
            train_err_best = train_err
            val_err_best = val_err
            tol_best = tolerance
        if val_err <= 10**-8:
            break


    # if print_best_tol:
        # print("Optimal tolerance:", tol_best)

    return w_best, tol_best, train_err_best, val_err_best


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
            term = f"{mag:.8e}*one"
        else:
            term = f"{mag:.8e}*{name}"
        if i == 0:
            pieces.append(term if c >= 0 else f"- {term}")
        else:
            pieces.append(f" {sign} {term}")
    return f"{lhs} =" + "".join(pieces)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Learn the startup Poiseuille PDE from saved data.")
    # parser.add_argument("--input", type=str, default=os.path.join(os.path.dirname(__file__), "startup_poiseuille_data.npz"))
    parser.add_argument("--lam", type=float, default=1e-9)
    parser.add_argument("--tol_min", type=float, default=1e-6)
    parser.add_argument("--tol_max", type=float, default=1)
    parser.add_argument("--num_tol", type=int, default=40)
    parser.add_argument("--l0_penalty", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--crop_t", type=int, default=800, help="discard this many time layers near each temporal boundary")
    parser.add_argument("--crop_y", type=int, default=2, help="discard this many spatial points near each wall")
    parser.add_argument(
        "--report",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "stridge_model_1d_report.md"),
        help="path for the generated Markdown report",
    )
    args = parser.parse_args()

    nu_true = nu
    g_true = G

    file_path = os.path.join(os.path.dirname(__file__), "output_2movingwalls", "u_velocity_field")

    time_len = count_files_scandir(file_path)
    results = []
    line_to_read = 15  # mid line of the file
    for i in range(150, time_len + 1):
        file_name = os.path.join(file_path, f"u_velocity_t={i}.txt")
        with open(file_name, "r") as file:
            lines = file.readlines()
            if line_to_read <= len(lines):
                line = lines[line_to_read - 1].strip()
                if line:
                    # Each entry is separated by whitespace; convert the tokens, not the characters.
                    floats = [float(value) for value in line.split()]
                    results.append(floats)

    U = np.array(results)
    print(U.shape)




    ut, uy, uyy = gradient(U, dt, dy)
    plt.plot(np.arange(len(uy[1000])),uy[1000,:])
    plt.legend()
    plt.show()
    t_slice = slice(args.crop_t, U.shape[0] - args.crop_t)
    y_slice = slice(args.crop_y, U.shape[1] - args.crop_y)

    u_crop = U[t_slice, y_slice]
    ut_crop = ut[t_slice, y_slice]
    uy_crop = uy[t_slice, y_slice]
    uyy_crop = uyy[t_slice, y_slice]

    oper_dict, dictionary = build_dict(u_crop,uy_crop,uyy_crop)

    x = oper_dict[:]
    y = ut_crop.reshape(-1)

    tol_values = np.logspace(np.log10(args.tol_min), np.log10(args.tol_max), args.num_tol)
    w_best, tol_best, err_best, val_error = trainStridge(x,y,tol_values,args.lam, args.l0_penalty)

    print(dictionary)

    print("=" * 72)
    print("Discovered equation")
    print(format_equation(w_best, dictionary, lhs="u_t"))
    print("=" * 72)
    print(f"Best tolerance      : {tol_best:.8e}")
    print(f"Train RMSE          : {err_best:.8e}")
    print(f"Validation RMSE     : {val_error:.8e}")
    print(f"True equation       : u_t = {g_true} + {nu_true}*u_yy")
    print("Estimated coefficients")
    for c, name in zip(w_best, dictionary):
        print(f"  {name:8s} : {c:+.8e}")

    write_markdown_report(
        args.report,
        w_best,
        dictionary,
        U.shape,
        u_crop.shape,
        tol_best,
        err_best,
        val_error,
        data_start=150,
        crop_settings={"t": args.crop_t, "x": 0, "y": args.crop_y},
        lam=args.lam,
        l0_penalty=args.l0_penalty,
        parameters={
            "rho": (rho, "kg m$^{-3}$", "Fluid density"),
            "viscosity": (viscosity, "Pa s", "Dynamic viscosity"),
            "nu": (nu, "m$^2$ s$^{-1}$", "Kinematic viscosity"),
            "u_max": (u_max, "m s$^{-1}$", "Characteristic velocity"),
            "D": (D, "m", "Characteristic length"),
            "Re": (Re, "dimensionless", "Reynolds number"),
            "G": (G, "m s$^{-2}$", "Body force"),
            "Nx": (Nx, "points", "Grid points"),
            "Ny": (Ny, "points", "Unused transverse grid points"),
            "dy": (dy, "m", "Grid spacing"),
            "dt": (dt, "s", "Time step"),
            "epsilon": (epsilon, "-", "Numerical threshold"),
        },
        reference_coefficients={"1": g_true, "u_yy": nu_true},
        input_directory=file_path,
        data_shape_labels="(t, y)",
    )
    print(f"Markdown report written to: {args.report}")



