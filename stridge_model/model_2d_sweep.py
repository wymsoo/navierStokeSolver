import numpy as np
from format_report import write_2d_markdown_report, format_equation
import sys; sys.path.append('../')
import itertools
import os
import argparse


dt = 0.0001
epsilon = 7e-11
# Grid size
Nx = 31
Ny = 31
dx = 1.0 / Nx
dy = 1.0 / Ny
G = 9.81
rho = 1000
u_max = 1
v_max = 1
# D = 0.005
D = 1.0
viscosity = 10
Re = (rho * u_max * D) / viscosity
nu = viscosity/rho
c_advective = 1
c_pressure = 1/rho
c_viscous = 1/Re





def count_files_scandir(directory):
    count = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file():
                count += 1
    return count


def load_case_fields(output_directory, data_start):
    field_directories = {
        "u": os.path.join(output_directory, "u_velocity_field"),
        "v": os.path.join(output_directory, "v_velocity_field"),
        "p": os.path.join(output_directory, "pressure_field"),
    }
    time_len = count_files_scandir(field_directories["u"])
    snapshot_indices = range(data_start, time_len + 1)

    fields = {}
    for field_name, directory in field_directories.items():
        if field_name == "p":
            filename = lambda index: f"pressure_field_t={index}.txt"
        else:
            filename = lambda index, name=field_name: f"{name}_velocity_t={index}.txt"
        fields[field_name] = np.array([
            np.loadtxt(os.path.join(directory, filename(i)))
            for i in snapshot_indices
        ])

    return fields


def load_case(output_directory, data_start, crop_t, crop_x, crop_y, fields=None):
    if fields is None:
        fields = load_case_fields(output_directory, data_start)

    U, V, P = fields["u"][:,:,:-1], fields["v"][:,:-1,:], fields["p"][:,:-1,:-1]
    print(U.shape, V.shape, P.shape)


    ut, uy, uyy = gradient(U, dt, dy, axis=2)
    _, ux, uxx = gradient(U, dt, dx, axis=1)
    vt, vy, vyy = gradient(V, dt, dy, axis=2)
    _, vx, vxx = gradient(V, dt, dx, axis=1)
    _, px, _ = gradient(P, dt, dx, axis=1)
    _, py, _ = gradient(P, dt, dy, axis=2)


    t_slice = slice(crop_t, U.shape[0])
    x_slice = slice(crop_x, U.shape[1])
    y_slice = slice(crop_y, U.shape[2])

    u_crop = U[t_slice, x_slice, y_slice].copy()
    v_crop = V[t_slice, x_slice, y_slice].copy()

    # compute operator matrix and dictionary keys
    oper_dict, dictionary = build_ns_dict(
        # u_crop[2:-2,4:-4,4:-4], # cropped to fit gradient
        # v_crop[2:-2,4:-4,4:-4],
        u_crop,
        v_crop,
        uy[t_slice, x_slice, y_slice],
        uyy[t_slice, x_slice, y_slice],
        ux[t_slice, x_slice, y_slice],
        uxx[t_slice, x_slice, y_slice],
        px[t_slice, x_slice, y_slice],
    )
    # compute operator matrix and dictionary keys
    oper_dict_v, dictionary_v = build_v_ns_dict(
        # u_crop[2:-2,4:-4,4:-4],
        # v_crop[2:-2,4:-4,4:-4],
        u_crop,
        v_crop,
        vx[t_slice, x_slice, y_slice],
        vxx[t_slice, x_slice, y_slice],
        vy[t_slice, x_slice, y_slice],
        vyy[t_slice, x_slice, y_slice],
        py[t_slice, x_slice, y_slice],
    )

    return (
        oper_dict, # u operator array
        ut[t_slice, x_slice, y_slice].reshape(-1),
        dictionary, # dictionary keys
        oper_dict_v, # v operator array
        vt[t_slice, x_slice, y_slice].reshape(-1),
        dictionary_v, # v dictionary keys
        U.shape,
        u_crop.shape,
    )


def identify_model(input_directories, data_start, crop_t, crop_x, crop_y,
                tol_values, lam, l0_penalty, seed=0, fields_cache=None):
    fields_cache = fields_cache if fields_cache is not None else {}
    case_results = []
    skipped_directories = []
    for output_directory in input_directories:
        if output_directory not in fields_cache: # add field to cache dictionary for reuse later
            fields_cache[output_directory] = load_case_fields(
                output_directory, data_start
            )
        case_result = load_case( # compute dictionary
            output_directory, data_start, crop_t, crop_x, crop_y,
            fields=fields_cache[output_directory],
        )
        if case_result[0].shape[0] == 0: # cropped too much, does not exist
            skipped_directories.append(output_directory)
        else:
            case_results.append(case_result) # append to new operator matrix

    if not case_results:
        raise ValueError(f"The crop_t={crop_t} leaves no regression samples.")

    dictionary = case_results[0][2] # first case, 3rd returned item which is dictionary keys
    if any(result[2] != dictionary for result in case_results[1:]): # for all other results, see if each of the third ditem (the dict keys) is different from the first
        raise ValueError("All cases must use the same U dictionary ordering")
    x = np.concatenate([result[0] for result in case_results]) # for each appended u operator
    y = np.concatenate([result[1] for result in case_results]) # ut
    w_best, tol_best, train_error, validation_error = trainStridge(
        x, y, tol_values, lam, l0_penalty, seed=seed
    )

    dictionary_v = case_results[0][5]
    if any(result[5] != dictionary_v for result in case_results[1:]):
        raise ValueError("All cases must use the same V dictionary ordering")
    x_v = np.concatenate([result[3] for result in case_results])# for each appended v operator
    y_v = np.concatenate([result[4] for result in case_results])# vt
    w_best_v, tol_best_v, train_error_v, validation_error_v = trainStridge(
        x_v, y_v, tol_values, lam, l0_penalty, seed=seed
    )

    return {
        "u_coefficients": w_best, "u_descriptions": dictionary,
        "v_coefficients": w_best_v, "v_descriptions": dictionary_v,
        "data_shape": case_results[0][6], "crop_shape": case_results[0][7],
        "u_tol": tol_best, "u_train_error": train_error,
        "u_validation_error": validation_error,
        "v_tol": tol_best_v, "v_train_error": train_error_v,
        "v_validation_error": validation_error_v,
        "sample_count": x.shape[0],
        "skipped_directories": skipped_directories,
    }


def write_sweep_report(path, results, input_directories, data_start,
                       lam, l0_penalty):
    references = {
        "u": {"uu_x": -c_advective, "vu_y": -c_advective,
              "u_xx": c_viscous, "u_yy": c_viscous, "p_x": -c_pressure},
        "v": {"uv_x": -c_advective, "vv_y": -c_advective,
              "v_xx": c_viscous, "v_yy": c_viscous, "p_y": -c_pressure},
    }
    lines = [
        "# STRidge Crop Sweep Parameter Errors", "",
        "Percentage differences between the reference and learned parameters.", "",
        f"- Input directories: `{', '.join(input_directories)}`",
        f"- First saved time index: `{data_start}`",
        f"- lambda: `{lam:.8e}`; L0 penalty: `{l0_penalty:.8e}`", "",
        "## Average Percentage Differences by Case", "",
        "The `u` and `v` columns average the five coefficient percentage differences for each equation. The case average includes all ten coefficients.", "",
        "| Case | t crop | x crop | y crop | Average `u` difference | Average `v` difference | Case average difference |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        settings = result["settings"]
        model = result["model"]
        coefficient_percentages = {}
        for equation in ("u", "v"):
            coefficient_percentages[equation] = [
                abs(learned - references[equation].get(name, 0.0))
                / abs(references[equation].get(name, 0.0)) * 100
                for learned, name in zip(
                    model[f"{equation}_coefficients"],
                    model[f"{equation}_descriptions"],
                )
            ]
        u_average = np.mean(coefficient_percentages["u"])
        v_average = np.mean(coefficient_percentages["v"])
        case_average = np.mean(coefficient_percentages["u"] + coefficient_percentages["v"])
        lines.append(
            f"| {result['label']} | {settings['t']} | {settings['x']} | {settings['y']} | "
            f"{u_average:.6f}% | {v_average:.6f}% | {case_average:.6f}% |"
        )

    lines.append("")
    for result in results:
        settings = result["settings"]
        model = result["model"]
        lines.extend([
            f"## {result['label']}", "",
            f"Crop settings: `t={settings['t']}, x={settings['x']}, y={settings['y']}`", "",
            f"Skipped input directories: `{', '.join(model['skipped_directories']) or 'none'}`", "",
            "| Equation | Term | True parameter | Learned parameter | Percentage difference |",
            "|---|---|---:|---:|---:|",
        ])
        for equation in ("u", "v"):
            for learned, name in zip(
                model[f"{equation}_coefficients"],
                model[f"{equation}_descriptions"],
            ):
                expected = references[equation].get(name, 0.0)
                lines.append(
                    f"| `{equation}` | `{name}` | `{expected:+.8e}` | "
                    f"`{learned:+.8e}` | `{abs(learned - expected)/abs(expected)*100:.8e}%` |"
                )
        lines.extend([
            "", f"Samples: `{model['sample_count']:,}`; "
            f"U validation RMSE: `{model['u_validation_error']:.8e}`; "
            f"V validation RMSE: `{model['v_validation_error']:.8e}`", "",
        ])

    report_directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(report_directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines))


def run_sweep(input_directories, data_start, tol_values, lam, l0_penalty,
              seed, fixed_crop=2, base_crop_t=1800):
    fields_cache = {}
    cases = []
    sweep = []
    for crop_t in (1800, 2100, 2000):
        for crop_x in (1, 4):
            for crop_y in (1, 4):
                cases.append(
                    (
                        f"Crop t={crop_t}, x={crop_x}, y={crop_y}",
                        {"t": crop_t, "x": crop_x, "y": crop_y},
                    )
                )
    for label, settings in cases:
        print(f"Running {label}: {settings}")
        model = identify_model(
            input_directories,
            data_start,
            settings["t"],
            settings["x"],
            settings["y"],
            tol_values,
            lam,
            l0_penalty,
            seed=seed,
            fields_cache=fields_cache,
        )
        ### Reference for model schema
        """{
        "u_coefficients": w_best, "u_descriptions": dictionary,
        "v_coefficients": w_best_v, "v_descriptions": dictionary_v,
        "data_shape": case_results[0][6], "crop_shape": case_results[0][7],
        "u_tol": tol_best, "u_train_error": train_error,
        "u_validation_error": validation_error,
        "v_tol": tol_best_v, "v_train_error": train_error_v,
        "v_validation_error": validation_error_v,
        "sample_count": x.shape[0],
        "skipped_directories": skipped_directories,
        }"""

        sweep.append({"label": label, "settings": settings, "model": model})
    return sweep


def gradient(u, dt, dy, axis):
    ut = np.gradient(u, dt, axis=0, edge_order=2)
    uy = np.gradient(u, dy, axis=axis, edge_order=2)
    uyy = np.gradient(uy, dy, axis=axis,edge_order=2)
    return ut, uy, uyy



import numpy as np

def gradient2(u, dt, dy, axis):
    def slicing(ax):
        ndim = u.ndim
        s_minus2 = [slice(None)] * ndim
        s_minus1 = [slice(None)] * ndim
        s_interior = [slice(None)] * ndim  # Target for the interior result
        s_plus1  = [slice(None)] * ndim
        s_plus2  = [slice(None)] * ndim

        s_minus2[ax] = slice(0, -4)    # i-2
        s_minus1[ax] = slice(1, -3)    # i-1
        s_interior[ax] = slice(2, -2)  # the interior region (i)
        s_plus1[ax]  = slice(3, -1)    # i+1
        s_plus2[ax]  = slice(4, None)  # i+2

        return s_minus2, s_minus1, s_interior, s_plus1, s_plus2

    # 1. Allocate arrays that are the EXACT same shape as u
    ut = np.empty_like(u)
    uy = np.empty_like(u)
    uyy = np.empty_like(u)

    # --- TIME DERIVATIVE (axis 0) ---
    st_m2, st_m1, st_int, st_p1, st_p2 = slicing(0)
    
    # Calculate Interior
    ut[tuple(st_int)] = (-u[tuple(st_p2)] + 8*u[tuple(st_p1)] - 8*u[tuple(st_m1)] + u[tuple(st_m2)]) / (12 * dt)
    
    # Calculate Boundaries (Left and Right edges)
    ut[0,:,:] = (-25*u[0,:,:] + 48*u[1,:,:] - 36*u[2,:,:] + 16*u[3,:,:] - 3*u[4,:,:]) / (12 * dt)
    ut[1,:,:] = (-3*u[0,:,:] - 10*u[1,:,:] + 18*u[2,:,:] - 6*u[3,:,:] + u[4,:,:]) / (12 * dt)
    ut[-1,:,:] = (25*u[-1,:,:] - 48*u[-2,:,:] + 36*u[-3,:,:] - 16*u[-4,:,:] + 3*u[-5,:,:]) / (12 * dt)
    ut[-2,:,:] = (3*u[-1,:,:] + 10*u[-2,:,:] - 18*u[-3,:,:] + 6*u[-4,:,:] - u[-5,:,:]) / (12 * dt)

    # --- FIRST SPATIAL DERIVATIVE (axis 1 or 2) ---
    s_m2, s_m1, s_int, s_p1, s_p2 = slicing(axis)
    
    # Calculate Interior
    uy[tuple(s_int)] = (-u[tuple(s_p2)] + 8*u[tuple(s_p1)] - 8*u[tuple(s_m1)] + u[tuple(s_m2)]) / (12 * dy)
    
    # Calculate Boundaries
    if axis == 1:
        uy[:,0,:] = (-25*u[:,0,:] + 48*u[:,1,:] - 36*u[:,2,:] + 16*u[:,3,:] - 3*u[:,4,:]) / (12 * dy)
        uy[:,1,:] = (-3*u[:,0,:] - 10*u[:,1,:] + 18*u[:,2,:] - 6*u[:,3,:] + u[:,4,:]) / (12 * dy)
        uy[:,-1,:] = (25*u[:,-1,:] - 48*u[:,-2,:] + 36*u[:,-3,:] - 16*u[:,-4,:] + 3*u[:,-5,:]) / (12 * dy)
        uy[:,-2,:] = (3*u[:,-1,:] + 10*u[:,-2,:] - 18*u[:,-3,:] + 6*u[:,-4,:] - u[:,-5,:]) / (12 * dy)
    elif axis == 2:
        uy[:,:,0] = (-25*u[:,:,0] + 48*u[:,:,1] - 36*u[:,:,2] + 16*u[:,:,3] - 3*u[:,:,4]) / (12 * dy)
        uy[:,:,1] = (-3*u[:,:,0] - 10*u[:,:,1] + 18*u[:,:,2] - 6*u[:,:,3] + u[:,:,4]) / (12 * dy)
        uy[:,:,-1] = (25*u[:,:,-1] - 48*u[:,:,-2] + 36*u[:,:,-3] - 16*u[:,:,-4] + 3*u[:,:,-5]) / (12 * dy)
        uy[:,:,-2] = (3*u[:,:,-1] + 10*u[:,:,-2] - 18*u[:,:,-3] + 6*u[:,:,-4] - u[:,:,-5]) / (12 * dy)

    # --- SECOND SPATIAL DERIVATIVE (axis 1 or 2 applied on uy) ---
    # Calculate Interior
    uyy[tuple(s_int)] = (-uy[tuple(s_p2)] + 8*uy[tuple(s_p1)] - 8*uy[tuple(s_m1)] + uy[tuple(s_m2)]) / (12 * dy)
    
    # Calculate Boundaries
    if axis == 1:
        uyy[:,0,:] = (-25*uy[:,0,:] + 48*uy[:,1,:] - 36*uy[:,2,:] + 16*uy[:,3,:] - 3*uy[:,4,:]) / (12 * dy)
        uyy[:,1,:] = (-3*uy[:,0,:] - 10*uy[:,1,:] + 18*uy[:,2,:] - 6*uy[:,3,:] + uy[:,4,:]) / (12 * dy)
        uyy[:,-1,:] = (25*uy[:,-1,:] - 48*uy[:,-2,:] + 36*uy[:,-3,:] - 16*uy[:,-4,:] + 3*uy[:,-5,:]) / (12 * dy)
        uyy[:,-2,:] = (3*uy[:,-1,:] + 10*uy[:,-2,:] - 18*uy[:,-3,:] + 6*uy[:,-4,:] - uy[:,-5,:]) / (12 * dy)
    elif axis == 2:
        uyy[:,:,0] = (-25*uy[:,:,0] + 48*uy[:,:,1] - 36*uy[:,:,2] + 16*uy[:,:,3] - 3*uy[:,:,4]) / (12 * dy)
        uyy[:,:,1] = (-3*uy[:,:,0] - 10*uy[:,:,1] + 18*uy[:,:,2] - 6*uy[:,:,3] + uy[:,:,4]) / (12 * dy)
        uyy[:,:,-1] = (25*uy[:,:,-1] - 48*uy[:,:,-2] + 36*uy[:,:,-3] - 16*uy[:,:,-4] + 3*uy[:,:,-5]) / (12 * dy)
        uyy[:,:,-2] = (3*uy[:,:,-1] + 10*uy[:,:,-2] - 18*uy[:,:,-3] + 6*uy[:,:,-4] - uy[:,:,-5]) / (12 * dy)

    return ut, uy, uyy

def build_ns_dict(u, v, uy, u_yy, ux, u_xx, p_x):
    dict_keys = []
    oper = []
    ns_dict = {}
    # ns_dict['1'] = np.ones_like(u)
    ns_dict['uu_x'] = u * ux
    ns_dict['vu_y'] = v * uy
    ns_dict['u_xx'] = u_xx
    ns_dict['u_yy'] = u_yy
    ns_dict['p_x'] = p_x

    for key, val in ns_dict.items():
        dict_keys.append(key)
        oper.append(val.reshape(-1))

    oper = np.transpose(np.array(oper))
    print("Dictionary shape:", oper.shape)
    N_features = oper.shape[1]
    print("N_features:", N_features)
    return oper, dict_keys


def build_v_ns_dict(u, v, vx, v_xx, vy, v_yy, p_y):
    dict_keys = []
    oper = []
    ns_dict = {
        'uv_x': u * vx,
        'vv_y': v * vy,
        'v_xx': v_xx,
        'v_yy': v_yy,
        'p_y': p_y,
    }

    for key, val in ns_dict.items():
        dict_keys.append(key)
        oper.append(val.reshape(-1))

    oper = np.transpose(np.array(oper))
    print("V dictionary shape:", oper.shape)
    print("V features:", oper.shape[1])
    return oper, dict_keys



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
        biginds = np.where(abs(weights/norm) >= tol)[0]
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

    biginds = np.where(abs(weights/norm)>= tol)[0]
    if biginds.size > 0:
        weights[:] = 0.0
        weights[biginds] = np.linalg.lstsq(X_norm[:, biginds], Y, rcond=None)[0]
    else:
        weights[:] = 0.0

    return np.array(weights/norm)

def trainStridge(X, y, tol_values, lam, l0_penalty, seed=0):
    rng = np.random.default_rng(seed)

    n_samples = X.shape[0]
    train_size = int(n_samples * 0.8)
    X_train, X_validate = X[:train_size], X[train_size:]
    y_train, y_validate = y[:train_size], y[train_size:]

    score_best = np.inf
    w_best = np.linalg.lstsq(X, y, rcond=None)[0]
    train_err_best = np.linalg.norm(y_train - X_train @ w_best, 2) + l0_penalty * np.count_nonzero(w_best)
    val_err_best = np.linalg.norm(y_validate - X_validate @ w_best, 2) + l0_penalty * np.count_nonzero(w_best)
    tol_best = 0

    for tolerance in tol_values:
        weights = stridge1(X_train, y_train, tolerance, lam)
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



    return w_best, tol_best, train_err_best, val_err_best





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Learn the startup Poiseuille PDE from saved data.")
    # parser.add_argument("--input", type=str, default=os.path.join(os.path.dirname(__file__), "startup_poiseuille_data.npz"))
    parser.add_argument("--lam", type=float, default=1e-8)
    parser.add_argument("--tol_min", type=float, default=1e-6)
    parser.add_argument("--tol_max", type=float, default=1e-2)
    parser.add_argument("--num_tol", type=int, default=40)
    parser.add_argument("--l0_penalty", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_start", type=int, default=150)
    parser.add_argument(
        "--input_directories",
        nargs="+",
        default=[
            os.path.join(os.path.dirname(__file__), "output/2movingwalls"),
            os.path.join(os.path.dirname(__file__), "output/4movingwalls_80"),
            os.path.join(os.path.dirname(__file__), "output/lidcavity"),
            os.path.join(os.path.dirname(__file__), "output/poiseuille"),
        ],
    )
    parser.add_argument("--crop_t", type=int, default=2500, help="discard this many time layers near each temporal boundary")
    parser.add_argument("--crop_y", type=int, default=1, help="discard this many spatial points near each wall")
    parser.add_argument("--crop_x", type=int, default=1, help="discard this many spatial points near each wall")
    # parser.add_argument(
    #     "--sweep",
    #     action="store_true",
    #     help="run the requested temporal, y-crop, and x-crop sweep",
    # )
    parser.add_argument(
        "--sweep_report",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "stridge_sweep_report.md"),
        help="path for the sweep Markdown report",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "stridge_report.md"),
        help="path for the generated Markdown report",
    )
    args = parser.parse_args()

    tol_values = np.logspace(np.log10(args.tol_min), np.log10(args.tol_max), args.num_tol)

    # if args.sweep:
    if 1==0:
        sweep_results = run_sweep(
            args.input_directories,
            args.data_start,
            tol_values,
            args.lam,
            args.l0_penalty,
            args.seed,
        )
        write_sweep_report(
            args.sweep_report,
            sweep_results,
            args.input_directories,
            args.data_start,
            args.lam,
            args.l0_penalty,
        )
        print(f"Markdown sweep report written to: {args.sweep_report}")
        sys.exit(0)
    else:
        case_results = [
            load_case(
                output_directory,
                args.data_start,
                args.crop_t,
                args.crop_x,
                args.crop_y,
            )
            for output_directory in args.input_directories
        ]
        oper_dict = np.concatenate([result[0] for result in case_results])
        y = np.concatenate([result[1] for result in case_results])
        dictionary = case_results[0][2]
        if any(result[2] != dictionary for result in case_results[1:]):
            raise ValueError("All cases must use the same U dictionary ordering")

        x = oper_dict
        w_best, tol_best, err_best, val_error = trainStridge(x,y,tol_values,args.lam, args.l0_penalty)

        oper_dict_v = np.concatenate([result[3] for result in case_results])
        y_v = np.concatenate([result[4] for result in case_results])
        dictionary_v = case_results[0][5]
        if any(result[5] != dictionary_v for result in case_results[1:]):
            raise ValueError("All cases must use the same V dictionary ordering")

        x_v = oper_dict_v
        w_best_v, tol_best_v, err_best_v, val_error_v = trainStridge(
            x_v, y_v, tol_values, args.lam, args.l0_penalty
        )

        write_2d_markdown_report(
            args.report,
            w_best,
            dictionary,
            w_best_v,
            dictionary_v,
            case_results[0][6],
            case_results[0][7],
            tol_best,
            err_best,
            val_error,
            tol_best_v,
            err_best_v,
            val_error_v,
            data_start=args.data_start,
            crop_settings={"t": args.crop_t, "x": args.crop_x, "y": args.crop_y},
            lam=args.lam,
            l0_penalty=args.l0_penalty,
            u_reference_coefficients={
                "uu_x": -c_advective,
                "vu_y": -c_advective,
                "u_xx": c_viscous,
                "u_yy": c_viscous,
                "p_x": -c_pressure,
            },
            v_reference_coefficients={
                "uv_x": -c_advective,
                "vv_y": -c_advective,
                "v_xx": c_viscous,
                "v_yy": c_viscous,
                "p_y": -c_pressure,
            },
            input_directory=", ".join(args.input_directories),
        )
        print(f"Markdown U/V report written to: {args.report}")



