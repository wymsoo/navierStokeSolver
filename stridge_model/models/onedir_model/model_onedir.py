import os

import numpy as np


def find_run_directories(input_directories):
    run_directories = []
    for input_directory in input_directories:
        if not os.path.isdir(input_directory):
            raise FileNotFoundError(f"Input directory does not exist: {input_directory}")
        for directory, _, filenames in os.walk(input_directory):
            if any(name.startswith("data_t=") and name.endswith(".npz") for name in filenames):
                run_directories.append(directory)
    return sorted(set(run_directories))


def load_case(output_directory, data_start=0, crop_t=1, crop_y=2):
    snapshot_files = []
    for filename in os.listdir(output_directory):
        if filename.startswith("data_t=") and filename.endswith(".npz"):
            try:
                index = int(filename[len("data_t="):-len(".npz")])
            except ValueError:
                continue
            if index >= data_start:
                snapshot_files.append((index, filename))
    snapshot_files.sort()
    if len(snapshot_files) < 3:
        raise ValueError(f"At least three NPZ snapshots are required in {output_directory}")

    profiles = []
    indices = []
    dt_value = None
    dy_value = None
    profile_size = None
    for index, filename in snapshot_files:
        filepath = os.path.join(output_directory, filename)
        with np.load(filepath) as data:
            required_fields = {"u", "dt", "dy"}
            missing_fields = required_fields.difference(data.files)
            if missing_fields:
                raise ValueError(f"Snapshot {filepath} is missing fields: {sorted(missing_fields)}")
            velocity = np.asarray(data["u"])
            if velocity.ndim != 2 or not np.issubdtype(velocity.dtype, np.number):
                raise ValueError(f"Velocity field in {filepath} must be a numeric 2D array")
            profile = velocity[velocity.shape[0] // 2, :]
            if profile_size is not None and profile.size != profile_size:
                raise ValueError(f"Inconsistent transverse profile sizes in {output_directory}")
            profile_size = profile.size

            snapshot_dt = float(np.asarray(data["dt"]).item())
            snapshot_dy = float(np.asarray(data["dy"]).item())
            if not np.isfinite(snapshot_dt) or snapshot_dt <= 0:
                raise ValueError(f"Snapshot time step must be positive and finite: {filepath}")
            if not np.isfinite(snapshot_dy) or snapshot_dy <= 0:
                raise ValueError(f"Snapshot dy must be positive and finite: {filepath}")
            if dt_value is not None and not np.isclose(snapshot_dt, dt_value):
                raise ValueError(f"Inconsistent dt values in {output_directory}")
            if dy_value is not None and not np.isclose(snapshot_dy, dy_value):
                raise ValueError(f"Inconsistent dy values in {output_directory}")

            profiles.append(profile)
            indices.append(index)
            dt_value = snapshot_dt
            dy_value = snapshot_dy

    velocity_profile = np.asarray(profiles)
    time = (np.asarray(indices, dtype=float) - indices[0]) * dt_value
    ut, uy, uyy = gradient(velocity_profile, time, dy_value)

    t_slice = slice(crop_t, velocity_profile.shape[0] - crop_t if crop_t else None)
    y_slice = slice(crop_y, velocity_profile.shape[1] - crop_y if crop_y else None)
    cropped = velocity_profile[t_slice, y_slice]
    if cropped.size == 0 or min(cropped.shape) < 1:
        raise ValueError(f"Cropping removes all samples in {output_directory}")

    operators, dictionary = build_dict(
        cropped,
        uy[t_slice, y_slice],
        uyy[t_slice, y_slice],
    )
    return {
        "X": operators,
        "y": ut[t_slice, y_slice].reshape(-1),
        "dictionary": dictionary,
        "data_shape": velocity_profile.shape,
        "crop_shape": cropped.shape,
        "dt": dt_value,
        "dy": dy_value,
    }


def gradient(u, time, dy):
    ut = np.gradient(u, time, axis=0, edge_order=2)
    uy = np.gradient(u, dy, axis=1, edge_order=2)
    uyy = np.gradient(uy, dy, axis=1, edge_order=2)
    return ut, uy, uyy


def build_dict(u, uy, u_yy):
    terms = [
        (u * uy, "u*u_y"),
        (u * u_yy, "u*u_yy"),
        (uy * u_yy, "u_y*u_yy"),
        (u, "u"),
        (uy, "u_y"),
        (u_yy, "u_yy"),
        (np.ones_like(u), "1"),
    ]
    operators = np.column_stack([value.reshape(-1) for value, _ in terms])
    return operators, [name for _, name in terms]



def stridge1(X, Y, tol, lam, max_iter=100):
    n_features = X.shape[1]
    column_norms = np.linalg.norm(X, axis=0)
    column_norms[column_norms < 1e-16] = 1.0
    target_norm = np.linalg.norm(Y)
    if target_norm < 1e-16:
        target_norm = 1.0

    normalized_x = X / column_norms
    normalized_y = Y / target_norm
    ridge = normalized_x.T @ normalized_x + lam * np.eye(n_features)
    weights = np.linalg.lstsq(ridge, normalized_x.T @ normalized_y, rcond=None)[0]
    previous_active = None

    for _ in range(max_iter):
        physical_weights = weights * target_norm / column_norms
        active = np.flatnonzero(np.abs(physical_weights) >= tol)
        if np.array_equal(active, previous_active):
            break
        if active.size == 0:
            weights[:] = 0.0
            break
        previous_active = active
        weights[:] = 0.0
        active_x = normalized_x[:, active]
        active_ridge = active_x.T @ active_x + lam * np.eye(active.size)
        weights[active] = np.linalg.lstsq(
            active_ridge, active_x.T @ normalized_y, rcond=None
        )[0]

    physical_weights = weights * target_norm / column_norms
    active = np.flatnonzero(np.abs(physical_weights) >= tol)
    weights[:] = 0.0
    if active.size:
        weights[active] = np.linalg.lstsq(
            normalized_x[:, active], normalized_y, rcond=None
        )[0]
    return weights * target_norm / column_norms


def trainStridge(X, y, tol_values, lam, l0_penalty, seed=0):
    if X.shape[0] != y.size or X.shape[0] < 2:
        raise ValueError("STRidge requires matching arrays with at least two samples")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(y.size)
    train_size = int(y.size * 0.8)
    train_size = min(max(train_size, 1), y.size - 1)
    train_indices, validate_indices = indices[:train_size], indices[train_size:]
    X_train, X_validate = X[train_indices], X[validate_indices]
    y_train, y_validate = y[train_indices], y[validate_indices]

    best_weights = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
    best_train_error = np.sqrt(np.mean((y_train - X_train @ best_weights) ** 2))
    best_validation_error = np.sqrt(np.mean((y_validate - X_validate @ best_weights) ** 2))
    best_score = best_validation_error + l0_penalty * np.count_nonzero(best_weights)
    best_tolerance = 0.0
    train_losses = []
    validation_losses = []
    evaluated_tolerances = []

    for tolerance in tol_values:
        weights = stridge1(X_train, y_train, tolerance, lam)
        print("WEIGHTS:", weights)
        train_error = np.sqrt(np.mean((y_train - X_train @ weights) ** 2))
        validation_error = np.sqrt(np.mean((y_validate - X_validate @ weights) ** 2))
        score = validation_error + l0_penalty * np.count_nonzero(np.abs(weights) > 1e-12)
        train_losses.append(train_error)
        validation_losses.append(validation_error)
        evaluated_tolerances.append(tolerance)
        if score < best_score:
            best_score = score
            best_weights = weights
            best_train_error = train_error
            best_validation_error = validation_error
            best_tolerance = tolerance

    return (
        best_weights,
        best_tolerance,
        best_train_error,
        best_validation_error,
        train_losses,
        validation_losses,
        evaluated_tolerances,
    )


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






