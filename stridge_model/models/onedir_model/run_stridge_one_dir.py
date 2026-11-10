import numpy as np

from format_report import format_equation, write_markdown_report
from config_1d import initialize_config
from model_onedir import find_run_directories, load_case, trainStridge


def main(argv=None):
    args = initialize_config(argv)
    run_directories = find_run_directories(args.input_directories)
    if not run_directories:
        raise FileNotFoundError("No data_t=*.npz snapshots found in the supplied input directories")

    cases = [
        load_case(directory, args.data_start, args.crop_t, args.crop_y)
        for directory in run_directories
    ]
    dictionary = cases[0]["dictionary"]
    if any(case["dictionary"] != dictionary for case in cases[1:]):
        raise ValueError("All runs must use the same 1D operator ordering")

    x = np.concatenate([case["X"] for case in cases])
    y = np.concatenate([case["y"] for case in cases])
    tol_values = np.logspace(
        np.log10(args.tol_min), np.log10(args.tol_max), args.num_tol
    )
    (
        weights,
        tolerance,
        train_error,
        validation_error,
        train_losses,
        validation_losses,
        evaluated_tolerances,
    ) = trainStridge(x, y, tol_values, args.lam, args.l0_penalty, args.seed)

    print(f"Loaded {len(cases)} run(s): {', '.join(run_directories)}")
    print(f"Regression matrix: {x.shape}; target: {y.shape}")
    print("Discovered equation")
    print(format_equation(weights, dictionary, lhs="u_t"))
    print(f"Best tolerance      : {tolerance:.8e}")
    print(f"Train RMSE          : {train_error:.8e}")
    print(f"Validation RMSE     : {validation_error:.8e}")
    print(f"Reference equation  : u_t = {args.force} + {args.nu}*u_yy")
    print("Estimated coefficients")
    for coefficient, name in zip(weights, dictionary):
        print(f"  {name:12s}: {coefficient:+.8e}")

    data_shapes = [case["data_shape"] for case in cases]
    crop_shapes = [case["crop_shape"] for case in cases]
    write_markdown_report(
        args.report,
        weights,
        dictionary,
        data_shapes,
        crop_shapes,
        tolerance,
        train_error,
        validation_error,
        data_start=args.data_start,
        crop_settings={"t": args.crop_t, "x": 0, "y": args.crop_y},
        lam=args.lam,
        l0_penalty=args.l0_penalty,
        parameters={
            "rho": (args.rho, "kg m$^{-3}$", "Fluid density"),
            "viscosity": (args.viscosity, "Pa s", "Dynamic viscosity"),
            "nu": (args.nu, "m$^2$ s$^{-1}$", "Kinematic viscosity"),
            "force": (args.force, "m s$^{-2}$", "Constant forcing"),
            "dt": (cases[0]["dt"], "s", "Time step"),
            "dy": (cases[0]["dy"], "m", "Transverse grid spacing"),
        },
        reference_coefficients={"1": args.force, "u_yy": args.nu},
        input_directory=", ".join(run_directories),
        data_shape_labels="(t, y)",
    )
    print(f"Markdown report written to: {args.report}")


if __name__ == "__main__":
    main()
