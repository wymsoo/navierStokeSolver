import numpy as np 
import os

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
# D = 0.005
D = 1.0
viscosity = 10
Re = (rho * u_max * D) / viscosity
nu = viscosity/rho
c_advective = 1
c_pressure = 1/rho
c_viscous = 1/Re
def write_markdown_report(path, coefficients, descriptions, data_shape,
                          crop_shape, tol_best, train_error,
                          validation_error, data_start, crop_settings,
                          lam, l0_penalty, parameters=None,
                          reference_coefficients=None, input_directory=None,
                          data_shape_labels="(t, x, y)"):
    parameters = parameters or {
        "rho": (rho, "kg m$^{-3}$", "Fluid density"),
        "viscosity": (viscosity, "Pa s", "Dynamic viscosity"),
        "nu": (nu, "m$^2$ s$^{-1}$", "Kinematic viscosity"),
        "u_max": (u_max, "m s$^{-1}$", "Characteristic velocity"),
        "D": (D, "m", "Characteristic length"),
        "Re": (Re, "dimensionless", "Reynolds number"),
        "G": (G, "m s$^{-2}$", "Body force"),
        "Nx": (Nx, "cells", "Grid points in x"),
        "Ny": (Ny, "cells", "Grid points in y"),
        "dx": (dx, "m", "Grid spacing in x"),
        "dy": (dy, "m", "Grid spacing in y"),
        "dt": (dt, "s", "Time step"),
        "epsilon": (epsilon, "-", "Numerical threshold"),
    }
    reference_coefficients = reference_coefficients or {
        "uu_x": -c_advective,
        "vu_y": -c_advective,
        "u_xx": c_viscous,
        "u_yy": c_viscous,
        "p_x": -c_pressure,
    }
    true_equation = [reference_coefficients.get(name, 0.0) for name in descriptions]
    parameter_rows = [
        f"| {label} | `{name}` | `{value:.8e}` | {unit} |"
        for name, (value, unit, label) in parameters.items()
    ]
    report_lines = [
        "# Physics Simulation and STRidge Identification Report",
        "",
        "> Numerical identification of the streamwise momentum equation from saved velocity and pressure fields.",
        "",
        "## 1. Simulation Parameters",
        "",
        "| Quantity | Symbol | Value | Unit |",
        "|---|---:|---:|---|",
        *parameter_rows,
        "",
        "## 2. Numerical Discretisation",
        "",
        "| Quantity | Symbol | Value | Unit |",
        "|---|---:|---:|---|",
        "| Parameters listed above are the values used by the selected solver. |  |  |  |",
        "",
        "## 3. Data and Pre-processing",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Loaded field shape ${data_shape_labels}$ | `{data_shape}` |",
        f"| Regression field shape ${data_shape_labels}$ | `{crop_shape}` |",
        f"| Regression samples | `{int(np.prod(crop_shape)):,}` |",
        f"| Dictionary terms | `{len(descriptions)}` |",
        f"| First saved time index | `{data_start}` |",
        f"| Temporal crop at each boundary | `{crop_settings['t']}` layers |",
        f"| x-boundary crop | `{crop_settings['x']}` points |",
        f"| y-boundary crop | `{crop_settings['y']}` points |",
        "",
        "## 4. Governing Equation Verification",
        "",
        "### Reference equation",
        "",
        f"`{format_equation(true_equation, descriptions, lhs='u_t')}`",
        "",
        "### Learned equation",
        "",
        f"`{format_equation(coefficients, descriptions, lhs='u_t')}`",
        "",
        "| Term | Reference coefficient | Learned coefficient | Absolute error |",
        "|---|---:|---:|---:|",
    ]
    for learned, name, expected in zip(coefficients, descriptions, true_equation):
        report_lines.append(
            f"| `{name}` | `{expected:+.8e}` | `{learned:+.8e}` | `{abs(learned - expected):.8e}` |"
        )
    report_lines.extend([
        "",
        "## 5. STRidge Selection and Error Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Regularisation parameter, $\\lambda$ | `{lam:.8e}` |",
        f"| Selected tolerance | `{tol_best:.8e}` |",
        f"| L0 penalty | `{l0_penalty:.8e}` |",
        f"| Training RMSE | `{train_error:.8e}` |",
        f"| Validation RMSE | `{validation_error:.8e}` |",
        "",
        "## 6. Reproducibility Notes",
        "",
        f"- Input directory: `{input_directory or 'not specified'}`",
        f"- Fields were loaded starting at saved time index `{data_start}`.",
        "- Derivatives were calculated using second-order edge-aware finite differences via `numpy.gradient`.",
        "- The learned equation is compared term-by-term with the supplied reference coefficients.",
        "",
    ])
    report_directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(report_directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(report_lines))


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
