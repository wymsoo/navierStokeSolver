# Physics Simulation and STRidge Identification Report

> Numerical identification of the streamwise momentum equation from saved velocity and pressure fields.

## 1. Simulation Parameters

| Quantity | Symbol | Value | Unit |
|---|---:|---:|---|
| Fluid density | `rho` | `1.00000000e+01` | kg m$^{-3}$ |
| Dynamic viscosity | `viscosity` | `1.00000000e+01` | Pa s |
| Kinematic viscosity | `nu` | `1.00000000e+00` | m$^2$ s$^{-1}$ |
| Characteristic velocity | `u_max` | `1.00000000e+00` | m s$^{-1}$ |
| Characteristic length | `D` | `1.00000000e+00` | m |
| Reynolds number | `Re` | `1.00000000e+00` | dimensionless |
| Body force | `Fx_ext` | `0.00000000e+00` | m s$^{-2}$ |
| Grid points in x | `Nx` | `3.10000000e+01` | cells |
| Grid points in y | `Ny` | `3.10000000e+01` | cells |
| Grid spacing in x | `dx` | `3.22580645e-02` | m |
| Grid spacing in y | `dy` | `3.22580645e-02` | m |
| Time step | `dt` | `1.00000000e-07` | s |
| Numerical threshold | `epsilon` | `7.00000000e-11` | - |

## 2. Numerical Discretisation

| Quantity | Symbol | Value | Unit |
|---|---:|---:|---|
| Parameters listed above are the values used by the selected solver. |  |  |  |

## 3. Data and Pre-processing

| Quantity | Value |
|---|---:|
| Loaded field shape $(t, x, y)$ | `[(2380, 30, 30), (479, 30, 30), (2765, 30, 30), (378, 30, 30), (2716, 30, 30), (1830, 30, 30), (1830, 30, 30), (2716, 30, 30), (2716, 30, 30), (1830, 30, 30)]` |
| Regression field shape $(t, x, y)$ | `[(2180, 28, 28), (279, 28, 28), (2565, 28, 28), (178, 28, 28), (2516, 28, 28), (1630, 28, 28), (1630, 28, 28), (2516, 28, 28), (2516, 28, 28), (1630, 28, 28)]` |
| Regression samples | `-1,220,475,499,017,404,416` |
| Dictionary terms | `5` |
| First saved time index | `1` |
| Temporal crop at each boundary | `100` layers |
| x-boundary crop | `1` points |
| y-boundary crop | `1` points |

## 4. Governing Equation Verification

### Reference equation

`u_t =- 1.00000000e+00*uu_x - 1.00000000e+00*vu_y + 1.00000000e+00*u_xx + 1.00000000e+00*u_yy - 1.00000000e-01*p_x`

### Learned equation

`u_t =- 2.92302565e+02*uu_x + 7.86571143e+02*vu_y + 2.99674243e+02*u_xx + 3.62457964e+02*u_yy - 3.35506122e+01*p_x`

| Term | Reference coefficient | Learned coefficient | Percentage error |
|---|---:|---:|---:|
| `uu_x` | `-1.00000000e+00` | `-2.92302565e+02` | `-2.91302565e+04%` |
| `vu_y` | `-1.00000000e+00` | `+7.86571143e+02` | `-7.87571143e+04%` |
| `u_xx` | `+1.00000000e+00` | `+2.99674243e+02` | `2.98674243e+04%` |
| `u_yy` | `+1.00000000e+00` | `+3.62457964e+02` | `3.61457964e+04%` |
| `p_x` | `-1.00000000e-01` | `-3.35506122e+01` | `-3.34506122e+04%` |

## 5. STRidge Selection and Error Metrics

| Metric | Value |
|---|---:|
| Regularisation parameter, $\lambda$ | `1.00000000e-08` |
| Selected tolerance | `1.00000000e-02` |
| L0 penalty | `1.00000000e-06` |
| Training RMSE | `4.61098495e+03` |
| Validation RMSE | `3.05907604e+03` |

## 6. Reproducibility Notes

- Input directory: `/Users/minnie/Desktop/PhysicsFYP/ns_solver/stridge_model/output_npz/fourmovingwalls_Nx31/run_001, /Users/minnie/Desktop/PhysicsFYP/ns_solver/stridge_model/output_npz/fourmovingwalls_Nx31/run_002, /Users/minnie/Desktop/PhysicsFYP/ns_solver/stridge_model/output_npz/lidcavity_Nx31/run_001, /Users/minnie/Desktop/PhysicsFYP/ns_solver/stridge_model/output_npz/lidcavity_Nx31/run_002, /Users/minnie/Desktop/PhysicsFYP/ns_solver/stridge_model/output_npz/poiseuille_Nx31/run_001, /Users/minnie/Desktop/PhysicsFYP/ns_solver/stridge_model/output_npz/poiseuille_Nx31/run_002, /Users/minnie/Desktop/PhysicsFYP/ns_solver/stridge_model/output_npz/poiseuille_Nx31/run_003, /Users/minnie/Desktop/PhysicsFYP/ns_solver/stridge_model/output_npz/twomovingwalls_Nx31/run_001, /Users/minnie/Desktop/PhysicsFYP/ns_solver/stridge_model/output_npz/twomovingwalls_Nx31/run_002, /Users/minnie/Desktop/PhysicsFYP/ns_solver/stridge_model/output_npz/twomovingwalls_Nx31/run_003`
- Fields were loaded starting at saved time index `1`.
- Derivatives were calculated using second-order edge-aware finite differences via `numpy.gradient`.
- The learned equation is compared term-by-term with the supplied reference coefficients.

## 7. Transverse Momentum Equation

### Reference equation

`v_t =- 1.00000000e+00*uv_x - 1.00000000e+00*vv_y + 1.00000000e+00*v_xx + 1.00000000e+00*v_yy - 1.00000000e-01*p_y`

### Learned equation

`v_t =- 1.28455123e+03*uv_x - 4.64627040e+02*vv_y + 3.67230830e+02*v_xx + 3.38224371e+02*v_yy - 3.59365467e+01*p_y`

| Term | Reference coefficient | Learned coefficient | Percentage error |
|---|---:|---:|---:|
| `uv_x` | `-1.00000000e+00` | `-1.28455123e+03` | `-1.28355123e+05%` |
| `vv_y` | `-1.00000000e+00` | `-4.64627040e+02` | `-4.63627040e+04%` |
| `v_xx` | `+1.00000000e+00` | `+3.67230830e+02` | `3.66230830e+04%` |
| `v_yy` | `+1.00000000e+00` | `+3.38224371e+02` | `3.37224371e+04%` |
| `p_y` | `-1.00000000e-01` | `-3.59365467e+01` | `-3.58365467e+04%` |

### V STRidge Selection and Error Metrics

| Metric | Value |
|---|---:|
| Selected tolerance | `1.00000000e-06` |
| Training RMSE | `4.42007985e+03` |
| Validation RMSE | `0.00000000e+00` |
