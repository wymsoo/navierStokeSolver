# Physics Simulation and STRidge Identification Report

> Numerical identification of the streamwise momentum equation from saved velocity and pressure fields.

## 1. Simulation Parameters

| Quantity | Symbol | Value | Unit |
|---|---:|---:|---|
| Fluid density | `rho` | `1.00000000e+03` | kg m$^{-3}$ |
| Dynamic viscosity | `viscosity` | `1.00000000e+01` | Pa s |
| Kinematic viscosity | `nu` | `1.00000000e-02` | m$^2$ s$^{-1}$ |
| Characteristic velocity | `u_max` | `1.00000000e+00` | m s$^{-1}$ |
| Characteristic length | `D` | `1.00000000e+00` | m |
| Reynolds number | `Re` | `1.00000000e+02` | dimensionless |
| Body force | `G` | `9.81000000e+00` | m s$^{-2}$ |
| Grid points in x | `Nx` | `3.10000000e+01` | cells |
| Grid points in y | `Ny` | `3.10000000e+01` | cells |
| Grid spacing in x | `dx` | `3.22580645e-02` | m |
| Grid spacing in y | `dy` | `3.22580645e-02` | m |
| Time step | `dt` | `1.00000000e-04` | s |
| Numerical threshold | `epsilon` | `7.00000000e-11` | - |

## 2. Numerical Discretisation

| Quantity | Symbol | Value | Unit |
|---|---:|---:|---|
| Parameters listed above are the values used by the selected solver. |  |  |  |

## 3. Data and Pre-processing

| Quantity | Value |
|---|---:|
| Loaded field shape $(t, x, y)$ | `(5472, 50, 50)` |
| Regression field shape $(t, x, y)$ | `(2972, 49, 49)` |
| Regression samples | `7,135,772` |
| Dictionary terms | `5` |
| First saved time index | `150` |
| Temporal crop at each boundary | `2500` layers |
| x-boundary crop | `1` points |
| y-boundary crop | `1` points |

## 4. Governing Equation Verification

### Reference equation

`u_t =- 1.00000000e+00*uu_x - 1.00000000e+00*vu_y + 1.00000000e-02*u_xx + 1.00000000e-02*u_yy - 1.00000000e-03*p_x`

### Learned equation

`u_t = 0`

| Term | Reference coefficient | Learned coefficient | Percentage error |
|---|---:|---:|---:|
| `uu_x` | `-1.00000000e+00` | `+0.00000000e+00` | `-1.00000000e+02%` |
| `vu_y` | `-1.00000000e+00` | `+0.00000000e+00` | `-1.00000000e+02%` |
| `u_xx` | `+1.00000000e-02` | `+0.00000000e+00` | `1.00000000e+02%` |
| `u_yy` | `+1.00000000e-02` | `+0.00000000e+00` | `1.00000000e+02%` |
| `p_x` | `-1.00000000e-03` | `+0.00000000e+00` | `-1.00000000e+02%` |

## 5. STRidge Selection and Error Metrics

| Metric | Value |
|---|---:|
| Regularisation parameter, $\lambda$ | `1.00000000e-08` |
| Selected tolerance | `1.00000000e-02` |
| L0 penalty | `1.00000000e-06` |
| Training RMSE | `1.68052147e-01` |
| Validation RMSE | `4.79091954e-02` |

## 6. Reproducibility Notes

- Input directory: `/Users/minnie/Desktop/PhysicsFYP/ns_solver/stridge_model/output/2movingwalls, /Users/minnie/Desktop/PhysicsFYP/ns_solver/stridge_model/output/4movingwalls_80, /Users/minnie/Desktop/PhysicsFYP/ns_solver/stridge_model/output/lidcavity, /Users/minnie/Desktop/PhysicsFYP/ns_solver/stridge_model/output/poiseuille`
- Fields were loaded starting at saved time index `150`.
- Derivatives were calculated using second-order edge-aware finite differences via `numpy.gradient`.
- The learned equation is compared term-by-term with the supplied reference coefficients.

## 7. Transverse Momentum Equation

### Reference equation

`v_t =- 1.00000000e+00*uv_x - 1.00000000e+00*vv_y + 1.00000000e-02*v_xx + 1.00000000e-02*v_yy - 1.00000000e-03*p_y`

### Learned equation

`v_t =- 1.51076158e+00*uv_x - 1.20263157e+00*vv_y + 4.40334025e-02*v_xx + 3.81540032e-02*v_yy - 1.53723020e-03*p_y`

| Term | Reference coefficient | Learned coefficient | Percentage error |
|---|---:|---:|---:|
| `uv_x` | `-1.00000000e+00` | `-1.51076158e+00` | `-5.10761585e+01%` |
| `vv_y` | `-1.00000000e+00` | `-1.20263157e+00` | `-2.02631569e+01%` |
| `v_xx` | `+1.00000000e-02` | `+4.40334025e-02` | `3.40334025e+02%` |
| `v_yy` | `+1.00000000e-02` | `+3.81540032e-02` | `2.81540032e+02%` |
| `p_y` | `-1.00000000e-03` | `-1.53723020e-03` | `-5.37230197e+01%` |

### V STRidge Selection and Error Metrics

| Metric | Value |
|---|---:|
| Selected tolerance | `1.51177507e-03` |
| Training RMSE | `8.58850923e-02` |
| Validation RMSE | `2.08857732e-02` |
