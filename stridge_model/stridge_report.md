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
| Loaded field shape $(t, x, y)$ | `[(1759, 80, 80), (1647, 80, 80)]` |
| Regression field shape $(t, x, y)$ | `[(1559, 78, 78), (1447, 78, 78)]` |
| Regression samples | `83,501,265,423,888` |
| Dictionary terms | `5` |
| First saved time index | `1` |
| Temporal crop at each boundary | `100` layers |
| x-boundary crop | `1` points |
| y-boundary crop | `1` points |

## 4. Governing Equation Verification

### Reference equation

`u_t =- 1.00000000e+00*uu_x - 1.00000000e+00*vu_y + 1.00000000e-01*u_xx + 1.00000000e-01*u_yy - 1.00000000e-02*p_x`

### Learned equation

`u_t =- 1.39065963e+00*uu_x - 3.33509247e+00*vu_y + 2.73183402e-01*u_xx + 2.87097553e-01*u_yy - 2.87070088e-02*p_x`

| Term | Reference coefficient | Learned coefficient | Percentage error |
|---|---:|---:|---:|
| `uu_x` | `-1.00000000e+00` | `-1.39065963e+00` | `-3.90659630e+01%` |
| `vu_y` | `-1.00000000e+00` | `-3.33509247e+00` | `-2.33509247e+02%` |
| `u_xx` | `+1.00000000e-01` | `+2.73183402e-01` | `1.73183402e+02%` |
| `u_yy` | `+1.00000000e-01` | `+2.87097553e-01` | `1.87097553e+02%` |
| `p_x` | `-1.00000000e-02` | `-2.87070088e-02` | `-1.87070088e+02%` |

## 5. STRidge Selection and Error Metrics

| Metric | Value |
|---|---:|
| Regularisation parameter, $\lambda$ | `1.00000000e-08` |
| Selected tolerance | `1.00000000e-02` |
| L0 penalty | `1.00000000e-06` |
| Training RMSE | `8.75377065e-01` |
| Validation RMSE | `6.34671243e-01` |

## 6. Reproducibility Notes

- Input directory: `/Users/minnie/Desktop/PhysicsFYP/ns_solver/stridge_model/output_npz/4movingwalls_81, /Users/minnie/Desktop/PhysicsFYP/ns_solver/stridge_model/output_npz/lidcavity_81`
- Fields were loaded starting at saved time index `1`.
- Derivatives were calculated using second-order edge-aware finite differences via `numpy.gradient`.
- The learned equation is compared term-by-term with the supplied reference coefficients.

## 7. Transverse Momentum Equation

### Reference equation

`v_t =- 1.00000000e+00*uv_x - 1.00000000e+00*vv_y + 1.00000000e-01*v_xx + 1.00000000e-01*v_yy - 1.00000000e-02*p_y`

### Learned equation

`v_t =- 4.03184496e+00*uv_x - 7.24924486e-01*vv_y + 2.98441750e-01*v_xx + 2.88329451e-01*v_yy - 3.02961983e-02*p_y`

| Term | Reference coefficient | Learned coefficient | Percentage error |
|---|---:|---:|---:|
| `uv_x` | `-1.00000000e+00` | `-4.03184496e+00` | `-3.03184496e+02%` |
| `vv_y` | `-1.00000000e+00` | `-7.24924486e-01` | `-2.75075514e+01%` |
| `v_xx` | `+1.00000000e-01` | `+2.98441750e-01` | `1.98441750e+02%` |
| `v_yy` | `+1.00000000e-01` | `+2.88329451e-01` | `1.88329451e+02%` |
| `p_y` | `-1.00000000e-02` | `-3.02961983e-02` | `-2.02961983e+02%` |

### V STRidge Selection and Error Metrics

| Metric | Value |
|---|---:|
| Selected tolerance | `1.00000000e-02` |
| Training RMSE | `7.36747037e-01` |
| Validation RMSE | `1.82501522e-01` |
