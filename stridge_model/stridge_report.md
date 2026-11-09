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
| Grid points in x | `Nx` | `8.10000000e+01` | cells |
| Grid points in y | `Ny` | `8.10000000e+01` | cells |
| Grid spacing in x | `dx` | `1.23456790e-02` | m |
| Grid spacing in y | `dy` | `1.23456790e-02` | m |
| Time step | `dt` | `1.00000000e-04` | s |
| Numerical threshold | `epsilon` | `5.00000000e-10` | - |

## 2. Numerical Discretisation

| Quantity | Symbol | Value | Unit |
|---|---:|---:|---|
| Parameters listed above are the values used by the selected solver. |  |  |  |

## 3. Data and Pre-processing

| Quantity | Value |
|---|---:|
| Loaded field shape $(t, x, y)$ | `[(4367, 81, 81), (2175, 81, 81), (2175, 81, 81), (2175, 81, 81), (2175, 81, 81)]` |
| Regression field shape $(t, x, y)$ | `[(1000, 77, 77), (1000, 77, 77), (1000, 77, 77), (1000, 77, 77), (1000, 77, 77)]` |
| Regression samples | `974,790,547,927,105,536` |
| Dictionary terms | `5` |
| First saved time index | `1` |
| Temporal crop at each boundary | `2000` layers |
| x-boundary crop | `2` points |
| y-boundary crop | `2` points |

## 4. Governing Equation Verification

### Reference equation

`u_t =- 1.00000000e+00*uu_x - 1.00000000e+00*vu_y + 1.00000000e+00*u_xx + 1.00000000e+00*u_yy - 1.00000000e-01*p_x`

### Learned equation

`u_t =- 9.41118491e-01*uu_x - 9.90131708e-01*vu_y + 9.13159709e-01*u_xx + 9.66162380e-01*u_yy - 9.98968541e-02*p_x`

| Term | Reference coefficient | Learned coefficient | Percentage error |
|---|---:|---:|---:|
| `uu_x` | `-1.00000000e+00` | `-9.41118491e-01` | `-5.88815093e+00%` |
| `vu_y` | `-1.00000000e+00` | `-9.90131708e-01` | `-9.86829166e-01%` |
| `u_xx` | `+1.00000000e+00` | `+9.13159709e-01` | `8.68402910e+00%` |
| `u_yy` | `+1.00000000e+00` | `+9.66162380e-01` | `3.38376196e+00%` |
| `p_x` | `-1.00000000e-01` | `-9.98968541e-02` | `-1.03145900e-01%` |

## 5. STRidge Selection and Error Metrics

| Metric | Value |
|---|---:|
| Regularisation parameter, $\lambda$ | `1.00000000e-06` |
| Selected tolerance | `1.00000000e-02` |
| L0 penalty | `1.00000000e-06` |
| Training RMSE | `3.07323403e+00` |
| Validation RMSE | `3.02273788e+00` |

## 6. Reproducibility Notes

- Input directory: `/Users/minnie/Desktop/PhysicsFYP/ns_solver/stridge_model/output_npz/fourmovingwalls_Nx81/run_001, /Users/minnie/Desktop/PhysicsFYP/ns_solver/stridge_model/output_npz/lidcavity_Nx81_sideE/run_001, /Users/minnie/Desktop/PhysicsFYP/ns_solver/stridge_model/output_npz/lidcavity_Nx81_sideN/run_001, /Users/minnie/Desktop/PhysicsFYP/ns_solver/stridge_model/output_npz/lidcavity_Nx81_sideS/run_001, /Users/minnie/Desktop/PhysicsFYP/ns_solver/stridge_model/output_npz/lidcavity_Nx81_sideW/run_001`
- Fields were loaded starting at saved time index `1`.
- Derivatives were calculated using second-order edge-aware finite differences via `numpy.gradient`.
- The learned equation is compared term-by-term with the supplied reference coefficients.

## 7. Transverse Momentum Equation

### Reference equation

`v_t =- 1.00000000e+00*uv_x - 1.00000000e+00*vv_y + 1.00000000e+00*v_xx + 1.00000000e+00*v_yy - 1.00000000e-01*p_y`

### Learned equation

`v_t =- 9.72430262e-01*uv_x - 9.52599492e-01*vv_y + 9.65003764e-01*v_xx + 9.13152692e-01*v_yy - 9.98561535e-02*p_y`

| Term | Reference coefficient | Learned coefficient | Percentage error |
|---|---:|---:|---:|
| `uv_x` | `-1.00000000e+00` | `-9.72430262e-01` | `-2.75697376e+00%` |
| `vv_y` | `-1.00000000e+00` | `-9.52599492e-01` | `-4.74005078e+00%` |
| `v_xx` | `+1.00000000e+00` | `+9.65003764e-01` | `3.49962363e+00%` |
| `v_yy` | `+1.00000000e+00` | `+9.13152692e-01` | `8.68473082e+00%` |
| `p_y` | `-1.00000000e-01` | `-9.98561535e-02` | `-1.43846530e-01%` |

### V STRidge Selection and Error Metrics

| Metric | Value |
|---|---:|
| Selected tolerance | `1.00000000e-02` |
| Training RMSE | `3.22473046e+00` |
| Validation RMSE | `3.17183285e+00` |
