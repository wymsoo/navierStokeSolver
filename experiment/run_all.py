import os
from pathlib import Path

import config
from metadata_schema import ExperimentMetadata
from solver_main import FourMovingWalls, LidCavitySolver, Poiseuille, TwoMovingWalls


FLOW_CONFIGS = [
    ("poiseuille", 10.0, 0.0, None),
    ("poiseuille", 0.0, 10.0, None),
    ("twomovingwalls", 10.0, 0.0, None),
    ("twomovingwalls", 0.0, 10.0, None),
    *(('lidcavity', 0.0, 0.0, side) for side in ('N', 'E', 'S', 'W')),
    ("fourmovingwalls", 0.0, 0.0, None),
]


def run_experiment(flow, fx_ext, fy_ext, side=None):
    nx = 81
    ny = 81
    rho = 10
    u_max = 1
    viscosity = 10
    dt = 0.00001
    stopping_bc_conditions = False
    stopping_bc_time = 0.0
    relaxation_time = 0.0

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    experiment_root = Path(os.path.dirname(base_dir)) / "ns_solver" / "stridge_model" / "output_npz"
    experiment_name = f"{flow}_Nx{nx}"
    if side is not None:
        experiment_name = f"{experiment_name}_side{side}"
    experiment_dir = experiment_root / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    existing_runs = [
        path for path in experiment_dir.glob("run_*")
        if path.is_dir() and path.name[4:].isdigit()
    ]
    run_number = max(
        (int(path.name[4:]) for path in existing_runs),
        default=0,
    ) + 1
    run_name = f"run_{run_number:03d}"
    trial = f"{experiment_name}_{run_name}"
    output_dir = experiment_dir / run_name
    output_dir.mkdir()

    config.init(
        nx, ny, trial, base_dir, str(output_dir), dt, rho, u_max, viscosity, fx_ext, fy_ext,
        stopping_bc_conditions_var=stopping_bc_conditions,
        stopping_bc_time_var=stopping_bc_time,
        relaxation_time_var=relaxation_time,
    )

    ExperimentMetadata(
        run_number=run_number,
        run_name=run_name,
        output_dir=str(output_dir),
        flow=flow,
        Nx=nx,
        Ny=ny,
        Fx_ext=fx_ext,
        Fy_ext=fy_ext,
        rho=rho,
        u_max=u_max,
        viscosity=viscosity,
        dt_initial=dt,
        stopping_bc_conditions=stopping_bc_conditions,
        stopping_bc_time=stopping_bc_time,
        relaxation_time=relaxation_time,
    ).append_to(Path(__file__).with_name("metadata.json"))

    match flow:
        case "poiseuille":
            solver = Poiseuille()
        case "twomovingwalls":
            solver = TwoMovingWalls()
        case "lidcavity":
            solver = LidCavitySolver(side=side)
        case "fourmovingwalls":
            solver = FourMovingWalls()
        case _:
            raise ValueError(f"Unsupported flow: {flow}")

    print(f"Starting {trial}; Fx={fx_ext}, Fy={fy_ext}, side={side}")
    solver.run()


def main():
    for flow_config in FLOW_CONFIGS:
        run_experiment(*flow_config)


if __name__ == "__main__":
    main()