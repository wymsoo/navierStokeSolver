import json
from dataclasses import asdict, dataclass
from pathlib import Path


# Todo: Add a datetime attribute to ensure distinguishable runs (Sometimes the run number and name will overlap especially if you delete bad datasets)

@dataclass(frozen=True)
class ExperimentMetadata:
    run_number: int
    run_name: str
    output_dir: str
    flow: str
    Nx: int
    Ny: int
    Fx_ext: float
    Fy_ext: float
    rho: float
    u_max: float
    viscosity: float
    dt_initial: float
    stopping_bc_conditions: bool
    stopping_bc_time: float | None
    relaxation_time: float

    def append_to(self, metadata_path: Path) -> None:
        if metadata_path.exists() and metadata_path.stat().st_size > 0:
            with metadata_path.open("r", encoding="utf-8") as metadata_file:
                records = json.load(metadata_file)
        else:
            records = []

        if isinstance(records, dict):
            records = [records]
        elif not isinstance(records, list):
            raise ValueError("The metadata registry must contain a JSON object or array.")

        records.append(asdict(self))
        with metadata_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(records, metadata_file, indent=2)