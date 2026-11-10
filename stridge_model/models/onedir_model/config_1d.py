import argparse
import os


def initialize_config(argv=None):
	parser = argparse.ArgumentParser(
		description="Identify a one-directional flow equation from NPZ snapshots."
	)
	parser.add_argument("--lam", type=float, default=1e-6)
	parser.add_argument("--tol_min", type=float, default=1e-8)
	parser.add_argument("--tol_max", type=float, default=1e-2)
	parser.add_argument("--num_tol", type=int, default=40)
	parser.add_argument("--l0_penalty", type=float, default=1e-3)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--data_start", type=int, default=3000)
	parser.add_argument(
		"--input_directories",
		nargs="+",
		default=[os.path.join(os.path.dirname(__file__), "one_dir_flow_output_npz")],
		help="directories containing run folders with data_t=*.npz snapshots",
	)
	parser.add_argument(
		"--crop_t", type=int, default=1,
		help="discard this many time layers from each temporal boundary",
	)
	parser.add_argument(
		"--crop_y", type=int, default=2,
		help="discard this many transverse points near each wall",
	)
	parser.add_argument("--rho", type=float, default=10.0)
	parser.add_argument("--viscosity", type=float, default=10.0)
	parser.add_argument("--force", type=float, default=10.0, help="constant forcing in the 1D equation")
	parser.add_argument(
		"--report",
		type=str,
		default=os.path.join(os.path.dirname(__file__), "stridge_model_1d_report.md"),
		help="path for the generated Markdown report",
	)
	args = parser.parse_args(argv)

	if args.lam < 0 or args.l0_penalty < 0:
		parser.error("--lam and --l0_penalty must be non-negative")
	if args.tol_min <= 0 or args.tol_max <= args.tol_min:
		parser.error("--tol_min must be positive and smaller than --tol_max")
	if args.num_tol < 1 or args.data_start < 0:
		parser.error("--num_tol must be positive and --data_start must be non-negative")
	if args.crop_t < 0 or args.crop_y < 0:
		parser.error("crop values must be non-negative")
	if args.rho <= 0 or args.viscosity < 0:
		parser.error("--rho must be positive and --viscosity must be non-negative")

	args.nu = args.viscosity / args.rho
	return args
