"""Compare the predictions of two taggers.

You can compare two models in the same file, or between two files.

Example usage:
    - export a model using to_onnx
    - dump using the tdd (use the -p flag)
    - evaluate the model using salt on the dumped file
    - compare outputs from the tdd and salt evaluation

For more info about model validation, see the docs:
https://ftag-salt.docs.cern.ch/export/#athena-validation

"""

import argparse
from pathlib import Path

import numpy as np
from ftag import Cuts
from ftag.hdf5 import H5Reader


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Script to compare predictions between two models.",
    )

    parser.add_argument(
        "--file_A", required=True, type=Path, help="Path to first evaluation h5 file."
    )
    parser.add_argument(
        "--file_B",
        required=False,
        type=Path,
        help="Path to second evaluation h5 file (defaults to file_A if not provided).",
    )
    parser.add_argument(
        "--tagger_A",
        required=True,
        type=str,
        help="Name of the first tagger.",
    )
    parser.add_argument(
        "--tagger_B",
        required=False,
        type=str,
        help="Name of the second tagger (defaults to tagger_A if not provided).",
    )
    parser.add_argument(
        "--gaussian_regression",
        help="Export gaussian regression model.",
        action="store_true",
    )
    parser.add_argument(
        "--vars",
        default=["b", "c", "u"],
        nargs="+",
        help="Which variables to compare (defaults to pb, pc, pu).",
    )
    parser.add_argument(
        "--cuts",
        default=["n_tracks <= 40"],
        nargs="+",
        help="Which cuts to apply (defaults to 'n_tracks <= 40').",
    )
    parser.add_argument(
        "-n",
        "--num",
        default=-1,
        type=float,
        help="Compare this many jets (defaults to all).",
    )

    args = parser.parse_args(args)
    args.file_B = args.file_B or args.file_A
    args.tagger_B = args.tagger_B or args.tagger_A

    return args


def main(args=None):
    args = parse_args(args)
    if args.file_A == args.file_B and args.tagger_A == args.tagger_B:
        raise ValueError("Attempted to compare the same model!")

    if args.gaussian_regression:
        vars_A = [f"{args.tagger_A}_{v}" for v in args.vars]
        vars_B = [f"{args.tagger_B}_{v}" for v in args.vars]
    else:
        vars_A = [f"{args.tagger_A}_p{v}" for v in args.vars]
        vars_B = [f"{args.tagger_B}_p{v}" for v in args.vars]

    cuts = Cuts.from_list(args.cuts)

    # load the data
    reader_A = H5Reader(args.file_A)
    reader_B = H5Reader(args.file_B)
    jets_A = reader_A.load({"jets": vars_A}, cuts=cuts, num_jets=args.num)["jets"]
    jets_B = reader_B.load({"jets": vars_B}, cuts=cuts, num_jets=args.num)["jets"]

    diff_regions = np.array([
        [float(f"1e-{i}"), float(f"5e-{i}")] for i in range(7, 1, -1)
    ]).flatten()

    print(f'\nComparing {len(jets_A):,} jets from "{args.file_A}" and "{args.file_B}"...\n')
    for var_A, var_B in zip(vars_A, vars_B, strict=True):
        array_A = jets_A[var_A]
        array_B = jets_B[var_B]
        if array_A.dtype != np.float32:
            raise ValueError(
                f"{var_A} in {args.file_A} is not float32. Please compare at full precision."
            )
        if array_B.dtype != np.float32:
            raise ValueError(
                f"{var_B} in {args.file_B} is not float32. Please compare at full precision."
            )

        # make sure there are no NaNs
        if np.isnan(array_A).any():
            raise ValueError(f"{var_A} in {args.file_A} contains NaNs.")
        if np.isnan(array_B).any():
            raise ValueError(f"{var_B} in {args.file_B} contains NaNs.")

        print(f"Comparing {var_A} and {var_B}...")
        diff = abs(array_A - array_B)
        for diff_region in diff_regions:
            selected = diff[diff > diff_region]
            pct = len(selected) / len(diff)
            print(f"Differences of {diff_region:.1e}: {pct:.2%}")
            if pct == 0:
                break
        print(f"max {diff.max():.2e} | mean {diff.mean():.2e} | median {np.median(diff):.2e}")
        print()

        np.testing.assert_allclose(array_A, array_B, rtol=1e-4, atol=1e-6)


if __name__ == "__main__":
    main()
