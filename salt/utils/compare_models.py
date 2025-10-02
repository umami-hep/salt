"""Compare the predictions of two taggers.

You can compare two models in the same file, or between two files.

Example usage
-------------
- Export a model using ``to_onnx``.
- Dump using the ``tdd`` (use the ``-p`` flag).
- Evaluate the model using ``salt`` on the dumped file.
- Compare outputs from the ``tdd`` and ``salt`` evaluation.

For more info about model validation, see the docs:
https://ftag-salt.docs.cern.ch/export/#athena-validation
"""

import argparse
from pathlib import Path

import numpy as np
from ftag import Cuts
from ftag.hdf5 import H5Reader


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for comparing two tagger models.

    Parameters
    ----------
    args : list[str] | None, optional
        List of arguments to parse (defaults to ``sys.argv`` if ``None``).

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes:
        - ``file_a`` (Path): Path to the first evaluation HDF5 file.
        - ``file_b`` (Path): Path to the second evaluation HDF5 file.
        - ``tagger_a`` (str): Name of the first tagger.
        - ``tagger_b`` (str): Name of the second tagger.
        - ``gaussian_regression`` (bool): Whether to export Gaussian regression model.
        - ``vars`` (list of str): Variables to compare.
        - ``cuts`` (list of str): Cuts to apply to the dataset.
        - ``num`` (float): Number of jets to compare.
    """
    parser = argparse.ArgumentParser(
        description="Script to compare predictions between two models.",
    )

    parser.add_argument(
        "--file_a", required=True, type=Path, help="Path to first evaluation h5 file."
    )
    parser.add_argument(
        "--file_b",
        required=False,
        type=Path,
        help="Path to second evaluation h5 file (defaults to file_a if not provided).",
    )
    parser.add_argument(
        "--tagger_a",
        required=True,
        type=str,
        help="Name of the first tagger.",
    )
    parser.add_argument(
        "--tagger_b",
        required=False,
        type=str,
        help="Name of the second tagger (defaults to tagger_a if not provided).",
    )
    parser.add_argument(
        "--gaussian_regression",
        help="Export Gaussian regression model.",
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

    parsed_args = parser.parse_args(args)
    parsed_args.file_b = parsed_args.file_b or parsed_args.file_a
    parsed_args.tagger_b = parsed_args.tagger_b or parsed_args.tagger_a

    return parsed_args


def main(args: list[str] | None = None) -> None:
    """Run comparison between two tagger models' predictions.

    Parameters
    ----------
    args : list[str] | None, optional
        List of arguments to parse (defaults to ``sys.argv`` if ``None``).

    Raises
    ------
    ValueError
        If the same model is compared against itself, or if variables contain
        NaNs, or if the arrays are not float32.
    """
    parsed_args = parse_args(args)
    if parsed_args.file_a == parsed_args.file_b and parsed_args.tagger_a == parsed_args.tagger_b:
        raise ValueError("Attempted to compare the same model!")

    if parsed_args.gaussian_regression:
        vars_a = [f"{parsed_args.tagger_a}_{v}" for v in parsed_args.vars]
        vars_b = [f"{parsed_args.tagger_b}_{v}" for v in parsed_args.vars]
    else:
        vars_a = [f"{parsed_args.tagger_a}_p{v}" for v in parsed_args.vars]
        vars_b = [f"{parsed_args.tagger_b}_p{v}" for v in parsed_args.vars]

    cuts = Cuts.from_list(parsed_args.cuts)

    # load the data
    reader_a = H5Reader(parsed_args.file_a)
    reader_b = H5Reader(parsed_args.file_b)
    jets_a = reader_a.load({"jets": vars_a}, cuts=cuts, num_jets=parsed_args.num)["jets"]
    jets_b = reader_b.load({"jets": vars_b}, cuts=cuts, num_jets=parsed_args.num)["jets"]

    diff_regions = np.array([
        [float(f"1e-{i}"), float(f"5e-{i}")] for i in range(7, 1, -1)
    ]).flatten()

    print(
        f"\nComparing {len(jets_a):,} jets from "
        f'"{parsed_args.file_a}" and "{parsed_args.file_b}"...\n'
    )
    for var_a, var_b in zip(vars_a, vars_b, strict=True):
        array_a = jets_a[var_a]
        array_b = jets_b[var_b]
        if array_a.dtype != np.float32:
            raise ValueError(
                f"{var_a} in {parsed_args.file_a} is not float32. Please compare at full precision."
            )
        if array_b.dtype != np.float32:
            raise ValueError(
                f"{var_b} in {parsed_args.file_b} is not float32. Please compare at full precision."
            )

        # make sure there are no NaNs
        if np.isnan(array_a).any():
            raise ValueError(f"{var_a} in {parsed_args.file_a} contains NaNs.")
        if np.isnan(array_b).any():
            raise ValueError(f"{var_b} in {parsed_args.file_b} contains NaNs.")

        print(f"Comparing {var_a} and {var_b}...")
        diff = abs(array_a - array_b)
        for diff_region in diff_regions:
            selected = diff[diff > diff_region]
            pct = len(selected) / len(diff)
            print(f"Differences of {diff_region:.1e}: {pct:.2%}")
            if pct == 0:
                break
        print(f"max {diff.max():.2e} | mean {diff.mean():.2e} | median {np.median(diff):.2e}")
        print()

        np.testing.assert_allclose(array_a, array_b, rtol=1e-4, atol=1e-6)


if __name__ == "__main__":
    main()
