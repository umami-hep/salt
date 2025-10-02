"""Inspect metadata embedded in an ONNX model.

This script extracts and prints JSON metadata stored in a custom key inside a
`.onnx` model file.

Typical usage
-------------
$ python get_onnx_metadata.py /path/to/model.onnx --key gnn_config
"""

import argparse
import json

import onnx
import onnxruntime as ort


def parse_args(args: list[str] | None):
    """Parse command-line arguments.

    Parameters
    ----------
    args : list[str] | None
        List of arguments to parse (e.g. ``sys.argv[1:]``). If ``None``, arguments
        are taken from the command line.

    Returns
    -------
    argparse.Namespace
        Parsed arguments containing:
        - file (str): Path to the ONNX model file.
        - key (str): Metadata key to extract.
    """
    parser = argparse.ArgumentParser(
        description="Get metadata from a .onnx file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        dest="file",
        type=str,
        help="Path to ONNX file.",
    )
    parser.add_argument(
        "-k",
        "--key",
        type=str,
        help="Custom metadata key to extract.",
        default="gnn_config",
    )

    return parser.parse_args(args)


def main(args: list[str] | None = None):
    """Main entry point for extracting metadata from an ONNX file.

    Parameters
    ----------
    args : list[str] | None, optional
        List of arguments to parse (e.g. ``sys.argv[1:]``). If ``None``,
        arguments are read from the command line.
    """
    parsed_args = parse_args(args)

    # Load and check ONNX model
    onnx_model = onnx.load(parsed_args.file)
    onnx.checker.check_model(onnx_model)

    # Create ONNX Runtime session with optimizations disabled
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        parsed_args.file, sess_options, providers=ort.get_available_providers()
    )

    # Extract metadata
    meta = sess.get_modelmeta()
    name = meta.description
    config = meta.custom_metadata_map[parsed_args.key]
    info = json.loads(config)
    info = json.dumps(info, indent=1)

    print(f"Printing config for model: '{name}' and key: '{parsed_args.key}'")
    print(info)


if __name__ == "__main__":
    main()
