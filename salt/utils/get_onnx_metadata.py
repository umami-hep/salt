import argparse
import json

import onnx
import onnxruntime as ort


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Get metdata from a .onnx file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        dest="file",
        type=str,
        help="Path to onnx file.",
    )
    parser.add_argument(
        "-k",
        "--key",
        type=str,
        help="Get config from the specified key.",
        default="gnn_config",
    )

    return parser.parse_args(args)


def main(args=None):
    args = parse_args(args)

    onnx_model = onnx.load(args.file)
    onnx.checker.check_model(onnx_model)

    sess = ort.InferenceSession(args.file, providers=ort.get_available_providers())
    meta = sess.get_modelmeta()
    name = meta.description
    config = meta.custom_metadata_map[args.key]
    info = json.loads(config)
    info = json.dumps(info, indent=1)

    print(f"Printing config for model: '{name}' and key: '{args.key}'")
    print(info)


if __name__ == "__main__":
    main()
