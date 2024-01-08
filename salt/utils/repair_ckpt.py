"""Repair checkpoint files after compilation.

Adapted from:
https://github.com/pytorch/pytorch/issues/101107#issuecomment-1801128683
"""

import argparse
import shutil

import torch


def repair_checkpoint(path: str) -> None:
    """Repair a PyTorch checkpoint file by removing specific prefixes from state_dict keys.

    Parameters
    ----------
    path : str
        The path to the checkpoint file to repair.
    """
    shutil.copyfile(path, path + ".bak")

    ckpt = torch.load(path)
    in_state_dict = ckpt["state_dict"]

    pairings = [(src_key, src_key.replace("_orig_mod.", "")) for src_key in in_state_dict]

    if all(src_key == dest_key for src_key, dest_key in pairings):
        print(f"No need to repair {path}")
        return

    out_state_dict = {}
    for src_key, dest_key in pairings:
        print(f"{src_key}  ==>  {dest_key}")
        out_state_dict[dest_key] = in_state_dict[src_key]

    ckpt["state_dict"] = out_state_dict
    torch.save(ckpt, path)
    print(f"Repaired {path}")


def main(args=None):
    parser = argparse.ArgumentParser(description="Repair PyTorch checkpoint files.")
    parser.add_argument("paths", nargs="+", help="The file paths of the checkpoints to repair.")
    args = parser.parse_args(args)
    for path in args.paths:
        repair_checkpoint(path)


if __name__ == "__main__":
    main()
