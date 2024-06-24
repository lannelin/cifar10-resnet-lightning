"""
heavily borrows from https://github.com/huggingface/safetensors/blob/v0.4.3/bindings/python/convert.py # noqa: E501

changes:
 - works on local file only
 - no PR creation etc.
"""

import argparse
import os

import torch
from safetensors.torch import _remove_duplicate_names, load_file, save_file


def confirm_continue(warning):
    answer = input(warning + " Continue [Y/n] ? ").strip()

    if answer not in ["Y", "n"]:
        print(f"{answer} is an invalid choice, please select Y/n")
        return confirm_continue()

    return answer == "Y"  # false if n


def check_file_size(sf_filename: str, pt_filename: str):
    sf_size = os.stat(sf_filename).st_size
    pt_size = os.stat(pt_filename).st_size

    if (sf_size - pt_size) / pt_size > 0.01:
        raise RuntimeError(
            f"""The file size different is more than 1%:
            - {sf_filename}: {sf_size}
            - {pt_filename}: {pt_size}
            """
        )


def check_tensors_match(
    loaded: dict[torch.Tensor], reloaded: dict[torch.Tensor]
):
    for k in loaded:
        pt_tensor = loaded[k]
        sf_tensor = reloaded[k]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f"The output tensors do not match for key {k}")


def convert_ckpt_to_safetensors(
    pt_filename: str,
    sf_filename: str,
    discard_names: list[str],
):
    metadata = {"format": "pt"}

    loaded = torch.load(pt_filename, map_location="cpu")

    if "state_dict" in loaded:
        loaded = loaded["state_dict"]
    to_removes = _remove_duplicate_names(loaded, discard_names=discard_names)

    for kept_name, to_remove_group in to_removes.items():
        for to_remove in to_remove_group:
            if to_remove not in metadata:
                metadata[to_remove] = kept_name
            del loaded[to_remove]
    # Force tensors to be contiguous
    loaded = {k: v.contiguous() for k, v in loaded.items()}

    dirname = os.path.dirname(sf_filename)
    os.makedirs(dirname, exist_ok=True)
    save_file(loaded, sf_filename, metadata=metadata)
    check_file_size(sf_filename, pt_filename)
    reloaded = load_file(sf_filename)
    check_tensors_match(loaded=loaded, reloaded=reloaded)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Convert local ckpt file to safetensors. \
Lossy - only designed to maintain state_dict."""
    )
    parser.add_argument(
        "--infile",
        type=str,
        help="Input .ckpt filepath",
        required=True,
    )
    parser.add_argument(
        "--outfile",
        type=str,
        help="""Output .safetensors filepath""",
        required=True,
    )
    args = parser.parse_args()

    if args.infile.endswith(".ckpt") and args.outfile.endswith(".safetensors"):
        warning = """This conversion script will unpickle a pickled file,\
which is inherently unsafe.
If you do not trust this file, we invite you to use \
https://huggingface.co/spaces/safetensors/convert or google colab or \
other hosted solution to avoid potential issues with this file."""
        # confirmation = confirm_continue(warning)
        # if not confirmation:
        #     exit("Exiting due to user choice.")

        convert_ckpt_to_safetensors(
            pt_filename=args.infile,
            sf_filename=args.outfile,
            discard_names=[],
        )
    else:
        raise Exception(
            "expected infile to be a '.ckpt' file"
            " and outfile to be '.safetensors' file"
        )
