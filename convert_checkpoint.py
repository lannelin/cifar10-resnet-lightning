"""
heavily borrows from https://github.com/huggingface/safetensors/blob/v0.4.3/bindings/python/convert.py # noqa: E501

changes:
 - works on local file only
 - no PR creation etc.
 - convert to *and* from safetensors
 - add pytorch-lightning_version to metadata with unsafe confirm
"""

import argparse
import json
import os
from collections import OrderedDict

import torch
from safetensors import safe_open
from safetensors.torch import _remove_duplicate_names, load_file, save_file

CKPT_METADATA_KEYS = [
    "pytorch-lightning_version",
]


# from https://github.com/huggingface/safetensors/issues/194#issuecomment-1466496698 # noqa: E501
def load_metadata(data):
    n_header = data[:8]
    n = int.from_bytes(n_header, "little")
    metadata_bytes = data[8 : 8 + n]
    header = json.loads(metadata_bytes)
    return header.get("__metadata__", {})


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

    for k in CKPT_METADATA_KEYS:
        if k in loaded:
            val = str(loaded[k])
            # TODO review assumption of unsafe str
            if confirm_continue(
                warning=f"adding metadata key: {k} "
                f"with value: '{val}' . Does this look safe and reasonable?"
            ):
                metadata[k] = val

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


def convert_safetensors_to_ckpt(
    sf_filename: str,
    pt_filename: str,
):
    obj = {}
    state_dict = OrderedDict()
    with safe_open(sf_filename, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

        # load metadata
        metadata = f.metadata()
        for k in CKPT_METADATA_KEYS:
            if k in metadata:
                val = metadata[k]
                # TODO review assumption of unsafe str
                if confirm_continue(
                    warning=f"adding key from metadata: {k} "
                    f"with value: '{val}' ."
                    " Does this look safe and reasonable?"
                ):
                    obj[k] = val

    obj["state_dict"] = state_dict
    torch.save(obj, pt_filename)

    check_file_size(sf_filename, pt_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Convert local ckpt file to/from safetensors. \
Only designed to maintain state_dict"""
    )
    parser.add_argument(
        "--infile",
        type=str,
        help="Input filepath (either a .ckpt file or a .safetensors)",
        required=True,
    )
    parser.add_argument(
        "--outfile",
        type=str,
        help="""Output filepath \
(either a .ckpt file or a .safetensors, depending on input)""",
        required=True,
    )
    args = parser.parse_args()

    if args.infile.endswith(".ckpt") and args.outfile.endswith(".safetensors"):
        warning = """This conversion script will unpickle a pickled file,\
which is inherently unsafe.
If you do not trust this file, we invite you to use \
https://huggingface.co/spaces/safetensors/convert or google colab or \
other hosted solution to avoid potential issues with this file."""
        confirmation = confirm_continue(warning)
        if not confirmation:
            exit("Exiting due to user choice.")

        convert_ckpt_to_safetensors(
            pt_filename=args.infile,
            sf_filename=args.outfile,
            discard_names=[],
        )
    elif args.infile.endswith(".safetensors") and args.outfile.endswith(
        ".ckpt"
    ):
        convert_safetensors_to_ckpt(
            sf_filename=args.infile, pt_filename=args.outfile
        )
    else:
        raise Exception(
            "expected one of infile/outfile to be a '.ckpt' file"
            " and other to be '.safetensors' file"
        )
