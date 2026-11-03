"""Fetch hash-pinned DNABERT-2 weights, then verify the study tensor payload.

Weights are external assets. Only load a pickle after checking the pinned hash;
PyTorch's weights_only=True is used for the official checkpoint.
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_data import download, sha


def verify_tensors(state, expected):
    if set(state) != set(expected):
        raise ValueError("Backbone tensor names differ from the scientific checkpoint")
    for name, tensor in state.items():
        entry = expected[name]
        array = tensor.detach().cpu().contiguous().numpy()
        if list(array.shape) != entry["shape"] or str(array.dtype) != entry["numpy_dtype"]:
            raise ValueError(f"Backbone tensor shape/dtype mismatch: {name}")
        if hashlib.sha256(array.tobytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Backbone tensor content mismatch: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-file", type=Path, help="Use a local model.safetensors instead of downloading")
    args = parser.parse_args()
    import torch
    from safetensors.torch import load_file, save_file

    manifest = json.loads((ROOT / "configs/backbone.json").read_text())
    directory = ROOT / "models/dnabert2"
    for name, expected in manifest["runtime_files"].items():
        if sha(directory / name) != expected:
            raise ValueError(f"Review modified backbone runtime before loading: {name}")
    output = directory / "model.safetensors"
    if output.exists():
        verify_tensors(load_file(str(output)), manifest["tensors"])
        print("Existing backbone tensors verified; nothing downloaded.")
        return
    if args.from_file:
        state = load_file(str(args.from_file))
    else:
        entry = manifest["official_checkpoint"]
        source = ROOT / ".download-cache/pytorch_model.bin"
        url = f"https://huggingface.co/{manifest['repository']}/resolve/{manifest['download_revision']}/pytorch_model.bin"
        download(url, source, entry)
        state = torch.load(source, map_location="cpu", weights_only=True)
    verify_tensors(state, manifest["tensors"])
    temporary = output.with_suffix(".safetensors.part")
    if args.from_file:
        shutil.copyfile(args.from_file, temporary)
    else:
        save_file({k: v.contiguous() for k, v in state.items()}, str(temporary), metadata={"format": "pt"})
    verify_tensors(load_file(str(temporary)), manifest["tensors"])
    temporary.replace(output)
    print("DNABERT-2 weights installed; every tensor matches the study checkpoint.")


if __name__ == "__main__":
    main()
