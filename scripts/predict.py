"""Predict using a trusted controlled-ablation or teacher-student checkpoint."""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "revision_20261022"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="JSONL containing id and sequence (labels not needed)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    import numpy as np
    import torch
    import run_controlled_experiments as control
    from src.models.bioqpred_v2 import encode_base_ids, compute_composition_features

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.output.exists():
        raise ValueError("Refusing to overwrite predictions")
    # Use only checkpoints you trained or independently trust.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "model_state_dict" not in checkpoint:
        raise ValueError("Expected a controlled-ablation or teacher-student checkpoint, not a fair-baseline checkpoint")
    spec = checkpoint["spec"]
    task = spec["task"]
    variant = spec["variant"]
    length, classes = (1000, 2) if task == "H3K27me3" else (600, 3 if task == "splice_sites_all" else 2)
    rows = control.read_rows(args.input)
    if not rows or any(not isinstance(r.get("id"), str) or len(r.get("sequence", "")) != length or set(r["sequence"].upper()) - set("ACGTN") for r in rows):
        raise ValueError("Expected nonempty ID/sequence rows with the exact task window and A/C/G/T/N alphabet")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Prediction IDs must be unique")
    device = torch.device(args.device)
    model = control.build_model(variant, length, classes, spec["channels"], spec["kernels"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    sequences = [r["sequence"].upper() for r in rows]
    bases = encode_base_ids(sequences, length)
    comp = compute_composition_features(sequences)
    pooled = np.zeros((len(rows), 1), dtype="float32")
    if control.flags(variant)["use_bert"]:
        cache = control.ContextCache(ROOT / "models/dnabert2", ROOT / "revision_20261022/context_cache", device, args.batch_size)
        pooled, _ = cache.get(task, "user_predictions", rows)
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            end = start + args.batch_size
            logits = model({}, bases[start:end].to(device), comp[start:end].to(device), bert_pooled=torch.from_numpy(pooled[start:end]).to(device))
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite prediction output")
            predictions.extend(logits.softmax(-1).cpu().tolist())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "prediction"] + [f"probability_{i}" for i in range(classes)])
        for row, values in zip(rows, predictions):
            writer.writerow([row["id"], int(np.argmax(values))] + values)
    print(f"Wrote {len(rows)} predictions. No evaluation labels were consumed.")


if __name__ == "__main__":
    main()
