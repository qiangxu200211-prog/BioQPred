"""Perturb completed hard-label models, recomputing all affected input branches."""

import argparse
import csv
import gc
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from run_controlled_experiments import ContextCache, ROOT, build_model, evaluate, read_rows, save_json, sha, tensors
from run_spliceai_reference import infer_anchor


CONDITIONS = {
    "unmodified": [],
    "both_candidate_cores_4N": [298, 299, 301, 302],
    "matched_flanks_4N": [278, 279, 321, 322],
    "central_11N": list(range(295, 306)),
}


def masked_rows(rows, positions):
    if len(set(positions)) != len(positions) or any(not 0 <= index < 600 for index in positions):
        raise ValueError("Invalid fixed masking coordinates")
    result = []
    for row in rows:
        sequence = list(row["sequence"].upper())
        if len(sequence) != 600:
            raise ValueError("Masking protocol requires the fixed 600-bp benchmark")
        for position in positions:
            sequence[position] = "N"
        result.append(dict(row, sequence="".join(sequence)))
    return result


def check_alignment(task, train):
    _, alignment = infer_anchor(train, "AG" if task == "splice_sites_acceptor" else "GT")
    peaks = alignment["zero_based_core_starts"]
    positions = {peaks["forward"], peaks["forward"] + 1, 599 - peaks["reverse"], 598 - peaks["reverse"]}
    if positions != set(CONDITIONS["both_candidate_cores_4N"]):
        raise ValueError("Training-derived core coordinates differ from the frozen masking protocol")
    return alignment


def write_predictions(path, rows, labels, probabilities):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label", "pred", "prob_class_0", "prob_class_1"])
        for row, label, ps in zip(rows, labels, probabilities):
            writer.writerow([row["id"], int(label), int(ps.argmax())] + ps.tolist())


def verify_clean_prediction_match(path, rows, labels, probabilities):
    with path.open() as handle:
        saved = list(csv.DictReader(handle))
    if [r["id"] for r in saved] != [r["id"] for r in rows] or [int(r["label"]) for r in saved] != labels.tolist():
        raise ValueError("Stored clean prediction IDs or labels do not align")
    expected = np.array([[float(r["prob_class_0"]), float(r["prob_class_1"])] for r in saved])
    max_difference = float(np.max(np.abs(expected - probabilities)))
    if max_difference > 1e-4 or not np.array_equal(expected.argmax(1), probabilities.argmax(1)):
        raise ValueError(f"Clean replay changed before masking: max probability difference {max_difference}")
    return max_difference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=("splice_sites_acceptor", "splice_sites_donor"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    args.device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "protocol.json").exists():
        raise ValueError("Preserve existing masking outputs; use a fresh directory")
    inputs, selected = [], []
    for path in args.inputs.rglob("result.json"):
        result = json.loads(path.read_text())
        spec = result["spec"]
        if spec["task"] not in args.tasks or spec["variant"] not in ("full", "local_comp"):
            continue
        if result.get("status") != "success" or result.get("test") is None or spec["phase"] != "ablation" or spec["teacher"] or spec["channels"] != 128 or spec["kernels"] != [3, 5, 7, 11, 15]:
            raise ValueError("Masking inputs must be completed standard hard-label component controls")
        if sha(path.with_name("best.pt")) != result["checkpoint_sha256"] or sha(path.with_name("test_predictions.csv")) != result["predictions_sha256"]:
            raise ValueError("Input model or clean prediction checksum changed")
        if spec["source_files"]["runner"] != sha(ROOT / "revision_20261022/run_controlled_experiments.py") or spec["source_files"]["model"] != sha(ROOT / "src/models/bioqpred_v2.py"):
            raise ValueError("The original controlled model sources changed")
        selected.append((path, result))
        inputs.append({"result": str(path), "sha256": sha(path), "checkpoint_sha256": result["checkpoint_sha256"], "clean_prediction_sha256": result["predictions_sha256"]})
    expected = {(task, variant, seed) for task in args.tasks for variant in ("full", "local_comp") for seed in (42, 2024, 3407)}
    if {(r["spec"]["task"], r["spec"]["variant"], r["spec"]["seed"]) for _, r in selected} != expected or len(selected) != len(expected):
        raise ValueError("Masking requires all three seeds for both declared input architectures")
    protocol = {"phase": "formal_perturbation", "script_sha256": sha(Path(__file__)), "input_models": inputs, "positions_zero_based": CONDITIONS, "primary_contrast": "Both candidate cores versus the equal-count flanking control; the 11-N central window is secondary.", "masking": "Replace the same fixed coordinates by N in EVERY sample, regardless of label, sequence motif presence, score or strand.", "branches": "Recompute base IDs, 12 composition values and frozen DNABERT mean+max from each perturbed sequence. Content-addressed cache keys include the complete perturbed sequence.", "selection": "No training, threshold selection or checkpoint reselection. Evaluate all predeclared perturbations and seeds.", "amp": args.amp, "torch": torch.__version__, "limitations": ["These are new hard-label model controls, not reproduction of manuscript Table 11 or the old distillation models.", "Four fixed bases are masked to cover both possible orientations; this is not a two-base transcript-oriented intervention.", "N masking is an out-of-distribution sensitivity probe and does not establish biological causality or a particular mechanism.", "Concurrent GPU use may affect elapsed time; no computational-efficiency claim is made.", "The CNN baseline perturbation remains a separate pending comparison."]}
    save_json(args.output / "protocol.json", protocol)
    cache = ContextCache(ROOT / "models/dnabert2", ROOT / "revision_20261022/context_cache", args.device, 16)
    all_results = []
    for task in args.tasks:
        directory = ROOT / "data/splits/nt_tasks" / task
        train, test = read_rows(directory / "train.jsonl"), read_rows(directory / "test.jsonl")
        alignment = check_alignment(task, train)
        protocol.setdefault("train_only_alignment", {})[task] = alignment
        protocol.setdefault("split_sha256", {})[task] = {split: sha(directory / (split + ".jsonl")) for split in ("train", "dev", "test")}
        task_models = [(path, result) for path, result in selected if result["spec"]["task"] == task]
        for condition, positions in CONDITIONS.items():
            perturbed = masked_rows(test, positions)
            context, event = cache.get(task, "test" if not positions else "test_mask_" + condition, perturbed)
            for path, original in task_models:
                spec = original["spec"]
                model = build_model(spec["variant"], 600, 2, 128, spec["kernels"]).to(args.device)
                model.load_state_dict(torch.load(path.with_name("best.pt"), map_location=args.device, weights_only=False)["model_state_dict"])
                loader = DataLoader(tensors(perturbed, 600, context if spec["variant"] == "full" else np.zeros((len(test), 1), dtype="float32")), batch_size=16, shuffle=False)
                metrics, labels, probabilities = evaluate(model, loader, args.device, args.amp)
                row = {"task": task, "variant": spec["variant"], "seed": spec["seed"], "condition": condition, "rows": len(test), "MCC": metrics["MCC"], "accuracy": metrics["accuracy"], "delta_MCC_vs_unmodified": metrics["MCC"] - original["test"]["MCC"], "mean_positive_score_true_positive_class": float(probabilities[labels == 1, 1].mean()), "mean_positive_score_negative_class": float(probabilities[labels == 0, 1].mean()), "source_checkpoint_sha256": original["checkpoint_sha256"], "context_cache": event}
                if condition == "unmodified":
                    row["clean_replay_max_probability_difference"] = verify_clean_prediction_match(path.with_name("test_predictions.csv"), test, labels, probabilities)
                prediction = args.output / f"{task}_{spec['variant']}_seed{spec['seed']}_{condition}_predictions.csv"
                write_predictions(prediction, test, labels, probabilities)
                row["prediction_file"] = str(prediction)
                row["prediction_sha256"] = sha(prediction)
                all_results.append(row)
                save_json(args.output / "per_seed_results.json", all_results)
                print(json.dumps({k: row[k] for k in ("task", "variant", "seed", "condition", "MCC", "delta_MCC_vs_unmodified")}), flush=True)
                del model, loader
                gc.collect()
                if args.device.type == "cuda":
                    torch.cuda.empty_cache()
    groups = defaultdict(list)
    for row in all_results:
        groups[(row["task"], row["variant"], row["condition"])].append(row)
    summaries = []
    for (task, variant, condition), values in sorted(groups.items()):
        if len(values) != 3 or {r["seed"] for r in values} != {42, 2024, 3407}:
            raise ValueError("Incomplete perturbation seed coverage")
        summaries.append({"task": task, "variant": variant, "condition": condition, "MCC_mean": statistics.mean(r["MCC"] for r in values), "MCC_sample_SD": statistics.stdev(r["MCC"] for r in values), "delta_MCC_mean": statistics.mean(r["delta_MCC_vs_unmodified"] for r in values), "delta_MCC_sample_SD": statistics.stdev(r["delta_MCC_vs_unmodified"] for r in values)})
    save_json(args.output / "summary.json", summaries)
    save_json(args.output / "protocol.json", protocol)
    save_json(args.output / "status.json", {"state": "complete", "evaluations": len(all_results), "scope": "Declared hard-label architectures/tasks only, not all manuscript masking comparisons."})


if __name__ == "__main__":
    main()
