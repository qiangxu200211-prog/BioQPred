"""Rebuild traceable high-positive-score PFMs without motif-selected orientation."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from run_spliceai_reference import RC, infer_anchor, read_rows


BASES = "ACGTN"
THRESHOLDS = (0.90, 0.95, 0.99)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_predictions(predictions, rows):
    truth = {r["id"]: r["label"] for r in rows}
    if len(truth) != len(rows) or len(predictions) != len(rows) or len({r["id"] for r in predictions}) != len(rows):
        raise ValueError("Missing or duplicated prediction/sample IDs")
    for row in predictions:
        p0, p1 = float(row["prob_class_0"]), float(row["prob_class_1"])
        if truth.get(row["id"]) != int(row["label"]) or not np.isfinite([p0, p1]).all() or min(p0, p1) < 0 or max(p0, p1) > 1 or abs(p0 + p1 - 1) > 0.002:
            raise ValueError("Invalid prediction label or native stored probability")
        if int(row["pred"]) != int(p1 > p0):
            raise ValueError("Prediction does not agree with probability argmax")


def pfm(rows, core_start, reverse=False):
    counts = np.zeros((11, 5), dtype=np.int64)
    for row in rows:
        seq = row["sequence"].upper()
        if reverse:
            seq = seq.translate(RC)[::-1]
        window = seq[core_start - 5:core_start + 6]
        if len(window) != 11 or set(window) - set(BASES):
            raise ValueError("Invalid candidate-aligned PFM window")
        for index, base in enumerate(window):
            counts[index, BASES.index(base)] += 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "protocol.json").exists():
        raise ValueError("Preserve the existing PFM output; use a fresh output directory")
    with (args.root / "reports/bioqpred_v2_summary.csv").open() as handle:
        variants = {r["task"]: r["selected_variant"] for r in csv.DictReader(handle)}
    matrices, cohorts, membership, inputs, alignments = [], [], [], [], {}
    for task, motif in (("splice_sites_acceptor", "AG"), ("splice_sites_donor", "GT")):
        directory = args.root / "data/splits/nt_tasks" / task
        train = read_rows(directory / "train.jsonl")
        test = read_rows(directory / "test.jsonl")
        _, alignment = infer_anchor(train, motif)
        alignments[task] = alignment
        by_id = {r["id"]: r for r in test}
        for split in ("train", "test"):
            inputs.append({"path": str(directory / (split + ".jsonl")), "sha256": sha(directory / (split + ".jsonl"))})
        paths = sorted((args.root / "runs/bioqpred_v2/predictions").glob(f"{task}_{variants[task]}_seed*_test_predictions.csv"))
        seeds = []
        for path in paths:
            with path.open() as handle:
                predictions = list(csv.DictReader(handle))
            validate_predictions(predictions, test)
            seed = int(predictions[0]["seed"])
            if {int(r["seed"]) for r in predictions} != {seed}:
                raise ValueError("Prediction seed metadata is inconsistent")
            seeds.append(seed)
            inputs.append({"path": str(path), "sha256": sha(path)})
            for threshold in THRESHOLDS:
                chosen = [r for r in predictions if int(r["pred"]) == 1 and float(r["prob_class_1"]) >= threshold]
                ids = [r["id"] for r in chosen]
                cohorts.append({"task": task, "seed": seed, "threshold": threshold, "rows": len(chosen), "true_positive_rows": sum(int(r["label"]) == 1 for r in chosen), "false_positive_rows": sum(int(r["label"]) == 0 for r in chosen), "selected_id_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
                membership.extend({"task": task, "seed": seed, "threshold": threshold, "id": r["id"], "label": int(r["label"]), "native_positive_score": float(r["prob_class_1"])} for r in chosen)
                for frame in ("forward", "reverse"):
                    for cohort, selected in (("all_test_background", test), ("high_positive_score", [by_id[sid] for sid in ids])):
                        counts = pfm(selected, alignment["zero_based_core_starts"][frame], reverse=frame == "reverse")
                        for position, values in enumerate(counts):
                            matrices.append({"task": task, "seed": seed, "threshold": threshold, "cohort": cohort, "frame": frame, "relative_position": position - 5, "rows": len(selected), **{f"count_{base}": int(value) for base, value in zip(BASES, values)}, **{f"frequency_{base}": float(value / len(selected)) if selected else "" for base, value in zip(BASES, values)}})
        if set(seeds) != {42, 2024, 3407} or len(seeds) != 3:
            raise ValueError("PFMs require all three preserved selected-model seeds")
    write_csv(args.output / "pfm_counts_and_frequencies.csv", matrices)
    write_csv(args.output / "cohort_counts.csv", cohorts)
    if membership:
        write_csv(args.output / "selected_samples.csv", membership)
    protocol = {"status": "success", "script_sha256": sha(Path(__file__)), "thresholds": THRESHOLDS, "primary_threshold": 0.95, "selection": "Stored native positive-class score >= threshold and predicted positive, regardless of ground truth; no calibration or test-driven threshold search.", "alignment": alignments, "orientation": "Every selected sample is shown in BOTH fixed frames, using core starts inferred from training positives. No test-motif-based strand choice or filtering is used.", "denominator": "All selected samples; N is retained as a fifth category. Empty cohorts have blank frequencies.", "inputs": inputs, "limitations": ["These are new traceable descriptive PFMs, not confirmation of untraced manuscript Tables 12/13.", "Mixed strands remain mixed within each fixed frame; canonical enrichment must not be manufactured by selecting a motif-bearing orientation.", "High prediction score is not calibrated confidence or evidence of a learned biological mechanism.", "Ground-truth counts describe the selected cohort but do not determine membership."]}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    print(json.dumps(cohorts, indent=2), flush=True)


if __name__ == "__main__":
    main()
