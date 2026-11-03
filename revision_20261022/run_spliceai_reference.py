"""Evaluate unchanged official SpliceAI weights under an explicit context limit.

The observed-600 setting pads missing flanks with unknown bases and is not a
full-context genome-wide SpliceAI benchmark. Binary tasks only are calibrated.
"""

import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, matthews_corrcoef, roc_auc_score

RC = str.maketrans("ACGTN", "TGCAN")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path, limit=0):
    with path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return rows[:limit] if limit else rows


def infer_anchor(train, motif):
    positives = [r["sequence"].upper() for r in train if r["label"] == 1]
    negatives = [r["sequence"].upper() for r in train if r["label"] == 0]
    if not positives or not negatives:
        raise ValueError("Binary training classes 0 and 1 are required")
    oriented_positives = [(seq, seq.translate(RC)[::-1]) for seq in positives]
    oriented_negatives = [(seq, seq.translate(RC)[::-1]) for seq in negatives]
    peaks = []
    for strand in (0, 1):
        counts = Counter({position: sum(pair[strand][position:position + 2] == motif for pair in oriented_positives) for position in range(295, 305)})
        peaks.append(min(p for p, count in counts.items() if count == max(counts.values())))
    forward_peak, reverse_peak = peaks
    positive_fraction = sum(forward[forward_peak:forward_peak + 2] == motif or reverse[reverse_peak:reverse_peak + 2] == motif for forward, reverse in oriented_positives) / len(positives)
    negative_fraction = sum(forward[forward_peak:forward_peak + 2] == motif or reverse[reverse_peak:reverse_peak + 2] == motif for forward, reverse in oriented_negatives) / len(negatives)
    if positive_fraction < 0.8 or positive_fraction <= negative_fraction:
        raise ValueError("Training motif does not establish the declared positive-class alignment")
    # SpliceAI scores the last exonic donor base and first exonic acceptor base.
    delta = -1 if motif == "GT" else 2
    anchors = {"forward": forward_peak + delta, "reverse": reverse_peak + delta}
    if anchors["forward"] + anchors["reverse"] != 599:
        raise ValueError("Training-derived opposite-strand candidate coordinates are inconsistent")
    return anchors, {"motif": motif, "zero_based_core_starts": {"forward": forward_peak, "reverse": reverse_peak}, "positive_fraction_either_strand": positive_fraction, "negative_fraction_either_strand": negative_fraction}


def encode_observed_600(rows, reverse=False):
    x = np.zeros((len(rows), 10600, 4), dtype="float32")
    lookup = {base: i for i, base in enumerate("ACGT")}
    for index, row in enumerate(rows):
        seq = row["sequence"].upper()
        if len(seq) != 600 or set(seq) - set("ACGTN"):
            raise ValueError("Expected valid 600-bp candidate sequences")
        if reverse:
            seq = seq.translate(RC)[::-1]
        for position, base in enumerate(seq):
            if base != "N":
                x[index, 5000 + position, lookup[base]] = 1
    return x


def threshold_on_dev(labels, scores):
    thresholds = [0.01] + [i / 20 for i in range(1, 20)] + [0.99]
    trials = [{"threshold": threshold, "dev_MCC": float(matthews_corrcoef(labels, scores >= threshold))} for threshold in thresholds]
    chosen = max(trials, key=lambda r: (r["dev_MCC"], -abs(r["threshold"] - 0.5), -r["threshold"]))
    return chosen, trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    model_files = [args.model_dir / f"spliceai{i}.h5" for i in range(1, 6)]
    protocol = {"method": "SpliceAI_official_5model_ensemble_observed600_both_strands", "scope": "Binary candidate classification with 600 observed bp and unknown-base flanks, not standard full-context genome-wide evaluation.", "model_sha256": {p.name: sha(p) for p in model_files}, "tensorflow": tf.__version__, "numpy": np.__version__, "input_shape": [10600, 4], "observed_interval": [5000, 5600], "strand_rule": "Average the five models separately on each strand; take the maximum of the two ensemble strand scores. Gene strand is not supplied to the model.", "threshold_grid": [0.01] + [i / 20 for i in range(1, 20)] + [0.99], "positive_label": 1, "calibration": "Core alignment from training positives only; score threshold from dev MCC only; fixed-0.5 result also retained.", "pretraining_overlap": "Not excluded. Pretrained human SpliceAI is a transfer reference, not evidence of training-data independence.", "phase": "smoke" if args.smoke else "formal_reference", "script_sha256": sha(Path(__file__))}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    for task, motif, channel in (("splice_sites_acceptor", "AG", 1), ("splice_sites_donor", "GT", 2)):
        directory = args.root / "data/splits/nt_tasks" / task
        anchor, alignment = infer_anchor(read_rows(directory / "train.jsonl"), motif)
        task_output = args.output / task
        task_output.mkdir(exist_ok=True)
        splits = ("dev",) if args.smoke else ("dev", "test")
        all_rows, sums = {}, {}
        split_hashes = {}
        for split in splits:
            source = directory / (split + ".jsonl")
            split_hashes[split] = sha(source)
            all_rows[split] = read_rows(source, 8 if args.smoke else 0)
            sums[split] = np.zeros((len(all_rows[split]), 2), dtype="float64")
        start = time.perf_counter()
        for model_file in model_files:
            tf.keras.backend.clear_session()
            model = tf.keras.models.load_model(str(model_file), compile=False)
            for split, rows in all_rows.items():
                score_path = task_output / f"{model_file.stem}_{split}.npy"
                identity = {"split_sha256": split_hashes[split], "rows": len(rows), "model_sha256": protocol["model_sha256"][model_file.name], "anchor": anchor, "channel": channel, "context": protocol["input_shape"], "observed": protocol["observed_interval"], "strand_rule": protocol["strand_rule"], "tensorflow": tf.__version__, "script_sha256": protocol["script_sha256"]}
                metadata_path = score_path.with_suffix(".json")
                if score_path.exists() and metadata_path.exists():
                    saved = json.loads(metadata_path.read_text())
                    if saved["identity"] != identity or sha(score_path) != saved["array_sha256"]:
                        raise ValueError("SpliceAI score cache identity mismatch")
                    scores = np.load(score_path)
                else:
                    scores = np.zeros((len(rows), 2), dtype="float32")
                    for offset in range(0, len(rows), args.batch_size):
                        batch = rows[offset:offset + args.batch_size]
                        encoded = np.concatenate([encode_observed_600(batch), encode_observed_600(batch, reverse=True)])
                        probabilities = model(encoded, training=False).numpy()
                        if probabilities.shape != (len(encoded), 600, 3):
                            raise ValueError(f"Unexpected official model output shape: {probabilities.shape}")
                        if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(-1), 1, atol=1e-5):
                            raise ValueError("Official model probabilities are invalid")
                        scores[offset:offset + len(batch)] = np.stack([probabilities[:len(batch), anchor["forward"], channel], probabilities[len(batch):, anchor["reverse"], channel]], axis=1)
                        if offset % 200 == 0:
                            print(task, model_file.name, split, offset, "/", len(rows), flush=True)
                    np.save(score_path, scores)
                    metadata_path.write_text(json.dumps({"identity": identity, "array_sha256": sha(score_path)}, indent=2))
                if scores.shape != sums[split].shape or not np.isfinite(scores).all():
                    raise ValueError("Invalid SpliceAI cached scores")
                sums[split] += scores / len(model_files)
            del model
        dev_labels = np.asarray([r["label"] for r in all_rows["dev"]])
        selected, trials = threshold_on_dev(dev_labels, sums["dev"].max(axis=1))
        result = {"task": task, "status": "success", "phase": protocol["phase"], "alignment": alignment, "zero_based_score_positions": anchor, "dev_selection": selected, "threshold_trials": trials, "split_sha256": split_hashes, "model_sha256": protocol["model_sha256"], "elapsed_seconds": time.perf_counter() - start, "test": None}
        if not args.smoke:
            rows = all_rows["test"]
            labels = np.asarray([r["label"] for r in rows])
            scores = sums["test"].max(axis=1)
            result["test"] = {"MCC_dev_calibrated": float(matthews_corrcoef(labels, scores >= selected["threshold"])), "MCC_fixed_0_5": float(matthews_corrcoef(labels, scores >= 0.5)), "accuracy_dev_calibrated": float(accuracy_score(labels, scores >= selected["threshold"])), "AUROC": float(roc_auc_score(labels, scores)), "AUPRC": float(average_precision_score(labels, scores)), "rows": len(rows)}
            with (task_output / "test_predictions.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["id", "label", "score", "pred_fixed_0_5", "pred_dev_calibrated"])
                for row, score in zip(rows, scores):
                    writer.writerow([row["id"], row["label"], score, int(score >= 0.5), int(score >= selected["threshold"])])
        (task_output / "result.json").write_text(json.dumps(result, indent=2))
        print("COMPLETE", task, result["test"], flush=True)


if __name__ == "__main__":
    main()
