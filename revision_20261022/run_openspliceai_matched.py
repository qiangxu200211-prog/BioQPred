"""Train the official OpenSpliceAI architecture with matched candidate labels."""

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, matthews_corrcoef

from openspliceai_candidate import (
    COMMIT, CONTEXT, DILATIONS, FILES, TASKS, UPSTREAM, WIDTHS,
    CandidateOpenSpliceAI, encode, infer_candidate, official_class,
)
from run_controlled_experiments import read_rows, save_json, seed_everything, sha
from run_fair_baselines import fit, final_test, freeze_selection

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
METHOD = "openspliceai2000_candidate_from_scratch"
GRID = [1e-4, 3e-4, 1e-3]
SEEDS = [42, 2024, 3407]


def source_hashes():
    names = ["run_openspliceai_matched.py", "openspliceai_candidate.py", "run_fair_baselines.py",
             "fair_baseline_models.py", "run_controlled_experiments.py"]
    files = [HERE / n for n in names] + sorted((ROOT / "src/models").glob("*.py"))
    files += [UPSTREAM / name for name in FILES]
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


class CandidateCase:
    def __init__(self, args, task):
        self.args, self.task, self.classes = args, task, 2
        self.directory = args.split_dir / task
        train = read_rows(self.directory / "train.jsonl")
        if {r["label"] for r in train} != {0, 1}:
            raise ValueError("This adapter accepts only the two binary splice tasks")
        self.anchors, self.alignment = infer_candidate(train, TASKS[task][0])
        self.rows = {"train": train[:64] if args.smoke else train,
                     "dev": read_rows(self.directory / "dev.jsonl", 32 if args.smoke else 0)}
        self.hashes = {s: sha(self.directory / (s + ".jsonl")) for s in ("train", "dev", "test")}
        counts = np.bincount([r["label"] for r in self.rows["train"]], minlength=2)
        if not counts.all():
            raise ValueError("Training subset lacks one of the binary classes")
        self.weights = torch.tensor(len(self.rows["train"]) / (2 * counts), dtype=torch.float32)
        self.base = {}
        self.external_params, self.cache_events = 0, []

    def collate(self, method, split):
        if method != METHOD:
            raise ValueError("Unexpected method")
        if split not in self.rows:
            if self.args.smoke:
                raise ValueError("Smoke tests must not read original test records")
            self.rows[split] = read_rows(self.directory / (split + ".jsonl"))
        if split not in self.base:
            self.base[split] = encode([r["sequence"] for r in self.rows[split]])

        def batch(items):
            indices, rows = zip(*items)
            return (self.base[split][list(indices)], torch.tensor([r["label"] for r in rows]),
                    [r["id"] for r in rows])
        return batch

    def model(self, method, seed):
        if method != METHOD:
            raise ValueError("Unexpected method")
        seed_everything(seed)
        return CandidateOpenSpliceAI(self.task, self.anchors)


def checked_fit(case, seed, lr, args, directory):
    for prior in directory.glob(f"seed{seed}_lr{lr}_*"):
        if not (prior / "fit_result.json").exists():
            raise ValueError("Partial attempt retained; review before restarting: " + str(prior))
    return fit(case, METHOD, seed, lr, args, directory)


def preflight(args):
    official_class()
    result = {"status": "verified_preflight_only", "fits_started": 0,
              "source_sha256": source_hashes(), "tasks": {}}
    for task in TASKS:
        case = CandidateCase(args, task)
        rows = {**case.rows, "test": read_rows(case.directory / "test.jsonl")}
        ids = {s: {str(r["id"]) for r in records} for s, records in rows.items()}
        for split, expected in (("train", 27000), ("dev", 3000), ("test", 3000)):
            if len(rows[split]) != expected or len(ids[split]) != expected:
                raise ValueError("Unexpected original split size or duplicate IDs")
            if any(len(r["sequence"]) != 600 or r["label"] not in (0, 1) for r in rows[split]):
                raise ValueError("Invalid original task records")
        if ids["train"] & ids["dev"] or ids["train"] & ids["test"] or ids["dev"] & ids["test"]:
            raise ValueError("Original IDs overlap across splits")
        model = case.model(METHOD, 42)
        result["tasks"][task] = {"split_sha256": case.hashes, "anchors": case.anchors,
                                 "alignment": case.alignment, "counts": {s: len(v) for s, v in rows.items()},
                                 "trainable_params": sum(p.numel() for p in model.parameters())}
    save_json(args.output / "preflight.json", result)
    return result


def audit(output):
    protocol = json.loads((output / "protocol.json").read_text())
    if protocol["phase"] != "formal" or protocol["source_sha256"] != source_hashes():
        raise ValueError("Only the frozen formal experiment can be audited")
    records = []
    for task in TASKS:
        directory = output / task / METHOD
        selection = json.loads((directory / "selection.json").read_text())
        trials = selection["trials"]
        if sorted(t["lr"] for t in trials) != GRID or not selection["selected_before_test"]:
            raise ValueError("Incomplete dev-only LR grid")
        winner = max(trials, key=lambda t: (t["dev_MCC"], -t["lr"]))
        if winner["lr"] != selection["selected_lr"] or winner["directory"] != selection["selected_directory"]:
            raise ValueError("LR selection did not replay")
        expected_splits = protocol["tasks"][task]["split_sha256"]
        for split, expected in expected_splits.items():
            if sha(ROOT / "data/splits/nt_tasks" / task / (split + ".jsonl")) != expected:
                raise ValueError("Original split changed")
        fits = list(directory.glob("*/fit_result.json"))
        keys = set()
        for path in fits:
            run, saved = path.parent, json.loads(path.read_text())
            spec = saved["spec"]
            key = (spec["seed"], spec["lr"])
            if key in keys:
                raise ValueError("Repeated fitting attempt")
            keys.add(key)
            if (saved["status"] != "success" or spec["source_sha256"] != protocol["source_sha256"]
                    or spec["split_sha256"] != expected_splits or spec["method"] != METHOD
                    or spec["task"] != task or spec["phase"] != "dev_fit"
                    or spec["epochs"] != 15 or spec["patience"] != 4
                    or spec["effective_batch_size"] != 16 or spec["weight_decay"] != 0.01
                    or spec["model_sha256"] != {} or not spec["amp"]):
                raise ValueError("Fit differs from frozen matched protocol")
            if sha(run / "best.pt") != saved["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed")
            history = json.loads((run / "history.json").read_text())
            best = max(history, key=lambda h: h["dev"]["MCC"])
            checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
            if (best["dev"]["MCC"] != saved["best_dev_MCC"] or checkpoint["epoch"] != best["epoch"]
                    or checkpoint["spec"] != spec or len(history) != saved["epochs_completed"]):
                raise ValueError("Checkpoint selection did not replay")
            if spec["seed"] == 42:
                trial = next(t for t in trials if t["lr"] == spec["lr"])
                if trial["dev_MCC"] != saved["best_dev_MCC"] or Path(trial["directory"]) != run:
                    raise ValueError("Search record differs from fit evidence")
            if spec["lr"] != selection["selected_lr"]:
                if (run / "test_result.json").exists() or (run / "test_predictions.csv").exists():
                    raise ValueError("Unselected LR was tested")
                continue
            result = json.loads((run / "test_result.json").read_text())
            if (result["status"] != "success" or result["checkpoint_sha256"] != saved["checkpoint_sha256"]
                    or result["selection_sha256"] != sha(directory / "selection.json")
                    or result["prediction_sha256"] != sha(run / "test_predictions.csv")):
                raise ValueError("Test artifact integrity failure")
            truth = read_rows(ROOT / "data/splits/nt_tasks" / task / "test.jsonl")
            with (run / "test_predictions.csv").open(newline="") as handle:
                preds = list(csv.DictReader(handle))
            if len(preds) != len(truth):
                raise ValueError("Test coverage mismatch")
            labels, predicted = [], []
            for row, true in zip(preds, truth):
                p = np.asarray([float(row[f"prob_class_{i}"]) for i in range(2)])
                if (str(row["id"]) != str(true["id"]) or int(row["label"]) != true["label"]
                        or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
                        or abs(p.sum() - 1) > 1e-5 or int(row["pred"]) != p.argmax()):
                    raise ValueError("Prediction identity/probability mismatch")
                labels.append(true["label"])
                predicted.append(int(p.argmax()))
            metrics = {"MCC": float(matthews_corrcoef(labels, predicted)),
                       "Accuracy": float(accuracy_score(labels, predicted)), "rows": len(truth)}
            if any(abs(result["test"][k] - v) > 1e-12 for k, v in metrics.items()):
                raise ValueError("Test metrics did not replay")
            records.append({"task": task, "seed": spec["seed"], "lr": spec["lr"], **metrics})
        expected = {(42, lr) for lr in GRID} | {(s, selection["selected_lr"]) for s in SEEDS[1:]}
        if keys != expected or len(fits) != 5:
            raise ValueError("Expected exactly five fits per task")
    if {(r["task"], r["seed"]) for r in records} != {(t, s) for t in TASKS for s in SEEDS}:
        raise ValueError("Missing task/seed test result")
    summaries = [{"task": task, "seeds": SEEDS,
                  **{f"{metric}_{suffix}": float(fn([r[metric] for r in records if r["task"] == task]))
                     for metric in ("MCC", "Accuracy")
                     for suffix, fn in (("mean", np.mean), ("sample_sd", lambda x: np.std(x, ddof=1)))}}
                 for task in TASKS]
    verified = {"status": "verified", "fits": 10, "test_results": 6, "records": records,
                "summary": summaries, "scope": "Checkpoint provenance and raw prediction/metric replay; not independent checkpoint inference."}
    save_json(output / "verification.json", verified)
    return verified


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if sum((args.smoke, args.preflight_only, args.verify_only)) > 1:
        raise ValueError("Choose at most one special mode")
    args.output.mkdir(parents=True, exist_ok=True)
    args.split_dir = ROOT / "data/splits/nt_tasks"
    args.device = torch.device(args.device)
    args.amp = args.device.type == "cuda"
    args.epochs, args.patience = (1 if args.smoke else 15), 4
    args.source_hashes, args.model_hashes = source_hashes(), {}
    args.runtime = {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "gpu": torch.cuda.get_device_name(args.device) if args.device.type == "cuda" else None}
    torch.set_num_threads(4)
    import fcntl
    with (args.output / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.preflight_only:
            preflight(args)
            return
        if args.verify_only:
            audit(args.output)
            return
        if not args.smoke and args.device.type != "cuda":
            raise ValueError("Formal matched run requires the reserved CUDA device and AMP")
        cases = {task: CandidateCase(args, task) for task in TASKS}
        protocol = {"method": METHOD, "phase": "smoke" if args.smoke else "formal",
                    "upstream": {"repository": "https://github.com/Kuanhao-Chao/OpenSpliceAI", "commit": COMMIT,
                                 "files": FILES, "license": "GPL-3.0; unchanged upstream files retained"},
                    "architecture": {"channels": 32, "widths": WIDTHS, "dilations": DILATIONS,
                                     "context": CONTEXT, "input_length": 2600, "observed_length": 600,
                                     "unknown_flanks_per_side": 1000, "output_length": 600},
                    "tasks": {t: {"anchors": c.anchors, "alignment": c.alignment, "split_sha256": c.hashes}
                              for t, c in cases.items()},
                    "initialization": "random; no pretrained weights or teacher", "grid": GRID, "seeds": SEEDS,
                    "training": "Candidate-label weighted CE only, no invented dense labels. Exact fair baseline optimizer, batch16, dev-MCC scheduler/checkpoint and patience; three seed42 LR trials then two further seeds.",
                    "strand_rule": "Maximum target-channel softmax probability across forward and reverse-complement candidate positions; differentiable maximum log odds in training.",
                    "selection": "Dev MCC only, smaller LR breaks ties; test once per selected seed at fixed argmax/0.5 rule.",
                    "limitations": ["Candidate-supervised adaptation, not a reproduction of dense genome-wide training.",
                                    "Single model per seed, not a five-model ensemble.",
                                    "2000-context official architecture supplied only the same 600 observed bases; other bases are unknown zeros.",
                                    "Original train/dev interval and gene overlaps remain; source test chromosomes are held out.",
                                    "Three seeds give descriptive sample SD, not a significance test."],
                    "source_sha256": args.source_hashes, "runtime": args.runtime}
        path = args.output / "protocol.json"
        if path.exists() and json.loads(path.read_text()) != protocol:
            raise ValueError("Existing specialist protocol differs")
        save_json(path, protocol)
        for task, case in cases.items():
            save_json(args.output / "status.json", {"state": "running", "pid": os.getpid(), "task": task,
                                                   "updated_unix": time.time(), "phase": protocol["phase"]})
            directory = args.output / task / METHOD
            directory.mkdir(parents=True, exist_ok=True)
            trials = [checked_fit(case, 42, lr, args, directory) for lr in (GRID[:1] if args.smoke else GRID)]
            winner, selected = freeze_selection(trials, directory)
            final_test(case, METHOD, 42, winner, args)
            for seed in ([] if args.smoke else SEEDS[1:]):
                run, _ = checked_fit(case, seed, selected["selected_lr"], args, directory)
                final_test(case, METHOD, seed, run, args)
            case.base.clear()
            gc.collect()
        if not args.smoke:
            audit(args.output)
        save_json(args.output / "status.json", {"state": "complete", "pid": os.getpid(), "phase": protocol["phase"],
                                               "fits": 2 if args.smoke else 10, "test_results": 0 if args.smoke else 6})


if __name__ == "__main__":
    main()
