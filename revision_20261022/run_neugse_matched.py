"""Give the stronger preserved NeuGSE-style architecture the same search budget."""

import argparse
import fcntl
import gc
import importlib.metadata
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import run_fair_baselines as fit
from runtime_helpers import gpu_free, process_identity
from src.models.baseline_neugse_style import NeuGSEStyleClassifier


HERE, ROOT = Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent
TASKS = ("splice_sites_all", "splice_sites_acceptor", "splice_sites_donor", "H3K27me3")
GRID = (1e-4, 3e-4, 1e-3)
METHOD = "neugse_adapted"


class NeuGSEAdapter(NeuGSEStyleClassifier):
    def forward(self, inputs):
        return super().forward((inputs + 4) % 5)


class TaskCase(fit.TaskCase):
    def collate(self, method, split):
        if method != METHOD:
            raise ValueError("Unexpected model in the standalone NeuGSE-style study")
        return super().collate("cnn", split)

    def model(self, method, seed):
        if method != METHOD:
            raise ValueError("Unexpected model in the standalone NeuGSE-style study")
        fit.seed_everything(seed)
        return NeuGSEAdapter(self.classes, 1000 if self.task == "H3K27me3" else 600)


def verify_predecessor(directory):
    state = json.loads((directory / "status.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    if state["state"] != "complete" or state["completed_fits"] != 48 or protocol["phase"] != "fixed_budget_overlap_control":
        raise ValueError("The original/filtered predecessor is not complete")
    observed = set()
    paths = list(directory.rglob("test_result.json"))
    for path in paths:
        result = json.loads(path.read_text())
        saved = json.loads(path.with_name("fit_result.json").read_text())
        spec = saved["spec"]
        partition = path.relative_to(directory).parts[0]
        observed.add((partition, spec["task"], spec["method"], spec["seed"]))
        if result["status"] != "success" or result["test"]["rows"] != 3000:
            raise ValueError("Incomplete predecessor test result")
        if fit.sha(path.with_name("best.pt")) != result["checkpoint_sha256"] or fit.sha(path.with_name("test_predictions.csv")) != result["prediction_sha256"]:
            raise ValueError("Predecessor artifact changed")
    expected = {(partition, task, method, seed) for partition in ("original", "train_dev_RC_disjoint") for task in TASKS for method in ("bio_full", "bio_local_comp") for seed in (42, 2024, 3407)}
    if observed != expected or len(paths) != 48:
        raise ValueError("Predecessor coverage differs from its fixed protocol")


def status(output, value):
    fit.save_json(output / "status.json", {**value, "pid": os.getpid(), "updated_unix": time.time()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--wait-for-overlap-pid", type=int)
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    if args.smoke and args.wait_for_overlap_pid:
        parser.error("Smoke is immediate, not a formal queued study")
    args.output.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), HERE / "run_fair_baselines.py", HERE / "fair_baseline_models.py", HERE / "run_controlled_experiments.py"] + list((ROOT / "src/models").glob("*.py"))
    source_hashes = {str(path.relative_to(ROOT)): fit.sha(path) for path in sources}
    protocol = {"phase": "smoke" if args.smoke else "matched_budget", "method": METHOD, "tasks": args.tasks, "LR_grid": GRID, "selection": "Exactly three seed-42 dev trials; select maximum dev MCC, ties smaller LR, before test. Reuse the selected LR for seeds 2024 and 3407 without new search.", "effective_batch": 16, "max_epochs": 1 if args.smoke else 15, "patience": 4, "amp": True, "pretrained_backbone": None, "architecture": "The preserved project NeuGSE-style local sequence architecture, with only base-ID interface conversion. Architectural adaptation, not direct source-paper task reproduction.", "source_sha256": source_hashes, "scope": "Additional strong comparator, not a replacement or deletion of earlier stronger historical results."}
    with (args.output / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.output / "protocol.json"
        if path.exists() and json.loads(path.read_text()) != protocol:
            raise ValueError("Existing strong-comparator protocol differs")
        fit.save_json(path, protocol)
        if args.wait_for_overlap_pid:
            parent = process_identity(args.wait_for_overlap_pid)
            if parent and "run_overlap_replay.py" not in parent["command"]:
                raise ValueError("Predecessor PID is not the expected overlap study")
            while parent:
                current = process_identity(args.wait_for_overlap_pid)
                if current is None or current["state"] == "Z" or current["start_ticks"] != parent["start_ticks"]:
                    break
                status(args.output, {"state": "waiting_for_overlap_study", "parent_pid": args.wait_for_overlap_pid, "parent_start_ticks": parent["start_ticks"]})
                time.sleep(60)
            verify_predecessor(HERE / "overlap_replay")
            while not gpu_free(args.gpu):
                status(args.output, {"state": "waiting_for_gpu", "gpu": args.gpu})
                time.sleep(60)
        if any(fit.sha(ROOT / name) != digest for name, digest in source_hashes.items()):
            raise ValueError("Source changed while the strong comparator was queued")
        if shutil.disk_usage(args.output).free < 2 * 1024**3:
            raise OSError("Strong-comparator output needs 2 GiB free")
        torch.set_num_threads(2)
        device = torch.device("cuda:0")
        runtime = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"), "numpy": np.__version__, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu": torch.cuda.get_device_name(device)}
        options = SimpleNamespace(smoke=args.smoke, split_dir=ROOT / "data/splits/nt_tasks", model_dir=ROOT / "models/dnabert2", context_cache=args.output / "unused_context_cache", device=device, amp=True, epochs=1 if args.smoke else 15, patience=4, source_hashes=source_hashes, model_hashes={}, runtime=runtime)
        completed = 0
        for task in args.tasks:
            case = TaskCase(options, task)
            directory = args.output / task / METHOD
            directory.mkdir(parents=True, exist_ok=True)
            status(args.output, {"state": "training", "task": task, "completed_fits": completed})
            trials = [fit.fit(case, METHOD, 42, lr, options, directory) for lr in GRID]
            selected, selection = fit.freeze_selection(trials, directory)
            completed += 3
            for trial, result in trials:
                if trial != selected and (trial / "best.pt").exists():
                    if not (trial / "best.pt").resolve().is_relative_to(args.output.resolve()):
                        raise ValueError("Temporary checkpoint is outside this study output")
                    (trial / "best.pt").unlink()
                    result["checkpoint_retained"] = False
                    result["pruning_note"] = "Unselected own temporary weights pruned after dev-only selection; logs/hash retained."
                    fit.save_json(trial / "fit_result.json", result)
            fit.final_test(case, METHOD, 42, selected, options)
            for seed in (() if args.smoke else (2024, 3407)):
                trial, _ = fit.fit(case, METHOD, seed, selection["selected_lr"], options, directory)
                fit.final_test(case, METHOD, seed, trial, options)
                completed += 1
            del case
            gc.collect()
            torch.cuda.empty_cache()
        status(args.output, {"state": "smoke_complete" if args.smoke else "complete", "completed_fits": completed, "scope": "Preserved strong architectural comparator under the declared matched trial budget."})


if __name__ == "__main__":
    main()
