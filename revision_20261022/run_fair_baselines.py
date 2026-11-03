"""Equal-trial dev-selected baselines with test isolation and bounded storage."""

import argparse
import gc
import importlib.metadata
import itertools
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, matthews_corrcoef
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fair_baseline_models import CNNAdapter, FrozenMeanHead, MaskedCAMPHead, MaskedTFBSHead
from run_controlled_experiments import ContextCache, amp_context, build_model, digest, read_rows, save_json, seed_everything, sha
from src.models.bioqpred_v2 import compute_composition_features
from src.models.dnabert2_classifier import DNABERT2Classifier


METHODS = ("bio_full", "bio_local_comp", "cnn", "dnabert_frozen", "dnabert_finetune", "camp_adapted_masked", "tfbs_adapted_masked")
TASKS = ("splice_sites_all", "splice_sites_acceptor", "splice_sites_donor", "H3K27me3")
GRIDS = {"bio_full": [3e-5, 1e-4, 3e-4], "bio_local_comp": [3e-5, 1e-4, 3e-4], "cnn": [1e-4, 3e-4, 1e-3], "dnabert_frozen": [3e-4, 1e-3, 3e-3], "dnabert_finetune": [1e-5, 2e-5, 5e-5], "camp_adapted_masked": [1e-4, 3e-4, 1e-3], "tfbs_adapted_masked": [1e-4, 3e-4, 1e-3]}


class Rows(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return index, self.rows[index]


def to_device(inputs, device):
    return {k: v.to(device) for k, v in inputs.items()} if isinstance(inputs, dict) else inputs.to(device)


class TaskCase:
    def __init__(self, args, task):
        self.args, self.task = args, task
        self.directory = args.split_dir / task
        self.rows = {split: read_rows(self.directory / (split + ".jsonl"), 64 if args.smoke and split == "train" else 32 if args.smoke else 0) for split in ("train", "dev")}
        self.classes = len({r["label"] for r in self.rows["train"]})
        if {r["label"] for r in self.rows["train"]} != set(range(self.classes)):
            raise ValueError("Non-contiguous training labels")
        self.hashes = {split: sha(self.directory / (split + ".jsonl")) for split in ("train", "dev", "test")}
        counts = np.bincount([r["label"] for r in self.rows["train"]], minlength=self.classes)
        self.weights = torch.tensor(len(self.rows["train"]) / (self.classes * counts), dtype=torch.float32)
        self.mean, self.token, self.base = {}, {}, {}
        self.context, self.comp = {}, {}
        self.tokenizer = None
        self.external_params = None
        self.cache_events = []

    def ensure_tokenizer(self):
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.args.model_dir), trust_remote_code=True, local_files_only=True)

    def load_backbone(self):
        from transformers import AutoModel
        backbone = AutoModel.from_pretrained(str(self.args.model_dir), trust_remote_code=True, local_files_only=True)
        self.external_params = sum(p.numel() for p in backbone.parameters())
        return backbone

    def ensure_split(self, split):
        if split not in self.rows:
            if self.args.smoke:
                raise RuntimeError("Smoke runs must not read test samples")
            self.rows[split] = read_rows(self.directory / (split + ".jsonl"))

    def ensure_features(self, method, split):
        self.ensure_split(split)
        rows = self.rows[split]
        if method.startswith("bio_"):
            self.ensure_features("cnn", split)
            if split not in self.comp:
                self.comp[split] = compute_composition_features([r["sequence"] for r in rows])
            if method == "bio_full":
                self.ensure_context(split)
        elif method == "cnn" and split not in self.base:
            length = 1000 if self.task == "H3K27me3" else 600
            lookup = np.zeros(256, dtype="int64")
            for code, base in enumerate("ACGT", 1):
                lookup[ord(base)] = code
            array = np.zeros((len(rows), length), dtype="int64")
            for index, row in enumerate(rows):
                seq = row["sequence"].upper()
                if len(seq) != length:
                    raise ValueError("The fixed benchmark window must be preserved")
                array[index] = lookup[np.frombuffer(seq.encode(), dtype="uint8")]
            self.base[split] = torch.from_numpy(array)
        elif method == "dnabert_frozen" and split not in self.mean:
            self.ensure_context(split)
            self.mean[split] = self.context[split][:, :768].contiguous()
        elif method.startswith(("camp_", "tfbs_")) and split not in self.token:
            self.ensure_tokenizer()
            backbone = self.load_backbone().to(self.args.device).eval()
            for parameter in backbone.parameters():
                parameter.requires_grad = False
            previous_bytes = sum(value.nbytes for split_values in self.token.values() for value in split_values)
            values, total_bytes = [], previous_bytes
            start = time.perf_counter()
            with torch.inference_mode():
                for offset in range(0, len(rows), 16):
                    batch = self.tokenizer([r["sequence"] for r in rows[offset:offset + 16]], padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.args.device)
                    outputs = backbone(**batch)
                    hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
                    for array, valid in zip(hidden, batch["attention_mask"].sum(1).tolist()):
                        item = array[:valid].float().cpu().numpy().copy()
                        total_bytes += item.nbytes
                        if total_bytes > 48 * 1024**3:
                            raise MemoryError("The predeclared 48-GiB token cache cap was exceeded")
                        values.append(item)
                    if offset % 1600 == 0:
                        print("TOKEN_CACHE", self.task, split, offset, "/", len(rows), flush=True)
            self.token[split] = values
            self.cache_events.append({"kind": "RAM_token_states", "split": split, "rows": len(rows), "bytes": total_bytes - previous_bytes, "aggregate_bytes": total_bytes, "seconds": time.perf_counter() - start})
            del backbone
            torch.cuda.empty_cache()

    def ensure_context(self, split):
        if split not in self.context:
            cache = ContextCache(self.args.model_dir, self.args.context_cache, self.args.device, 16)
            values, event = cache.get(self.task, split, self.rows[split])
            self.context[split] = torch.from_numpy(values)
            self.external_params = json.loads(Path(event["cache"]).with_suffix(".json").read_text())["backbone_params"]
            self.cache_events.append(event)

    def collate(self, method, split):
        self.ensure_features(method, split)
        if method == "dnabert_finetune":
            self.ensure_tokenizer()

        def collate(batch):
            indices, rows = zip(*batch)
            labels = torch.tensor([r["label"] for r in rows], dtype=torch.long)
            if method.startswith("bio_"):
                inputs = {"base_ids": self.base[split][list(indices)], "comp_features": self.comp[split][list(indices)], "bert_pooled": self.context[split][list(indices)] if method == "bio_full" else torch.zeros(len(rows), 1)}
            elif method == "cnn":
                inputs = self.base[split][list(indices)]
            elif method == "dnabert_frozen":
                inputs = self.mean[split][list(indices)]
            elif method == "dnabert_finetune":
                inputs = dict(self.tokenizer([r["sequence"] for r in rows], padding=True, truncation=True, max_length=512, return_tensors="pt"))
            else:
                features = [self.token[split][i] for i in indices]
                length = max(len(value) for value in features)
                hidden, mask = torch.zeros(len(features), length, 768), torch.zeros(len(features), length, dtype=torch.bool)
                for index, feature in enumerate(features):
                    hidden[index, :len(feature)] = torch.from_numpy(feature)
                    mask[index, :len(feature)] = True
                inputs = {"hidden": hidden, "mask": mask}
            return inputs, labels, [r["id"] for r in rows]

        return collate

    def model(self, method, seed):
        seed_everything(seed)
        if method.startswith("bio_"):
            return build_model(method[4:], 1000 if self.task == "H3K27me3" else 600, self.classes, 128, (3, 5, 7, 11, 15))
        if method == "cnn":
            return CNNAdapter(self.classes)
        if method == "dnabert_frozen":
            return FrozenMeanHead(self.classes)
        if method == "camp_adapted_masked":
            return MaskedCAMPHead(self.classes)
        if method == "tfbs_adapted_masked":
            return MaskedTFBSHead(self.classes)
        return DNABERT2Classifier(self.load_backbone(), self.classes, frozen=False)


def forward(model, method, inputs):
    if method.startswith("bio_"):
        return model({}, **inputs)
    return model(**inputs) if method == "dnabert_finetune" else model(inputs)


@torch.inference_mode()
def evaluate(model, method, loader, args):
    model.eval()
    ys, probabilities, ids = [], [], []
    for inputs, labels, batch_ids in loader:
        with amp_context(args.device, args.amp):
            logits = forward(model, method, to_device(inputs, args.device))
        probabilities.append(logits.float().softmax(-1).cpu().numpy())
        ys.extend(labels.tolist())
        ids.extend(batch_ids)
    probs = np.concatenate(probabilities)
    if not np.isfinite(probs).all():
        raise RuntimeError("Nonfinite baseline probabilities")
    result = {"MCC": float(matthews_corrcoef(ys, probs.argmax(1))), "Accuracy": float(accuracy_score(ys, probs.argmax(1))), "rows": len(ys)}
    return result, ys, probs, ids


def make_loader(case, method, split, seed, batch_size, shuffle):
    collate = case.collate(method, split)
    return DataLoader(Rows(case.rows[split]), batch_size=batch_size, shuffle=shuffle, generator=torch.Generator().manual_seed(seed), collate_fn=collate)


def fit(case, method, seed, lr, args, output):
    inputs = {"task": case.task, "method": method, "seed": seed, "lr": lr, "epochs": args.epochs, "patience": args.patience, "effective_batch_size": 16, "microbatch": 4 if method == "dnabert_finetune" else 16, "weight_decay": 0.01, "amp": args.amp, "split_sha256": case.hashes, "source_sha256": args.source_hashes, "model_sha256": args.model_hashes, "runtime": args.runtime, "phase": "smoke" if args.smoke else "dev_fit"}
    directory = output / f"seed{seed}_lr{lr}_{digest(inputs)[:10]}"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "fit_result.json").exists():
        saved = json.loads((directory / "fit_result.json").read_text())
        if saved["spec"] != inputs:
            raise ValueError("Baseline trial identity mismatch")
        if saved.get("status") != "success":
            raise ValueError("Cached baseline trial did not finish successfully")
        if saved.get("checkpoint_retained") and (not (directory / "best.pt").exists() or sha(directory / "best.pt") != saved["checkpoint_sha256"]):
            raise ValueError("Retained baseline checkpoint is missing or changed")
        return directory, saved
    if shutil.disk_usage(output).free < 2 * 1024**3:
        raise OSError("Baseline training requires at least 2 GiB free")
    train_loader = make_loader(case, method, "train", seed, inputs["microbatch"], True)
    dev_loader = make_loader(case, method, "dev", seed, 16, False)
    model = case.model(method, seed).to(args.device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    external = case.external_params if method not in ("cnn", "bio_local_comp", "dnabert_finetune") else 0
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss(weight=case.weights.to(args.device), reduction="sum")
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and args.device.type == "cuda")
    best, bad, history = -float("inf"), 0, []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        iterator = iter(train_loader)
        accumulation = 16 // inputs["microbatch"]
        while True:
            group = list(itertools.islice(iterator, accumulation))
            if not group:
                break
            denominator = sum(float(case.weights[labels].sum()) for _, labels, _ in group)
            optimizer.zero_grad(set_to_none=True)
            for batch_inputs, labels, _ in group:
                with amp_context(args.device, args.amp):
                    logits = forward(model, method, to_device(batch_inputs, args.device))
                    loss = criterion(logits, labels.to(args.device)) / denominator
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite baseline loss")
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        dev, _, _, _ = evaluate(model, method, dev_loader, args)
        scheduler.step(dev["MCC"])
        history.append({"epoch": epoch, "dev": dev, "elapsed_seconds": time.perf_counter() - start})
        save_json(directory / "history.json", history)
        print(case.task, method, seed, lr, epoch, "dev_MCC", dev["MCC"], flush=True)
        if dev["MCC"] > best:
            best, bad = dev["MCC"], 0
            torch.save({"state": model.state_dict(), "spec": inputs, "epoch": epoch}, directory / "best.tmp.pt")
            (directory / "best.tmp.pt").replace(directory / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                break
    result = {"status": "success", "spec": inputs, "best_dev_MCC": best, "epochs_completed": len(history), "checkpoint_sha256": sha(directory / "best.pt"), "checkpoint_retained": True, "trainable_params": trainable, "stored_head_or_full_params": sum(p.numel() for p in model.parameters()), "external_frozen_params": external or 0, "training_seconds_excluding_feature_cache": time.perf_counter() - start, "feature_cache_events": case.cache_events}
    save_json(directory / "fit_result.json", result)
    del model, optimizer
    gc.collect()
    if args.device.type == "cuda":
        torch.cuda.empty_cache()
    return directory, result


def final_test(case, method, seed, directory, args):
    selection_path = directory.parent / "selection.json"
    selection = json.loads(selection_path.read_text())
    fit_spec = json.loads((directory / "fit_result.json").read_text())["spec"]
    if not selection.get("selected_before_test") or selection["selected_lr"] != fit_spec["lr"]:
        raise ValueError("Final testing requires the frozen dev-only selection")
    selection_hash = sha(selection_path)
    path = directory / "test_result.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if args.smoke and saved.get("status") == "smoke_success" and saved.get("test") is None:
            return
        prediction = directory / "test_predictions.csv"
        if saved.get("status") != "success" or saved.get("test") is None or saved.get("selection_sha256") != selection_hash or not prediction.exists() or not (directory / "best.pt").exists() or sha(prediction) != saved.get("prediction_sha256") or sha(directory / "best.pt") != saved.get("checkpoint_sha256"):
            raise ValueError("Final baseline artifacts are not verifiably complete")
        return
    if args.smoke:
        save_json(path, {"status": "smoke_success", "test": None})
        return
    saved = json.loads((directory / "fit_result.json").read_text())
    if sha(directory / "best.pt") != saved["checkpoint_sha256"]:
        raise ValueError("Selected baseline checkpoint changed")
    loader = make_loader(case, method, "test", seed, 16, False)
    model = case.model(method, seed).to(args.device)
    model.load_state_dict(torch.load(directory / "best.pt", map_location=args.device, weights_only=False)["state"])
    result, labels, probs, ids = evaluate(model, method, loader, args)
    import csv
    prediction = directory / "test_predictions.csv"
    with prediction.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label", "pred"] + [f"prob_class_{i}" for i in range(case.classes)])
        for sid, label, ps in zip(ids, labels, probs):
            writer.writerow([sid, label, int(ps.argmax())] + ps.tolist())
    save_json(path, {"status": "success", "test": result, "prediction_sha256": sha(prediction), "checkpoint_sha256": saved["checkpoint_sha256"], "selection_sha256": selection_hash})
    print("FINAL_TEST", case.task, method, seed, result, flush=True)
    del model
    gc.collect()
    if args.device.type == "cuda":
        torch.cuda.empty_cache()


def freeze_selection(trials, directory):
    best_dir, best = max(trials, key=lambda item: (item[1]["best_dev_MCC"], -item[1]["spec"]["lr"]))
    selection = {"selected_lr": best["spec"]["lr"], "seed42_best_dev_MCC": best["best_dev_MCC"], "selected_directory": str(best_dir), "trials": [{"directory": str(p), "lr": r["spec"]["lr"], "dev_MCC": r["best_dev_MCC"]} for p, r in trials], "selected_before_test": True}
    selection_file = directory / "selection.json"
    if selection_file.exists() and json.loads(selection_file.read_text()) != selection:
        raise ValueError("Frozen baseline selection changed")
    save_json(selection_file, selection)
    return best_dir, selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data/splits/nt_tasks")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/dnabert2")
    parser.add_argument("--context-cache", type=Path, default=ROOT / "revision_20261022/context_cache")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    import fcntl
    args.device = torch.device(args.device)
    args.runtime = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"), "numpy": np.__version__, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu": torch.cuda.get_device_name(args.device) if args.device.type == "cuda" else None}
    torch.set_num_threads(4)
    if args.smoke:
        args.epochs = 1
    args.output.mkdir(parents=True, exist_ok=True)
    source = [Path(__file__), Path(__file__).with_name("fair_baseline_models.py"), Path(__file__).with_name("run_controlled_experiments.py"), ROOT / "src/models/bioqpred_v2.py", ROOT / "src/models/cnn_baseline.py", ROOT / "src/models/dnabert2_classifier.py", ROOT / "src/models/baseline_dnabert2_camp_style.py", ROOT / "src/models/baseline_tfbs_finder_style.py"]
    args.source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in source}
    args.model_hashes = {p.name: sha(p) for p in sorted(args.model_dir.iterdir()) if p.suffix in (".py", ".json", ".safetensors", ".bin", ".txt", ".model")}
    with (args.output / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = {"methods": args.methods, "tasks": args.tasks, "grids": GRIDS, "selection": "Exactly three seed-42 learning-rate trials per method/task; maximize dev MCC with smaller LR breaking ties. Freeze selection before any test evaluation. Two additional seeds use that LR without new search.", "teacher": False, "effective_batch": 16, "epochs": args.epochs, "patience": args.patience, "amp": args.amp, "determinism": "Seeded cuDNN deterministic, unsupported CUDA pooling/attention kernels warn only; bitwise identity is not guaranteed.", "adapted_models": "CAMP and TFBS are padding-aware architectural adaptations, including valid-token BatchNorm statistics, not direct source-paper reproductions.", "source_sha256": args.source_hashes, "model_sha256": args.model_hashes, "runtime": args.runtime, "phase": "smoke" if args.smoke else "formal"}
        if (args.output / "protocol.json").exists() and json.loads((args.output / "protocol.json").read_text()) != protocol:
            raise ValueError("Existing baseline protocol differs; preserve previous outputs")
        save_json(args.output / "protocol.json", protocol)
        for task in args.tasks:
            case = TaskCase(args, task)
            for method in args.methods:
                directory = args.output / task / method
                directory.mkdir(parents=True, exist_ok=True)
                trials = [fit(case, method, 42, lr, args, directory) for lr in GRIDS[method]]
                best_dir, selection = freeze_selection(trials, directory)
                for path, result in trials:
                    if path != best_dir and (path / "best.pt").exists():
                        if not (path / "best.pt").resolve().is_relative_to(args.output.resolve()):
                            raise ValueError("Unselected checkpoint is outside this experiment's output")
                        (path / "best.pt").unlink()
                        result["checkpoint_retained"] = False
                        result["pruning_note"] = "Only the unselected temporary weights were pruned after dev-only selection; logs and original hash are retained."
                        save_json(path / "fit_result.json", result)
                final_test(case, method, 42, best_dir, args)
                for seed in (() if args.smoke else (2024, 3407)):
                    path, _ = fit(case, method, seed, selection["selected_lr"], args, directory)
                    final_test(case, method, seed, path, args)
            del case
            gc.collect()
            if args.device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
