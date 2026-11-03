"""Reproducible hard-label branch ablations and validation-only sensitivity runs.

This is a new experimental protocol, not a replacement for published results.
Frozen context is cached with content-addressed manifests; teacher supervision is
deliberately absent so that branch effects are not confounded by distillation.
"""

import argparse
import contextlib
import csv
import fcntl
import hashlib
import json
import os
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from sklearn.metrics import accuracy_score, matthews_corrcoef
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.models.bioqpred_v2 import BioQPredV2Classifier, PositionAwareLocalBranch, compute_composition_features

TASKS = ["splice_sites_all", "splice_sites_acceptor", "splice_sites_donor", "H3K27me3"]
VARIANTS = ["local_comp", "full", "local_only", "without_comp", "without_position", "without_local"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # CUDA adaptive average-pooling backward has no strict deterministic kernel.
    torch.use_deterministic_algorithms(True, warn_only=True)


def flags(variant):
    return dict(use_local=variant != "without_local", use_bert=variant not in ("local_comp", "local_only"), use_comp=variant not in ("local_only", "without_comp"), use_position=variant != "without_position")


def build_model(variant, length, classes, channels, kernels):
    stub = nn.Module()
    stub.config = SimpleNamespace(hidden_size=768)
    switches = flags(variant)
    model = BioQPredV2Classifier(backbone=stub, num_classes=classes, max_len=length, bert_input_dim=1536, fusion="concat", dropout=0.2, **switches)
    if switches["use_local"] and (channels != 128 or tuple(kernels) != (3, 5, 7, 11, 15)):
        model.local_branch = PositionAwareLocalBranch(max_len=length, branch_channels=channels, kernel_sizes=kernels, use_position=switches["use_position"], dropout=0.2)
    return model


class ContextCache:
    def __init__(self, model_dir, directory, device, batch_size):
        self.model_dir, self.directory = model_dir, directory
        self.device, self.batch_size = device, batch_size
        self.directory.mkdir(parents=True, exist_ok=True)
        self.source_hashes = {p.name: sha(p) for p in sorted(model_dir.iterdir()) if p.suffix in (".json", ".py", ".safetensors", ".bin")}

    def get(self, task, split, rows):
        import transformers

        identity = {
            "task": task, "split": split,
            "ordered_id_sequence_sha256": digest([(r["id"], r["sequence"]) for r in rows]),
            "source_hashes": self.source_hashes,
            "max_tokens": 512, "pooling": "mask_aware_mean_max", "dtype": "float32",
            "transformers": transformers.__version__, "torch": torch.__version__,
        }
        key = digest(identity)[:20]
        path = self.directory / f"{task}_{split}_{key}.npy"
        metadata = path.with_suffix(".json")
        start = time.perf_counter()
        if path.exists() and metadata.exists():
            saved = json.loads(metadata.read_text())
            if saved["identity"] != identity or saved["array_sha256"] != sha(path):
                raise RuntimeError(f"Cache integrity check failed: {path}")
            arr = np.load(path)
            if arr.shape != (len(rows), 1536) or not np.isfinite(arr).all():
                raise RuntimeError(f"Cache shape/value check failed: {path}")
            return arr, {"cache": str(path), "reused": True, "seconds": time.perf_counter() - start, "array_sha256": saved["array_sha256"]}

        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), trust_remote_code=True, local_files_only=True)
        backbone = AutoModel.from_pretrained(str(self.model_dir), trust_remote_code=True, local_files_only=True).to(self.device).eval()
        for parameter in backbone.parameters():
            parameter.requires_grad = False
        buffers, untruncated_lengths = [], []
        with torch.inference_mode():
            for offset in range(0, len(rows), self.batch_size):
                seqs = [r["sequence"] for r in rows[offset:offset + self.batch_size]]
                untruncated_lengths.extend(len(v) for v in tokenizer(seqs, truncation=False, padding=False)["input_ids"])
                encoded = tokenizer(seqs, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
                outputs = backbone(**encoded)
                hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
                mask = encoded["attention_mask"].unsqueeze(-1)
                mean = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                maximum = hidden.masked_fill(mask == 0, torch.finfo(hidden.dtype).min).amax(1)
                buffers.append(torch.cat([mean, maximum], -1).float().cpu().numpy())
                if offset % (self.batch_size * 100) == 0:
                    print(f"CACHE {task} {split} {offset}/{len(rows)}", flush=True)
        arr = np.concatenate(buffers).astype("float32")
        if arr.shape != (len(rows), 1536) or not np.isfinite(arr).all():
            raise RuntimeError("Invalid newly extracted context features")
        with path.with_suffix(".npy.tmp").open("wb") as handle:
            np.save(handle, arr)
        path.with_suffix(".npy.tmp").replace(path)
        saved = {
            "identity": identity, "array_sha256": sha(path), "backbone_params": sum(p.numel() for p in backbone.parameters()),
            "extraction_seconds": time.perf_counter() - start,
            "untruncated_token_min": min(untruncated_lengths), "untruncated_token_max": max(untruncated_lengths),
            "truncated_rows": sum(v > 512 for v in untruncated_lengths), "rows": len(rows),
        }
        save_json(metadata, saved)
        del backbone
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return arr, {"cache": str(path), "reused": False, "seconds": saved["extraction_seconds"], "array_sha256": saved["array_sha256"]}


def read_rows(path, limit=0):
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return rows[:limit] if limit else rows


def tensors(rows, length, context):
    lookup = np.zeros(256, dtype="int64")
    for base, number in zip("ACGT", (1, 2, 3, 4)):
        lookup[ord(base)] = number
    bases = np.zeros((len(rows), length), dtype="int64")
    for index, row in enumerate(rows):
        values = lookup[np.frombuffer(row["sequence"].upper()[:length].encode("ascii"), dtype="uint8")]
        bases[index, :len(values)] = values
    return TensorDataset(torch.from_numpy(bases), compute_composition_features([r["sequence"] for r in rows]), torch.from_numpy(context), torch.tensor([r["label"] for r in rows], dtype=torch.long))


def amp_context(device, enabled):
    return torch.autocast("cuda", dtype=torch.float16) if enabled and device.type == "cuda" else contextlib.nullcontext()


@torch.inference_mode()
def evaluate(model, loader, device, amp):
    model.eval()
    labels, probabilities = [], []
    for bases, comp, context, truth in loader:
        with amp_context(device, amp):
            logits = model({}, bases.to(device), comp.to(device), bert_pooled=context.to(device))
        probabilities.append(logits.float().softmax(-1).cpu().numpy())
        labels.append(truth.numpy())
    y, probs = np.concatenate(labels), np.concatenate(probabilities)
    if not np.isfinite(probs).all():
        raise RuntimeError("Nonfinite evaluation probabilities")
    return {"MCC": float(matthews_corrcoef(y, probs.argmax(1))), "accuracy": float(accuracy_score(y, probs.argmax(1)))}, y, probs


def run_one(args, task, variant, seed, cache, output):
    task_dir = ROOT / "data/splits/nt_tasks" / task
    length = 1000 if task == "H3K27me3" else 600
    source_files = {"runner": sha(__file__), "model": sha(ROOT / "src/models/bioqpred_v2.py")}
    split_hashes = {s: sha(task_dir / (s + ".jsonl")) for s in ("train", "dev", "test")}
    spec = {"task": task, "variant": variant, "seed": seed, "channels": args.channels, "kernels": args.kernels, "epochs": args.epochs, "patience": args.patience, "batch_size": args.batch_size, "lr": args.lr, "weight_decay": 0.01, "amp": args.amp, "teacher": False, "determinism": "seeded_and_cudnn_deterministic; unsupported_pooling_kernels_warn_only", "phase": "smoke" if args.smoke else "validation_only" if args.validation_only else "ablation", "source_files": source_files, "split_sha256": split_hashes, "frozen_backbone_sources": cache.source_hashes if flags(variant)["use_bert"] else None}
    run_dir = output / task / f"{variant}_c{args.channels}_k{'-'.join(map(str, args.kernels))}_seed{seed}_{digest(spec)[:10]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        required = (run_dir / "best.pt").exists() and sha(run_dir / "best.pt") == saved.get("checkpoint_sha256")
        if not args.validation_only and not args.smoke:
            predictions = run_dir / "test_predictions.csv"
            required = required and saved.get("test") is not None and predictions.exists() and sha(predictions) == saved.get("predictions_sha256")
        if saved.get("status") == "success" and saved["spec"] == spec and required:
            print("SKIP verified completed run", run_dir, flush=True)
            return
        raise RuntimeError(f"Existing result is not verifiably complete: {run_dir}")
    save_json(run_dir / "spec.json", spec)
    save_json(output / "status.json", {"state": "running", "pid": os.getpid(), "run": str(run_dir), "updated_utc": datetime.now(timezone.utc).isoformat()})
    train_rows = read_rows(task_dir / "train.jsonl", 64 if args.smoke else 0)
    dev_rows = read_rows(task_dir / "dev.jsonl", 32 if args.smoke else 0)
    classes = max(r["label"] for r in train_rows) + 1
    if set(r["label"] for r in train_rows) != set(range(classes)):
        raise ValueError("Training labels are not contiguous")
    contexts, cache_events = {}, {}
    for split, rows in (("train", train_rows), ("dev", dev_rows)):
        if flags(variant)["use_bert"]:
            contexts[split], cache_events[split] = cache.get(task, split, rows)
        else:
            contexts[split] = np.zeros((len(rows), 1), dtype="float32")
    seed_everything(seed)
    model = build_model(variant, length, classes, args.channels, args.kernels).to(args.device)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(tensors(train_rows, length, contexts["train"]), batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
    dev_loader = DataLoader(tensors(dev_rows, length, contexts["dev"]), batch_size=args.batch_size, shuffle=False, num_workers=0)
    counts = np.bincount([r["label"] for r in train_rows], minlength=classes)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(len(train_rows) / (classes * counts), dtype=torch.float32, device=args.device))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and args.device.type == "cuda")
    if args.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)
    best, bad, history = -float("inf"), 0, []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for bases, comp, context, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with amp_context(args.device, args.amp):
                logits = model({}, bases.to(args.device), comp.to(args.device), bert_pooled=context.to(args.device))
                loss = criterion(logits, labels.to(args.device))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        dev_metrics, _, _ = evaluate(model, dev_loader, args.device, args.amp)
        scheduler.step(dev_metrics["MCC"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "dev": dev_metrics, "elapsed_seconds": time.perf_counter() - start})
        save_json(run_dir / "history.json", history)
        print(f"{task} {variant} seed={seed} epoch={epoch} dev_MCC={dev_metrics['MCC']:.6f} loss={np.mean(losses):.6f}", flush=True)
        if dev_metrics["MCC"] > best:
            best, bad = dev_metrics["MCC"], 0
            torch.save({"model_state_dict": model.state_dict(), "spec": spec, "epoch": epoch}, run_dir / "best.tmp.pt")
            (run_dir / "best.tmp.pt").replace(run_dir / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                break
    training_seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated(args.device) if args.device.type == "cuda" else None
    checkpoint = torch.load(run_dir / "best.pt", map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    result = {"status": "success", "spec": spec, "best_epoch": checkpoint["epoch"], "best_dev_MCC": best, "epochs_completed": len(history), "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad), "stored_head_params": sum(p.numel() for p in model.parameters()), "backbone_weights_external": flags(variant)["use_bert"], "training_with_cached_context_seconds": training_seconds, "training_peak_allocated_bytes_excluding_backbone_cache": peak, "cache_events": cache_events, "checkpoint_sha256": sha(run_dir / "best.pt"), "test": None}
    if not args.validation_only and not args.smoke:
        test_rows = read_rows(task_dir / "test.jsonl")
        if flags(variant)["use_bert"]:
            # Release the head before extracting test context to keep memory use bounded.
            model.cpu()
            contexts["test"], cache_events["test"] = cache.get(task, "test", test_rows)
            model.to(args.device)
        else:
            contexts["test"] = np.zeros((len(test_rows), 1), dtype="float32")
        loader = DataLoader(tensors(test_rows, length, contexts["test"]), batch_size=args.batch_size, shuffle=False)
        test_start = time.perf_counter()
        metrics, labels, probs = evaluate(model, loader, args.device, args.amp)
        result["test"] = {**metrics, "cached_head_evaluation_seconds": time.perf_counter() - test_start, "rows": len(test_rows)}
        with (run_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "task", "seed", "variant", "label", "pred"] + [f"prob_class_{i}" for i in range(classes)])
            for row, label, ps in zip(test_rows, labels, probs):
                writer.writerow([row["id"], task, seed, variant, int(label), int(ps.argmax())] + ps.tolist())
        result["predictions_sha256"] = sha(run_dir / "test_predictions.csv")
    save_json(result_path, result)
    print("COMPLETE", run_dir, "test=", result["test"], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=["local_comp", "full"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2024, 3407])
    parser.add_argument("--kernels", nargs="+", type=int, default=[3, 5, 7, 11, 15])
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(k < 1 or k % 2 == 0 for k in args.kernels) or args.channels < 1:
        parser.error("Kernels must be positive odd integers; channels must be positive")
    if args.smoke:
        args.epochs = 1
    args.device = torch.device(args.device)
    torch.set_num_threads(4)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save_json(output / "invocation.json", {"argv": sys.argv, "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch": torch.__version__})
        cache = ContextCache(ROOT / "models/dnabert2", ROOT / "revision_20261022/context_cache", args.device, args.batch_size)
        try:
            for task in args.tasks:
                for seed in args.seeds:
                    for variant in args.variants:
                        run_one(args, task, variant, seed, cache, output)
        except BaseException:
            save_json(output / "status.json", {"state": "failed", "pid": os.getpid(), "traceback": traceback.format_exc(), "updated_utc": datetime.now(timezone.utc).isoformat()})
            raise
        save_json(output / "status.json", {"state": "complete", "pid": os.getpid(), "updated_utc": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
