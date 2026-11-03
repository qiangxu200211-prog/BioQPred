"""Separate teacher intervention, paired to the retained no-teacher study."""

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import run_controlled_experiments as control
import run_fair_baselines as fair


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TASKS = ("splice_sites_all", "splice_sites_acceptor", "splice_sites_donor", "H3K27me3")
VARIANTS = ("full", "local_comp")
SEEDS = (42, 2024, 3407)
ALPHA, TEMPERATURE = 0.5, 2.0


def distillation_loss(student, teacher, temperature=TEMPERATURE):
    if student.shape != teacher.shape or temperature <= 0:
        raise ValueError("Aligned student and teacher logits and positive temperature required")
    return F.kl_div(F.log_softmax(student.float() / temperature, dim=-1), F.softmax(teacher.detach().float() / temperature, dim=-1), reduction="batchmean") * temperature**2


def train_epoch(model, loader, optimizer, scaler, criterion, device, amp, alpha=ALPHA):
    model.train()
    losses = []
    for bases, comp, context, labels, teacher in loader:
        optimizer.zero_grad(set_to_none=True)
        with control.amp_context(device, amp):
            logits = model({}, bases.to(device), comp.to(device), bert_pooled=context.to(device))
            loss = criterion(logits, labels.to(device)) + alpha * distillation_loss(logits, teacher.to(device))
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite teacher-ablation loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
    return float(np.mean(losses))


def selected_teacher(directory, split_hashes):
    selection_path = directory / "selection.json"
    selection = json.loads(selection_path.read_text())
    trials = selection["trials"]
    if not selection.get("selected_before_test") or len(trials) != 3:
        raise ValueError("Teacher needs the declared three-trial dev-only selection")
    best = max(trials, key=lambda r: (r["dev_MCC"], -r["lr"]))
    path = Path(selection["selected_directory"]).resolve()
    if not path.is_relative_to(directory.resolve()) or path != Path(best["directory"]).resolve() or selection["selected_lr"] != best["lr"]:
        raise ValueError("Teacher was not chosen by the frozen dev-only rule")
    result_path = path / "fit_result.json"
    result = json.loads(result_path.read_text())
    spec = result["spec"]
    if result["status"] != "success" or not result["checkpoint_retained"] or spec["seed"] != 42 or spec["method"] != "dnabert_finetune" or spec["phase"] != "dev_fit" or spec["split_sha256"] != split_hashes or spec["lr"] != selection["selected_lr"]:
        raise ValueError("Teacher training provenance does not match this intervention")
    if control.sha(path / "best.pt") != result["checkpoint_sha256"]:
        raise ValueError("Teacher checkpoint changed")
    return path, result, {"selection_sha256": control.sha(selection_path), "fit_result_sha256": control.sha(result_path), "checkpoint_sha256": result["checkpoint_sha256"], "directory": str(path)}


def check_teacher_array(values, rows, classes):
    if values.dtype != np.float32 or values.shape != (rows, classes) or not np.isfinite(values).all():
        raise ValueError("Teacher logits have invalid shape, dtype or values")


def teacher_logits(case, fair_directory, output):
    directory, fit, provenance = selected_teacher(fair_directory / case.task / "dnabert_finetune", case.hashes)
    for name, expected in fit["spec"]["source_sha256"].items():
        if control.sha(ROOT / name) != expected:
            raise ValueError("Teacher source differs from its retained training specification")
    for name, expected in fit["spec"]["model_sha256"].items():
        if control.sha(case.args.model_dir / name) != expected:
            raise ValueError("Teacher pretrained-model source changed")
    rows = case.rows["train"]
    identity = {"task": case.task, "teacher": provenance, "ordered_train_id_sequence_label_sha256": control.digest([(r["id"], r["sequence"], r["label"]) for r in rows]), "split_sha256": case.hashes, "script_sha256": control.sha(Path(__file__)), "AMP": case.args.amp, "classes": case.classes, "rows": len(rows)}
    output.mkdir(parents=True, exist_ok=True)
    metadata_path, array_path = output / "manifest.json", output / "train_logits.npy"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["identity"] != identity or control.sha(array_path) != metadata["array_sha256"] or control.sha(output / "train_ids.csv") != metadata["ids_sha256"]:
            raise ValueError("Teacher cache identity or payload changed")
        values = np.load(array_path)
        check_teacher_array(values, len(rows), case.classes)
        return values, metadata
    if array_path.exists():
        raise ValueError("Incomplete teacher cache retained; review it before a new attempt")
    loader = fair.make_loader(case, "dnabert_finetune", "train", 42, 16, False)
    model = case.model("dnabert_finetune", 42).to(case.args.device).eval()
    checkpoint = torch.load(directory / "best.pt", map_location=case.args.device, weights_only=False)
    if checkpoint["spec"] != fit["spec"]:
        raise ValueError("Teacher checkpoint specification changed")
    model.load_state_dict(checkpoint["state"])
    values, ids, labels = [], [], []
    start = time.perf_counter()
    with torch.inference_mode():
        for inputs, truth, batch_ids in loader:
            with control.amp_context(case.args.device, case.args.amp):
                logits = fair.forward(model, "dnabert_finetune", fair.to_device(inputs, case.args.device))
            values.append(logits.float().cpu().numpy())
            ids.extend(batch_ids)
            labels.extend(truth.tolist())
    values = np.concatenate(values).astype("float32")
    if ids != [r["id"] for r in rows] or labels != [r["label"] for r in rows]:
        raise ValueError("Teacher logits are not aligned to the exact training order")
    check_teacher_array(values, len(rows), case.classes)
    np.save(array_path, values)
    with (output / "train_ids.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label", "sequence_sha256"])
        import hashlib
        writer.writerows((r["id"], r["label"], hashlib.sha256(r["sequence"].encode()).hexdigest()) for r in rows)
    metadata = {"identity": identity, "array_sha256": control.sha(array_path), "ids_sha256": control.sha(output / "train_ids.csv"), "teacher_training_seconds_from_fair_run": fit["training_seconds_excluding_feature_cache"], "teacher_train_extraction_seconds": time.perf_counter() - start, "scope": "In-sample train logits from one dev-selected seed42 teacher shared by all student seeds. No dev/test logits or test labels enter distillation; this is not out-of-fold distillation."}
    control.save_json(metadata_path, metadata)
    del model, checkpoint
    gc.collect()
    if case.args.device.type == "cuda":
        torch.cuda.empty_cache()
    return values, metadata


def no_teacher_records():
    found = {}
    for directory in (HERE / "branch_ablation", HERE / "component_ablation"):
        for path in directory.rglob("result.json"):
            row = json.loads(path.read_text())
            spec = row["spec"]
            if spec["variant"] not in VARIANTS:
                continue
            key = spec["task"], spec["variant"], spec["seed"]
            if key in found or row["status"] != "success" or spec["teacher"] or spec["phase"] != "ablation" or row["test"]["rows"] != 3000:
                raise ValueError("Invalid no-teacher paired result")
            if control.sha(path.with_name("best.pt")) != row["checkpoint_sha256"] or control.sha(path.with_name("test_predictions.csv")) != row["predictions_sha256"]:
                raise ValueError("No-teacher artifacts changed")
            for name, relative in (("runner", "revision_20261022/run_controlled_experiments.py"), ("model", "src/models/bioqpred_v2.py")):
                if control.sha(ROOT / relative) != spec["source_files"][name]:
                    raise ValueError("No-teacher training source changed")
            found[key] = (path, row)
    if set(found) != {(task, variant, seed) for task in TASKS for variant in VARIANTS for seed in SEEDS}:
        raise ValueError("All 24 paired no-teacher controls must be retained")
    return found


def student(case, variant, seed, baseline, logits, teacher_manifest, args):
    baseline_path, base = baseline
    original = base["spec"]
    if original["split_sha256"] != case.hashes or original["amp"] != args.amp:
        raise ValueError("Paired no-teacher data or numerical precision differ")
    spec = {**original, "phase": "teacher_ablation", "teacher": True, "teacher_alpha": ALPHA, "teacher_temperature": TEMPERATURE, "teacher_train_array_sha256": teacher_manifest["array_sha256"], "teacher_identity": teacher_manifest["identity"], "paired_off_result": str(baseline_path), "paired_off_result_sha256": control.sha(baseline_path), "teacher_runner_sha256": control.sha(Path(__file__))}
    directory = args.output / case.task / f"{variant}_seed{seed}_{control.digest(spec)[:10]}"
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        if saved["status"] != "success" or saved["spec"] != spec or control.sha(directory / "best.pt") != saved["checkpoint_sha256"] or control.sha(directory / "test_predictions.csv") != saved["predictions_sha256"]:
            raise ValueError("Existing teacher result is not verifiably complete")
        return saved
    control.save_json(directory / "spec.json", spec)
    length = 1000 if case.task == "H3K27me3" else 600
    cache = control.ContextCache(args.model_dir, args.context_cache, args.device, original["batch_size"])
    contexts, events = {}, {}
    for split in ("train", "dev"):
        if variant == "full":
            contexts[split], events[split] = cache.get(case.task, split, case.rows[split])
        else:
            contexts[split] = np.zeros((len(case.rows[split]), 1), dtype="float32")
    train_data = control.tensors(case.rows["train"], length, contexts["train"])
    train_data = TensorDataset(*train_data.tensors, torch.from_numpy(logits))
    train_loader = DataLoader(train_data, batch_size=original["batch_size"], shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    dev_loader = DataLoader(control.tensors(case.rows["dev"], length, contexts["dev"]), batch_size=original["batch_size"], shuffle=False)
    control.seed_everything(seed)
    model = control.build_model(variant, length, case.classes, original["channels"], original["kernels"]).to(args.device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=original["lr"], weight_decay=original["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss(weight=case.weights.to(args.device))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and args.device.type == "cuda")
    best, bad, history = -float("inf"), 0, []
    start = time.perf_counter()
    for epoch in range(1, original["epochs"] + 1):
        loss = train_epoch(model, train_loader, optimizer, scaler, criterion, args.device, args.amp)
        dev, _, _ = control.evaluate(model, dev_loader, args.device, args.amp)
        scheduler.step(dev["MCC"])
        history.append({"epoch": epoch, "train_loss": loss, "dev": dev, "elapsed_seconds": time.perf_counter() - start})
        control.save_json(directory / "history.json", history)
        print(case.task, variant, seed, epoch, "dev_MCC", dev["MCC"], flush=True)
        if dev["MCC"] > best:
            best, bad = dev["MCC"], 0
            torch.save({"model_state_dict": model.state_dict(), "spec": spec, "epoch": epoch}, directory / "best.tmp.pt")
            (directory / "best.tmp.pt").replace(directory / "best.pt")
        else:
            bad += 1
            if bad >= original["patience"]:
                break
    training_seconds = time.perf_counter() - start
    checkpoint = torch.load(directory / "best.pt", map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    # Test data are only opened after the student's dev-selected checkpoint is frozen.
    test = control.read_rows(case.directory / "test.jsonl")
    if variant == "full":
        contexts["test"], events["test"] = cache.get(case.task, "test", test)
    else:
        contexts["test"] = np.zeros((len(test), 1), dtype="float32")
    loader = DataLoader(control.tensors(test, length, contexts["test"]), batch_size=original["batch_size"], shuffle=False)
    metrics, labels, probabilities = control.evaluate(model, loader, args.device, args.amp)
    with (directory / "test_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "task", "seed", "variant", "label", "pred"] + [f"prob_class_{c}" for c in range(case.classes)])
        for row, label, values in zip(test, labels, probabilities):
            writer.writerow([row["id"], case.task, seed, variant, int(label), int(values.argmax())] + values.tolist())
    result = {"status": "success", "spec": spec, "test": {**metrics, "rows": len(test)}, "best_epoch": checkpoint["epoch"], "best_dev_MCC": best, "epochs_completed": len(history), "checkpoint_sha256": control.sha(directory / "best.pt"), "predictions_sha256": control.sha(directory / "test_predictions.csv"), "teacher_off_test_MCC": base["test"]["MCC"], "paired_MCC_difference_on_minus_off": metrics["MCC"] - base["test"]["MCC"], "student_training_seconds_excluding_cache_and_teacher": training_seconds, "cache_events": events, "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    control.save_json(result_path, result)
    del model, optimizer, checkpoint
    gc.collect()
    if args.device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fair-directory", type=Path, default=HERE / "fair_comparison")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.device = torch.device(args.device)
    args.model_dir, args.context_cache = ROOT / "models/dnabert2", HERE / "context_cache"
    args.amp = args.device.type == "cuda"
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.output / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        baseline = no_teacher_records()
        protocol = {"phase": "paired_teacher_intervention", "tasks": TASKS, "variants": VARIANTS, "seeds": SEEDS, "new_student_fits": 24, "reused_no_teacher_fits": 24, "alpha": ALPHA, "temperature": TEMPERATURE, "loss": "Original weighted hard-label CE + alpha * T^2 * KL(teacher_softmax_T || student_softmax_T)", "teacher": "One fair-study dev-selected seed42 fine-tuned DNABERT-2 checkpoint per task, reused across student seeds. Only exact-ID-aligned training logits are used.", "selection": "All student optimization settings match the retained no-teacher paired controls; no teacher/student hyperparameter search or test-based selection.", "source_sha256": {str(p.relative_to(ROOT)): control.sha(p) for p in (Path(__file__), HERE / "run_controlled_experiments.py", HERE / "run_fair_baselines.py", ROOT / "src/models/bioqpred_v2.py")}, "scope": "Separate teacher intervention, not a change to the completed hard-label ablations. Training logits are in-sample, not OOF. Timing is recorded for cost accounting, not a controlled speed comparison."}
        protocol_path = args.output / "protocol.json"
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != json.loads(json.dumps(protocol)):
            raise ValueError("Existing teacher protocol differs")
        control.save_json(protocol_path, protocol)
        options = SimpleNamespace(smoke=False, split_dir=ROOT / "data/splits/nt_tasks", model_dir=args.model_dir, context_cache=args.context_cache, device=args.device, amp=args.amp)
        results = []
        for task in TASKS:
            case = fair.TaskCase(options, task)
            logits, metadata = teacher_logits(case, args.fair_directory, args.output / task / "teacher_cache")
            for seed in SEEDS:
                for variant in VARIANTS:
                    control.save_json(args.output / "status.json", {"state": "training", "task": task, "variant": variant, "seed": seed, "completed_fits": len(results), "pid": os.getpid()})
                    results.append(student(case, variant, seed, baseline[(task, variant, seed)], logits, metadata, args))
            del case, logits
            gc.collect()
        control.save_json(args.output / "status.json", {"state": "complete", "completed_fits": len(results), "reused_paired_off": len(baseline), "pid": os.getpid()})


if __name__ == "__main__":
    main()
