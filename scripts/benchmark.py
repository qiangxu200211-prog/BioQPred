"""Portable wrapper for the preserved BioQPred efficiency measurement kernel.

No admission lock or background scheduler is created. Reserve resources before
running. An existing output is rejected, including a partial failed attempt.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "revision_20261022"))
sys.path.insert(0, str(ROOT))


def observe():
    def query(options):
        return subprocess.check_output(["nvidia-smi"] + options, text=True, timeout=15).strip().splitlines()
    devices = query(["--query-gpu=index,uuid,name,memory.free,utilization.gpu", "--format=csv,noheader,nounits"])
    apps = query(["--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"])
    processes = []
    for line in apps:
        columns = [p.strip() for p in line.split(",")]
        identity = {"nvidia_smi": columns}
        try:
            pid = int(columns[1])
            proc = Path("/proc") / str(pid)
            stat = (proc / "stat").read_text()
            tail = stat[stat.rfind(")") + 2:].split()
            identity.update(pid=pid, uid=proc.stat().st_uid, start_ticks=tail[19], state=tail[0])
        except (OSError, ValueError, IndexError) as error:
            identity["identity_unknown"] = type(error).__name__
        processes.append(identity)
    return dict(unix=time.time(), gpu_rows=devices, compute_processes=processes,
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("splice_sites_all", "splice_sites_acceptor", "splice_sites_donor", "H3K27me3"))
    parser.add_argument("--fair-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT.resolve():
        parser.error("Run from the repository root so recorded teacher paths resolve consistently")
    if args.output.exists():
        parser.error("Use a new output directory; failed or completed measurements are never overwritten")
    import torch
    import run_fair_baselines as fair
    import run_teacher_ablation as teacher
    import efficiency_measurement_v1 as science

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        parser.error("Bind exactly one CUDA device with CUDA_VISIBLE_DEVICES")
    snapshot = observe()
    options = SimpleNamespace(smoke=False, split_dir=ROOT / "data/splits/nt_tasks",
                              model_dir=ROOT / "models/dnabert2", context_cache=args.output / "unused",
                              device=torch.device("cuda:0"), amp=True)
    case = fair.TaskCase(options, args.task)
    directory = args.fair_directory.resolve() / args.task / "dnabert_finetune"
    selected, result, provenance = teacher.selected_teacher(directory, case.hashes)
    teacher_record = dict(provenance=provenance,
                         selected_training_seconds_excluding_feature_cache=result["training_seconds_excluding_feature_cache"],
                         wall_time_is_not_controlled_efficiency_evidence=True)
    selection = json.loads((directory / "selection.json").read_text())
    teacher_record["search_observed_training_seconds_excluding_feature_cache"] = sum(
        json.loads((Path(row["directory"]) / "fit_result.json").read_text())["training_seconds_excluding_feature_cache"]
        for row in selection["trials"])
    plan = dict(root=str(ROOT), protocol=science.protocol(), splits={args.task: case.hashes})
    dependencies = dict(teacher_view=str(args.fair_directory.resolve()), teachers={args.task: teacher_record})
    del case
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = science.measure_task(plan, args.task, args.output, dependencies, SimpleNamespace(observe=observe))
    result["public_wrapper"] = dict(admission="Resources reserved by caller; no cluster lock or isolation assertion was verified",
                                    initial_observation=snapshot,
                                    no_new_accuracy_or_test_results=True)
    # The original kernel records this legacy field; do not retain an unverified assertion.
    result["isolated_from_own_accuracy_training"] = None
    fair.save_json(args.output / "benchmark.json", result)
    print("Measured one task. Inspect load/monitor records before making efficiency comparisons.")


if __name__ == "__main__":
    main()
