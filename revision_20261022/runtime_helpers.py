"""Public runtime helpers extracted from the original cluster queue.

Only the two read-only helpers used by NeuGSE are retained. No queue,
historical process IDs, remote execution, or signal-sending code is shipped.
"""

import subprocess
from pathlib import Path


def process_identity(pid):
    directory = Path("/proc") / str(pid)
    try:
        stat = (directory / "stat").read_text()
        tail = stat[stat.rfind(")") + 2:].split()
        command = (directory / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        return {"start_ticks": tail[19], "command": command, "state": tail[0]}
    except FileNotFoundError:
        return None


def gpu_free(index):
    uuid = subprocess.check_output(
        ["nvidia-smi", f"--id={index}", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    applications = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
    )
    return not any(line.split(",")[0].strip() == uuid for line in applications.splitlines())
