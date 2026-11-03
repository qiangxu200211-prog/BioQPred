"""Rebuild the pinned public benchmark and verify all twelve split hashes."""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "revision_20261022"))
from verify_pinned_nt import read_fna


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def download(url, path, expected):
    if path.exists():
        if path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
            raise ValueError(f"Existing file does not match the pinned source: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as source, temporary.open("wb") as target:
        while True:
            block = source.read(1048576)
            if not block:
                break
            target.write(block)
    if temporary.stat().st_size != expected["bytes"] or sha(temporary) != expected["sha256"]:
        raise ValueError(f"Downloaded content failed SHA256/size validation: {temporary}")
    temporary.replace(path)


def split_rows(task, split, rows):
    return [
        {"id": f"{task}|{split}|{index}", "task": task,
         "sequence": row["sequence"], "label": row["label"],
         "label_name": str(row["label"]), "split": split,
         "sequence_length": len(row["sequence"]), "source_id": row["id"],
         "source_split": row["source_split"]}
        for index, row in enumerate(rows)
    ]


def save_split(path, rows, expected):
    if len(rows) != expected["rows"]:
        raise ValueError(f"Unexpected row count for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if sha(path) != expected["sha256"]:
            raise ValueError(f"Refusing to overwrite a different split: {path}")
        return
    temporary = path.with_suffix(".jsonl.part")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if sha(temporary) != expected["sha256"]:
        raise ValueError(f"Reconstructed split differs from the executed study: {temporary}")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=("splice_sites_all", "splice_sites_acceptor", "splice_sites_donor", "H3K27me3"))
    parser.add_argument("--upstream-dir", type=Path, default=ROOT / "data/upstream_pinned")
    parser.add_argument("--output", type=Path, default=ROOT / "data/splits/nt_tasks")
    parser.add_argument("--offline", action="store_true", help="Use already downloaded, hash-verified FASTA files only")
    args = parser.parse_args()
    from sklearn.model_selection import train_test_split

    manifest = json.loads((ROOT / "configs/data_sources.json").read_text())
    expected = json.loads((ROOT / "configs/expected_splits.json").read_text())
    tasks = args.tasks or list(expected["tasks"])
    for task in tasks:
        raw = {}
        for split in ("train", "test"):
            entry = next(e for e in manifest["files"] if e["task"] == task and e["split"] == split)
            path = args.upstream_dir / task / (split + ".fna")
            if args.offline:
                if not path.is_file() or sha(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
                    raise ValueError(f"Offline source is missing or changed: {path}")
            else:
                url = f"https://huggingface.co/datasets/{manifest['repository']}/resolve/{manifest['revision']}/{entry['upstream_path']}"
                download(url, path, entry)
            raw[split] = [dict(row, id=f"{task}|{split}|{i}", source_split=split) for i, row in enumerate(read_fna(path))]
        train, dev = train_test_split(raw["train"], test_size=0.1, random_state=42,
                                     shuffle=True, stratify=[r["label"] for r in raw["train"]])
        for split, records in (("train", train), ("dev", dev), ("test", raw["test"])):
            save_split(args.output / task / (split + ".jsonl"), split_rows(task, split, records), expected["tasks"][task][split])
            print(f"VERIFIED {task}/{split}: {len(records)} rows", flush=True)
    print("All requested splits match the original study byte for byte.")


if __name__ == "__main__":
    main()
