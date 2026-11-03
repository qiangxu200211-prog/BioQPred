"""Compare every local row to cryptographically pinned upstream FASTA content."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


REVISION = "851f9946252e90c665cdb3cc3eedb78f1f26197c"
TASKS = ("splice_sites_all", "splice_sites_acceptor", "splice_sites_donor", "H3K27me3")


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def read_fna(path):
    records, header, pieces = [], None, []

    def append():
        if header is None:
            return
        match = re.fullmatch(r"([^:]+):(\d+)-(\d+)\|(\d+)", header)
        if match is None:
            raise ValueError("Unexpected upstream header; do not invent coordinates or labels")
        chrom, start, end, label = match.groups()
        sequence = "".join(pieces).upper()
        if int(end) - int(start) != len(sequence) or set(sequence) - set("ACGTN"):
            raise ValueError("Header span does not agree with the actual sequence length/alphabet")
        records.append({"header": header, "chrom": chrom, "start": int(start), "end": int(end), "label": int(label), "sequence": sequence})

    with path.open(encoding="ascii") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                append()
                header, pieces = line[1:], []
            else:
                if header is None:
                    raise ValueError("Sequence appears before the first FASTA header")
                pieces.append(line)
    append()
    return records


def read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def payload(row):
    return row["sequence"].upper(), int(row["label"])


def check_partition(upstream, local, source_split):
    indices, mismatches = [], []
    for row in local:
        if row.get("source_split") != source_split:
            raise ValueError("Local source_split metadata does not match its declared source")
        try:
            index = int(row["source_id"].rsplit("|", 1)[1])
        except (KeyError, ValueError, IndexError) as error:
            raise ValueError("Local source-row index is missing or malformed") from error
        indices.append(index)
        if index < 0 or index >= len(upstream) or payload(row) != payload(upstream[index]):
            mismatches.append(row["id"])
    complete_indices = len(indices) == len(upstream) and len(set(indices)) == len(upstream) and set(indices) == set(range(len(upstream)))
    content_match = Counter(payload(r) for r in local) == Counter(payload(r) for r in upstream)
    return {"upstream_rows": len(upstream), "local_rows": len(local), "source_index_bijection": complete_indices, "ordered_source_index_mismatches": len(mismatches), "first_mismatching_ids": mismatches[:10], "content_multiset_match": content_match, "exact_source_index_match": complete_indices and not mismatches and content_match}


def csv_write(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-split", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.upstream / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "success" or manifest["revision"] != REVISION:
        raise ValueError("All eight pinned upstream downloads must be complete")
    files = {(r["task"], r["split"]): r for r in manifest["files"]}
    if len(manifest["files"]) != 8 or set(files) != {(task, split) for task in TASKS for split in ("train", "test")}:
        raise ValueError("Missing or duplicated upstream files")
    report, coordinate_rows, inputs = [], [], []
    for task in TASKS:
        upstream = {}
        for split in ("train", "test"):
            path = args.upstream / task / (split + ".fna")
            expected = files[(task, split)]["pointer"]
            if sha(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
                raise ValueError("Pinned upstream file changed after download verification")
            upstream[split] = read_fna(path)
            inputs.append({"file": str(path), "sha256": expected["sha256"]})
        local = {split: read_jsonl(args.split_dir / task / (split + ".jsonl")) for split in ("train", "dev", "test")}
        for split in local:
            path = args.split_dir / task / (split + ".jsonl")
            inputs.append({"file": str(path), "sha256": sha(path)})
        checks = {"train_plus_dev": check_partition(upstream["train"], local["train"] + local["dev"], "train"), "test": check_partition(upstream["test"], local["test"], "test")}
        exact = all(r["exact_source_index_match"] for r in checks.values())
        chromosomes = {split: dict(sorted(Counter(r["chrom"] for r in data).items())) for split, data in upstream.items()}
        intersection = sorted(set(chromosomes["train"]) & set(chromosomes["test"]))
        split_check = {"performed": False}
        if args.check_split and exact:
            import sklearn
            from sklearn.model_selection import train_test_split
            train_indices, dev_indices = train_test_split(list(range(len(upstream["train"]))), test_size=0.1, random_state=42, shuffle=True, stratify=[r["label"] for r in upstream["train"]])
            actual_train = [int(r["source_id"].rsplit("|", 1)[1]) for r in local["train"]]
            actual_dev = [int(r["source_id"].rsplit("|", 1)[1]) for r in local["dev"]]
            split_check = {"performed": True, "sklearn": sklearn.__version__, "seed": 42, "dev_fraction": 0.1, "train_order_matches": train_indices == actual_train, "dev_order_matches": dev_indices == actual_dev}
            if not split_check["train_order_matches"] or not split_check["dev_order_matches"]:
                raise ValueError("The declared stratified seed-42 split did not replay exactly")
        report.append({"task": task, "exact_source_index_match": exact, "partitions": checks, "upstream_chromosome_counts": chromosomes, "upstream_train_test_shared_chromosomes": intersection, "seed42_split_replay": split_check})
        if exact:
            for split, data in local.items():
                source = "test" if split == "test" else "train"
                for row in data:
                    index = int(row["source_id"].rsplit("|", 1)[1])
                    original = upstream[source][index]
                    coordinate_rows.append({"task": task, "split": split, "id": row["id"], "source_id": row["source_id"], "source_split": source, "upstream_row": index, "label": row["label"], "upstream_header": original["header"], "upstream_chromosome": original["chrom"], "upstream_start": original["start"], "upstream_end": original["end"], "sequence_sha256": hashlib.sha256(original["sequence"].encode()).hexdigest()})
    complete = all(r["exact_source_index_match"] for r in report)
    if coordinate_rows:
        csv_write(args.output / "upstream_coordinate_manifest.csv", coordinate_rows)
    result = {"status": "success" if complete else "mismatch", "revision": REVISION, "download_manifest_sha256": sha(manifest_path), "tasks": report, "coordinate_rows_recovered": len(coordinate_rows), "inputs": inputs, "limitations": ["Source chromosomes/coordinates are recovered from verified upstream headers, not inferred by aligning to another assembly.", "Header coordinate indexing and reference-assembly identity require separate verification before extracting genomic sequence or overlaying annotations.", "Matching a release supports its documented provenance; it does not independently reproduce peak sampling or exclude pretrained exposure.", "Chromosome holdout does not exclude sequence homology, paralogy or all forms of biological dependence."]}
    (args.output / "lineage_report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"status": result["status"], "coordinate_rows_recovered": len(coordinate_rows), "tasks": report}, indent=2), flush=True)
    if not complete:
        raise SystemExit("Pinned source matching failed; preserve the mismatch report")


if __name__ == "__main__":
    main()
