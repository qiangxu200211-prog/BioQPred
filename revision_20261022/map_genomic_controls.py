"""Map benchmark sequences to the available reference for locus-overlap checks.

Mapping to a local reference is evidence about projected loci, not proof of the
benchmark's original genome build, chromosome split, or gene annotations.
"""

import argparse
import bisect
import csv
import hashlib
import itertools
import json
import time
from collections import Counter, defaultdict
from pathlib import Path


TASKS = ("splice_sites_all", "splice_sites_acceptor", "splice_sites_donor", "H3K27me3")


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def interval_index(rows):
    by_chrom = defaultdict(list)
    for row in rows:
        by_chrom[row["chrom"]].append((row["start"], row["end"]))
    result = {}
    for chrom, intervals in by_chrom.items():
        intervals.sort()
        starts, prefix_ends = [], []
        maximum = -1
        for start, end in intervals:
            starts.append(start)
            maximum = max(maximum, end)
            prefix_ends.append(maximum)
        result[chrom] = starts, prefix_ends
    return result


def nearest_gap(index, chrom, start, end):
    if chrom not in index:
        return None, False
    starts, ends = index[chrom]
    stop = bisect.bisect_left(starts, end)
    overlap = stop > 0 and ends[stop - 1] > start
    if overlap:
        return 0, True
    candidates = []
    if stop:
        candidates.append(max(0, start - ends[stop - 1]))
    if stop < len(starts):
        candidates.append(max(0, starts[stop] - end))
    return min(candidates), False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-per-split", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    import mappy

    args.output.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    protocol = {"reference": str(args.reference.resolve()), "reference_sha256": sha(args.reference), "mappy_version": mappy.__version__, "preset": "sr", "minimum_query_coverage": 0.98, "minimum_alignment_identity": 0.99, "minimum_mapq": 30, "limit_per_split": args.limit_per_split, "coordinate_convention": "zero_based_half_open", "locus_bin_bp": 100000, "scope": "Projected mapping to the available reference only; unmapped/ambiguous sequences remain excluded from coordinate claims."}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    print("INDEX reference", args.reference, flush=True)
    aligner = mappy.Aligner(str(args.reference), preset="sr", n_threads=args.threads, best_n=10)
    if not aligner:
        raise RuntimeError("Reference indexing failed")
    print("INDEX complete", len(aligner.seq_names), "contigs", flush=True)
    mapped_groups, mapping_cache, summaries = {}, {}, []
    input_manifest = []
    fields = ["id", "task", "split", "label", "sequence_sha256", "length", "accepted", "reason", "chrom", "start", "end", "strand", "query_coverage", "identity", "mapq", "returned_alignments"]
    with (args.output / "sequence_mappings.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for task in TASKS:
            mapped_groups[task] = {}
            for split in ("train", "dev", "test"):
                source = args.root / "data/splits/nt_tasks" / task / (split + ".jsonl")
                input_manifest.append({"path": str(source), "sha256": sha(source)})
                accepted, reasons, total = [], Counter(), 0
                with source.open() as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        if args.limit_per_split and total >= args.limit_per_split:
                            break
                        sample = json.loads(line)
                        seq = sample["sequence"].upper()
                        key = hashlib.sha256(seq.encode()).hexdigest()
                        if key not in mapping_cache:
                            hits = list(aligner.map(seq))
                            match = {"accepted": False, "reason": "unmapped", "chrom": "", "start": "", "end": "", "strand": "", "query_coverage": 0.0, "identity": 0.0, "mapq": 0, "returned_alignments": len(hits)}
                            if hits:
                                best = max(hits, key=lambda h: (h.is_primary, h.mapq, h.mlen))
                                coverage = (best.q_en - best.q_st) / len(seq)
                                identity = best.mlen / max(1, best.blen)
                                passed = coverage >= 0.98 and identity >= 0.99 and best.mapq >= 30
                                match.update(accepted=passed, reason="accepted" if passed else "ambiguous_or_low_coverage_identity", chrom=best.ctg, start=best.r_st, end=best.r_en, strand=best.strand, query_coverage=coverage, identity=identity, mapq=best.mapq)
                            mapping_cache[key] = match
                        row = {"id": sample["id"], "task": task, "split": split, "label": sample["label"], "sequence_sha256": key, "length": len(seq), **mapping_cache[key]}
                        writer.writerow(row)
                        total += 1
                        reasons[row["reason"]] += 1
                        if row["accepted"]:
                            accepted.append(row)
                        if total % 3000 == 0:
                            print("MAPPED", task, split, total, "accepted", len(accepted), flush=True)
                mapped_groups[task][split] = accepted
                summaries.append({"task": task, "split": split, "rows": total, "accepted_rows": len(accepted), "accepted_fraction": len(accepted) / total if total else 0, "reason_counts": dict(reasons), "accepted_chromosomes": dict(Counter(r["chrom"] for r in accepted))})
                output.flush()
    overlap_rows = []
    for task, splits in mapped_groups.items():
        for left, right in itertools.combinations(("train", "dev", "test"), 2):
            index = interval_index(splits[left])
            distances = [nearest_gap(index, r["chrom"], r["start"], r["end"]) for r in splits[right]]
            left_bins = {(r["chrom"], ((r["start"] + r["end"]) // 2) // 100000) for r in splits[left]}
            right_bins = {(r["chrom"], ((r["start"] + r["end"]) // 2) // 100000) for r in splits[right]}
            overlap_rows.append({"task": task, "left": left, "right": right, "left_accepted_rows": len(splits[left]), "right_accepted_rows": len(splits[right]), "shared_chromosomes": sorted(set(index) & {r["chrom"] for r in splits[right]}), "right_rows_with_reference_interval_overlap": sum(overlap for _, overlap in distances), "right_rows_within_1000bp": sum(gap is not None and gap <= 1000 for gap, _ in distances), "right_rows_within_10000bp": sum(gap is not None and gap <= 10000 for gap, _ in distances), "shared_100kb_bins": len(left_bins & right_bins)})
    report = {"protocol": protocol, "split_mapping_summary": summaries, "projected_overlap_summary": overlap_rows, "input_manifest": input_manifest, "elapsed_seconds": time.perf_counter() - start_time, "reference_contigs": {name: len(aligner.seq(name)) for name in aligner.seq_names}, "limitations": ["No original benchmark build or split provenance is inferred from a successful alignment.", "Unmapped and ambiguous rows are not treated as biologically independent.", "No gene-disjointness claim is possible without verified gene annotations.", "A 100-kb projected-locus bin is a diagnostic grouping, not a reconstructed official split."]}
    (args.output / "mapping_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(summaries, indent=2), flush=True)
    print("COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
