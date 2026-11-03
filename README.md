# BioQPred

[中文使用说明](README.zh-CN.md)

Source release for BioQPred candidate-window DNA sequence classification and the
controlled experiments used in the paper revision.

**本仓库包含论文模型和实验复现代码，不包含原始数据、训练权重、服务器管理脚本、
审稿材料或私密配置。** 数据版本与原实验划分由 SHA256 固定；可按下面步骤重新下载并构建。
开源协议尚待作者确定，第三方代码保留其自身许可。代码打包测试不等于重新完成论文全量实验。

## Model Overview

BioQPred separates local sequence patterns, composition features, and optional
frozen DNABERT-2 context. The released scientific implementation uses concatenation
fusion, not the optional gated-fusion variant.

| Configuration | Local branch | Composition | Frozen DNABERT-2 context |
|---|---|---|---|
| `local_only` | Yes | No | No |
| `local_comp` | Yes | Yes | No |
| `full` | Yes | Yes | Yes |
| `without_comp` | Yes | No | Yes |
| `without_position` | Without learned position embeddings | Yes | Yes |
| `without_local` | No | Yes | Yes |

The local branch uses learned position embeddings, parallel convolution kernels
`3, 5, 7, 11, 15`, dilated convolutions, channel attention, and spatial pyramid
pooling. The default **128 channels are per kernel branch**; five branches are
concatenated into 640 channels. Composition is a deterministic 12-value vector.
The context branch uses attention-mask-aware mean/max pooling of a frozen
DNABERT-2 backbone. A local student does not require that backbone at inference;
a full model does, unless its input-specific context has already been cached.

Optional teacher distillation is a training intervention, not a fourth inference
branch. It must be evaluated against matched fixed-learning-rate teacher-off
controls rather than against separately tuned models.

## Repository Layout

```text
src/models/                  BioQPred and comparison model implementations
revision_20261022/           Preserved scientific training/analysis entry points
models/dnabert2/             Reviewed DNABERT-2 runtime and tokenizer, no weights
scripts/prepare_data.py      Pinned downloads and byte-exact split reconstruction
scripts/prepare_backbone.py  Verified external pretrained-weight installation
scripts/predict.py           Inference for controlled/teacher student checkpoints
scripts/benchmark.py         Public entry point for the original BioQPred timing protocol
scripts/check_integrity.py   Source-package SHA256 verification
configs/                    Dataset, split, backbone and source provenance hashes
tests/                      CPU functional checks
docs/EXPERIMENTS.md          Reproduction commands, protocols and scope
third_party/                Third-party attribution and license text
```

`revision_20261022` is retained because the original runners derive the project
root from their own location. Do not move those files to another directory.

## Installation

Use **Linux or WSL2**, Python 3.9.23 (the recorded environment) and a compatible
PyTorch installation. Scientific runners use POSIX file locks (`fcntl`). CUDA is
recommended for full experiments; the small functional tests run on CPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the torch==2.8.0 wheel appropriate to your CUDA/CPU environment first.
python -m pip install -r requirements.txt
python scripts/check_integrity.py
python -m unittest discover -s tests -v
```

Always run the following commands from the **repository root**. External model
code is loaded locally with `trust_remote_code=True`; review the bundled runtime
before using it. Never load an untrusted PyTorch checkpoint.

## Dataset

The public source is
[InstaDeepAI/nucleotide_transformer_downstream_tasks_revised](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks_revised),
pinned to commit `851f9946252e90c665cdb3cc3eedb78f1f26197c`.

| Task | Window | Classes | Train | Dev | Test |
|---|---:|---:|---:|---:|---:|
| `splice_sites_all` | 600 bp | 3 | 27,000 | 3,000 | 3,000 |
| `splice_sites_acceptor` | 600 bp | 2 | 27,000 | 3,000 | 3,000 |
| `splice_sites_donor` | 600 bp | 2 | 27,000 | 3,000 | 3,000 |
| `H3K27me3` | 1,000 bp | 2 | 27,000 | 3,000 | 3,000 |

This is **132,000 task-records**, not a claim that every sequence across tasks is
unique or biologically independent. The official test split is preserved.
Dev is a stratified 10% split of the official training set, using random state 42.
The scripts check the exact content **and order** of all twelve resulting files.

```bash
python scripts/prepare_data.py
# Or prepare just one task first:
python scripts/prepare_data.py --tasks splice_sites_acceptor
```

Eight upstream FASTA files are downloaded over HTTPS and checked before use.
No Hugging Face token is needed for the public files. Existing mismatched files
are rejected instead of overwritten. For offline reconstruction, provide
`--offline --upstream-dir /path/to/pinned_fasta`, where canonical task directories
contain `train.fna` and `test.fna`.

Train/dev interval or shared-gene overlap is possible in this benchmark. A
chromosome-held-out test set does not establish independence of all partitions
or absence of pretraining exposure. See the genomic audit entry points in
`docs/EXPERIMENTS.md`.

## Quick Functional Run

No pretrained backbone is required for `local_comp` or `local_only`.

```bash
python revision_20261022/run_controlled_experiments.py \
  --tasks splice_sites_acceptor --variants local_comp --seeds 42 \
  --device cpu --smoke --output outputs/smoke_local
```

Smoke mode uses 64 training and 32 dev examples for one epoch. It does **not**
evaluate test samples and must not be reported as a scientific result. The test
file's checksum is still recorded as provenance. Use a new output directory for
each distinct protocol; do not mix smoke, validation-only and formal results.

## Formal Training

```bash
# Fixed-budget three-seed controls; no teacher.
python revision_20261022/run_controlled_experiments.py \
  --tasks splice_sites_acceptor --variants local_comp --seeds 42 2024 3407 \
  --epochs 15 --patience 4 --batch-size 16 --lr 0.0001 \
  --device cuda:0 --amp --output outputs/acceptor_local_comp
```

For a full/context configuration, install external DNABERT-2 weights first:

```bash
python scripts/prepare_backbone.py
python revision_20261022/run_controlled_experiments.py \
  --tasks splice_sites_acceptor --variants full --seeds 42 2024 3407 \
  --device cuda:0 --amp --output outputs/acceptor_full
```

The installer downloads the hash-pinned official checkpoint and checks **every
tensor** against the executed backbone before saving `model.safetensors`.
The pinned official download commit records the release dependency; it is not a
claim about the historical download date. Safetensors serialization may have a
different file hash while the verified tensor payload is identical. A local
original `.safetensors` can be supplied using `--from-file`.

The included DNABERT-2 runtime preserves the study's already-declared PyTorch
attention fallback, rather than silently re-enabling a custom Triton kernel.
Local configurations do not use or load pretrained weights.

## Prediction

Use `best.pt` from the controlled or teacher-student runner. The fair-comparison
checkpoint format is different and is intentionally rejected by this helper.
Input JSONL needs only unique `id` and `sequence` fields, with the exact task
window length and the `A/C/G/T/N` alphabet. Labels are not consumed.

```bash
python scripts/predict.py \
  --checkpoint /path/to/trusted/best.pt \
  --input /path/to/sequences.jsonl --output outputs/predictions.csv --device cpu
```

Output contains the ID, predicted class index and per-class probabilities. For
context variants, the external backbone is required and features are recomputed
or retrieved by an input-content-addressed cache.

## Other Experiments

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for component/teacher ablations,
matched learning-rate search, kernel/capacity/channel controls, the pinned
OpenSpliceAI candidate-classification adapter, NeuGSE-style baseline, genomic
checks, masking and separate efficiency measurements.

The source release does **not** imply that BioQPred outperforms every specialist.
Do not replace measured values with architecture-based estimates, report only
the best seed, or equate trainable parameter counts with end-to-end cost.

## Outputs And Reporting

Depending on the runner, each run records a protocol/specification, history,
checkpoint checksum, `result.json` or `fit_result.json`, and final predictions.
Dev MCC determines checkpoint/model selection. Test is evaluated only after
that selection. Report three-seed means and **sample** standard deviations
(`ddof=1`), retaining low-performing seeds and declared failures. Three seeds
share the same 3,000 test examples; 9,000 prediction rows are not 9,000
independent examples.

This package intentionally omits trained checkpoints, sequence data, caches,
full prediction archives, internal launch/recovery state, manuscript/reviewer
files, SSH configuration, secrets, and unrelated retrieval experiments.
It does not automatically replay the cluster's historical recovery operations.
See `configs/source_provenance.json` for unchanged scientific files and the
single management-only NeuGSE import adaptation.

## License And Attribution

**The authors have not yet selected a repository-wide license.** Packaging this
source does not grant a new MIT/Apache license. Select an appropriate license
before publishing it as an openly licensed repository, and verify third-party
obligations. Do not apply a new license to third-party files by implication.

DNABERT-2 runtime copyright headers are retained and its Apache-2.0 license is
included. OpenSpliceAI is an optional external **GPL-3.0** dependency, cloned only
when requested; its source/weights are not vendored here. See
[third_party/NOTICE.md](third_party/NOTICE.md). Dataset and pretrained-weight
terms remain with their respective sources. Add the final paper citation and
author/contact information when available; no DOI or author list is invented.

## Uploading To GitHub

Create an empty repository under your own account, then run these commands from
this directory (replace the URL with your actual repository):

```bash
git init
git add .
git diff --cached --stat
# Review the staged list: it must contain source and docs, not data or weights.
git commit -m "Release BioQPred source and reproduction instructions"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/BioQPred.git
git push -u origin main
```

The supplied ZIP is a clean source snapshot; no repository has been created or
pushed on the author's behalf.
