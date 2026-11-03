# Experiment Reproduction

Run all commands from the repository root on Linux/WSL. Prepare the pinned data
first. Full/context, CAMP-style, TFBS-style and DNABERT baselines require the
external backbone. Commands below do not launch a remote scheduler or alter any
other process. Choose your own CUDA device and available resources.

## Fixed-LR Component And Teacher Controls

```bash
python revision_20261022/run_controlled_experiments.py \
  --variants local_comp full --seeds 42 2024 3407 --device cuda:0 --amp \
  --output revision_20261022/branch_ablation
python revision_20261022/run_controlled_experiments.py \
  --variants local_only without_comp without_position without_local \
  --seeds 42 2024 3407 --device cuda:0 --amp \
  --output revision_20261022/component_ablation
```

These commands cover six configurations, four tasks and three seeds: 72 fits.
The `local_comp` configuration is the no-context control; `without_local` removes
the entire local branch. Defaults are AdamW LR 1e-4, weight decay .01, batch 16,
15 maximum epochs, patience 4, dev-MCC checkpoint selection and gradient clipping
at 1. AMP is FP16 on CUDA; seeding does not guarantee bitwise identity for every
CUDA pooling/attention kernel.

Teacher distillation additionally requires the completed DNABERT fine-tuning
search described below and all 24 `full`/`local_comp` teacher-off controls in the
fixed directories above:

```bash
python revision_20261022/run_teacher_ablation.py \
  --fair-directory revision_20261022/fair_comparison \
  --output revision_20261022/teacher_ablation --device cuda:0
```

The seed-42 teacher is selected by dev MCC from exactly three learning rates.
Only training logits are distilled (in-sample, not out-of-fold). Student loss is
hard-label CE plus `0.5 * T^2 * KL`, with temperature 2. No teacher logits from
dev/test are used for student training. The teacher is not needed for local
student inference. Do not call the tuned BioQPred-vs-baseline table a pure
teacher effect; use these paired controls.

## Matched Baseline Search

```bash
python revision_20261022/run_fair_baselines.py \
  --methods bio_full bio_local_comp cnn dnabert_frozen dnabert_finetune \
  camp_adapted_masked tfbs_adapted_masked \
  --output revision_20261022/fair_comparison --device cuda:0 --amp
```

| Methods | Three seed-42 learning rates |
|---|---|
| `bio_full`, `bio_local_comp` | 3e-5, 1e-4, 3e-4 |
| `cnn`, `camp_adapted_masked`, `tfbs_adapted_masked` | 1e-4, 3e-4, 1e-3 |
| `dnabert_frozen` | 3e-4, 1e-3, 3e-3 |
| `dnabert_finetune` | 1e-5, 2e-5, 5e-5 |

The selected LR maximizes dev MCC (ties choose the smaller LR), then is reused
for seeds 2024/3407. Effective batch is 16; DNABERT fine-tuning uses microbatch 4
and accumulation. Seven methods by four tasks give 28 groups; a fully successful
search would make five fits and three tests per group. CAMP/TFBS/NeuGSE are
declared architectural adaptations, not exact source-paper task reproductions.

**Failure handling:** the study retained four specific CAMP seed-42 LR=.001
nonfinite-loss failures, one per task, under explicit per-case author approval.
The three splice tasks selected .0001; H3K27me3 selected .0003 from the remaining
finite dev candidates. The accepted historical matrix was 136 successful fits
plus those four failures, with 84 final tests. The original runner is shipped
unchanged and stops on a nonfinite loss. This package does not silently add a
general failure exclusion/retry policy, fabricate successful trials, or replay
private recovery controllers. Fresh failures require review and a recorded
scientific decision; historical exception approvals do not cover new failures.

## Kernel, Capacity And Channel Sensitivity

Use seed 42 and validation-only evaluation; do not select architectures on test.
Change only the configuration declared in each run and keep the budget fixed.

```bash
python revision_20261022/run_controlled_experiments.py \
  --tasks splice_sites_all splice_sites_acceptor splice_sites_donor \
  --variants full --seeds 42 --kernels 3 15 --channels 128 \
  --validation-only --device cuda:0 --amp --output outputs/kernel_3_15
```

The additional four configurations per splice task were `[3,15]/128`,
`[3,7,15]/128`, `[7]/642`, and `[3,7,15]/213`. The latter two were chosen by
parameter count before observing scores, to approximate the default five-kernel
capacity. For channel sensitivity, run 64/128/256 **per branch** with the same
five kernels and distinct output directories. Single-kernel and other original
controls use the same CLI. Twelve added controls plus the original 54 are 66
single-seed validation entries, not 66 final tests or multi-seed robustness.

## Matched OpenSpliceAI Candidate Classification

The optional upstream source must be installed separately:

```bash
mkdir -p revision_20261022/research_dependencies
git clone https://github.com/Kuanhao-Chao/OpenSpliceAI.git \
  revision_20261022/research_dependencies/OpenSpliceAI
git -C revision_20261022/research_dependencies/OpenSpliceAI checkout \
  13ff432c211d0871eec27a1884d59df0a99df5eb
python revision_20261022/run_openspliceai_matched.py \
  --output outputs/openspliceai_matched --device cuda:0 --preflight-only
python revision_20261022/run_openspliceai_matched.py \
  --output outputs/openspliceai_matched --device cuda:0
```

The adapter checks three upstream file/license hashes. It uses the original
randomly initialized residual module, 32 channels and its native three-channel
output. Each 600-bp observed candidate window is padded with 1,000 zeros on each
side for the 2,000-bp receptive-field context. Anchors/orientation are inferred
from training motifs, not test labels. The task-channel versus other-channel
log-odds is read at the candidate and maximized across strands for binary
acceptor/donor classification. It is **not** the pretrained ensemble or the
original dense whole-genome evaluation. OpenSpliceAI is not inherently limited
to binary output. Its test accuracy and resource use must be measured, not
estimated from depth or parameter counts.

## NeuGSE-Style Comparator

```bash
CUDA_VISIBLE_DEVICES=0 python revision_20261022/run_neugse_matched.py \
  --output outputs/neugse_matched
```

This preserved comparator uses three dev-only LR candidates (.0001/.0003/.001)
and three selected-LR test seeds for each task: 20 fits/12 final tests. It is a
GPU-only original entry point. Do **not** use its legacy
`--wait-for-overlap-pid` option in this source release; cluster queues and their
historical predecessor output are deliberately not distributed. The only
release adaptation redirects its two read-only runtime helpers into a small
standalone module; no model, split, loss, seed or budget is changed.

## Dataset, Genomic And Motif Audits

`verify_pinned_nt.py` contains the original strict FASTA/header, partition and
stratified-split checks. `scripts/prepare_data.py` adds portable downloads and
exact split hashes. `map_genomic_controls.py --help` documents reference mapping
and overlap checks; supply your own reference/annotations and external minimap2
where needed. Mapping to an available reference is not proof of the original
genome build or full biological independence.

For the completed no-teacher `full`/`local_comp` controls:

```bash
python revision_20261022/run_full_branch_masking.py \
  --inputs revision_20261022/branch_ablation \
  --tasks splice_sites_acceptor splice_sites_donor \
  --output outputs/fixed_masking --device cuda:0 --amp
```

The script predeclares fixed zero-based candidate-core, equal-count flank and
central-window conditions. It replaces the specified positions by N in every
sample and recomputes **all** sequence-derived branches, including composition
and context. It does not retrain or select a checkpoint. N masking is an
out-of-distribution sensitivity test, not causal biological proof.

`rebuild_prediction_pfms.py` and `run_spliceai_reference.py` are preserved
advanced/legacy analysis entry points. The former expects the historical
`reports/bioqpred_v2_summary.csv` and `runs/bioqpred_v2/predictions` layout, not
arbitrary new runner outputs. The latter's actual pretrained SpliceAI inference
requires separately installed TensorFlow/SpliceAI and pretrained assets; its
motif helpers import without them. Those external assets and legacy result
archives are not bundled. No claim is made that one command rebuilds every
manuscript table from this source-only ZIP.

## Efficiency Protocol

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark.py \
  --task splice_sites_acceptor \
  --fair-directory revision_20261022/fair_comparison \
  --output outputs/efficiency_acceptor
```

Requires prepared data, pretrained weights and the selected fine-tuned teacher.
The public wrapper calls the original `efficiency_measurement_v1.py`, without
private queue/process dependencies. It refuses an existing output. Explicitly
reserve a suitable GPU before starting; it does not acquire a cluster GPU lock,
kill other processes, or enforce the original multi-controller admission gate.

Per task: three seeds by five student conditions (local-only/local+composition/
full teacher-off, local+composition/full teacher-on). Each measures **one full
27,000-example epoch after three warm-up updates**, not an entire 15-epoch fit.
Batch is 16; the separate teacher epoch uses microbatch 4/accumulation 4.
Measure three passes of the 3,000-example dev set, batch-1 latency on 64 dev
examples, synchronized CUDA timing, and per-process peak allocated/reserved
memory. Cold FP32 train/dev context extraction, warm-cache reads, teacher logit
extraction, and teacher costs are reported separately. No test predictions or
accuracy fits are generated. This BioQPred protocol is not itself an
OpenSpliceAI resource-comparison result.

The wrapper records GPU UUID, compute-process identities and load every five
seconds. Other activity must be disclosed; do not turn shared-load timing or
training pauses into an unconditional speed advantage. Model-only cached
throughput is not end-to-end throughput including fresh feature extraction.
Functional tests and checksum validation do not constitute new timing results.
