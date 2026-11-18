# BioQPred

[中文说明](README.zh-CN.md)

BioQPred is a DNA sequence classification model for splice-site and H3K27me3
prediction. It combines a local sequence encoder, nucleotide-composition
features, and frozen DNABERT-2 representations.

This repository contains the model implementations and scripts for data
preparation, training, evaluation, and the experiments reported in the paper.
Data and DNABERT-2 weights are downloaded using the preparation scripts.
Trained BioQPred checkpoints are not included.

## Model

The local encoder adds learned position embeddings before convolution and uses
five kernel sizes (3, 5, 7, 11, and 15), with 128 channels per branch. Dilated
convolutions, channel attention, and spatial pyramid pooling produce the local
representation. The composition branch takes 12 sequence-derived features, and
the context branch uses masked mean/max pooling of frozen DNABERT-2 outputs.
The resulting representations are concatenated for classification.

The training scripts provide the following configurations:

| Configuration | Components |
|---|---|
| `full` | Local encoder + composition + frozen DNABERT-2 |
| `local_comp` | Local encoder + composition |
| `local_only` | Local encoder |
| `without_comp` | Local encoder + frozen DNABERT-2 |
| `without_position` | Full model without position embeddings in the local encoder |
| `without_local` | Composition + frozen DNABERT-2 |

The `local_only` and `local_comp` configurations do not require a pretrained
backbone. Teacher distillation is available as a separate training experiment;
the teacher is not used for student inference.

## Installation

The experiments used Python 3.9.23 and PyTorch 2.8.0. Use Linux or WSL2 because
the training scripts depend on POSIX file locks. CUDA is recommended for
training; the functional checks can run on CPU.

```bash
git clone https://github.com/qiangxu200211-prog/BioQPred.git
cd BioQPred

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install PyTorch 2.8.0 for your CUDA or CPU environment first.
python -m pip install -r requirements.txt
```

Run all commands below from the repository root. To check the source files and
run the CPU functional tests:

```bash
python scripts/check_integrity.py
python -m unittest discover -s tests -v
```

## Data

The four tasks use
[InstaDeepAI/nucleotide_transformer_downstream_tasks_revised](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks_revised)
at revision `851f9946252e90c665cdb3cc3eedb78f1f26197c`.

| Task | Sequence length | Classes | Train | Validation | Test |
|---|---:|---:|---:|---:|---:|
| `splice_sites_all` | 600 bp | 3 | 27,000 | 3,000 | 3,000 |
| `splice_sites_acceptor` | 600 bp | 2 | 27,000 | 3,000 | 3,000 |
| `splice_sites_donor` | 600 bp | 2 | 27,000 | 3,000 | 3,000 |
| `H3K27me3` | 1,000 bp | 2 | 27,000 | 3,000 | 3,000 |

The official test sets are retained. Each validation set is a stratified 10%
split of the official training set, using random seed 42.

```bash
python scripts/prepare_data.py
```

The script downloads the source FASTA files and writes
`data/splits/nt_tasks/<task>/{train,dev,test}.jsonl`. SHA256 checks verify the
downloads and the reconstructed splits against the recorded experiment files.
To prepare one task, add `--tasks splice_sites_acceptor`. For existing FASTA
files, use `--offline --upstream-dir /path/to/pinned_fasta`.

Training and validation partitions can share genomic intervals or genes.
The genomic-overlap checks are described in
[the experiment documentation](docs/EXPERIMENTS.md).

## Training

### Quick run

After preparing the data, run a small CPU example:

```bash
python revision_20261022/run_controlled_experiments.py \
  --tasks splice_sites_acceptor --variants local_comp --seeds 42 \
  --device cpu --smoke --output outputs/smoke_local
```

This runs one epoch on 64 training and 32 validation examples, without test-set
evaluation.

### Local model

The following command trains the local encoder with composition features using
three random seeds:

```bash
python revision_20261022/run_controlled_experiments.py \
  --tasks splice_sites_acceptor --variants local_comp --seeds 42 2024 3407 \
  --epochs 15 --patience 4 --batch-size 16 --lr 0.0001 \
  --device cuda:0 --amp --output outputs/acceptor_local_comp
```

Training uses AdamW with an initial learning rate of `1e-4` and weight decay
`0.01`. The learning rate is reduced when validation MCC plateaus. The best
checkpoint is selected by validation MCC, with early stopping after four
epochs without improvement, and is then evaluated on the test set.

### Full model

Download the DNABERT-2 weights before training a configuration with the context
branch:

```bash
python scripts/prepare_backbone.py
python revision_20261022/run_controlled_experiments.py \
  --tasks splice_sites_acceptor --variants full --seeds 42 2024 3407 \
  --epochs 15 --patience 4 --batch-size 16 --lr 0.0001 \
  --device cuda:0 --amp --output outputs/acceptor_full
```

The preparation script verifies the pretrained tensors and saves
`models/dnabert2/model.safetensors`. A local weight file can be supplied with
`--from-file /path/to/model.safetensors`. The included DNABERT-2 runtime uses
PyTorch attention. Context features are cached for reuse.

Each run is saved in a task-specific subdirectory under `--output`, containing
`best.pt`, `spec.json`, `history.json`, and `result.json`. Runs with test
evaluation also write `test_predictions.csv`. To summarize the three-seed
results, use the mean and sample standard deviation (`ddof=1`).

## Prediction

Use a `best.pt` checkpoint from the controlled-experiment or teacher-student
training script. The prediction helper accepts JSONL with a unique string
`id` and a `sequence` on each line. Sequences must use the A/C/G/T/N alphabet
and match the task's window length; labels are not required.

```bash
python scripts/predict.py \
  --checkpoint /path/to/best.pt \
  --input /path/to/sequences.jsonl \
  --output outputs/predictions.csv --device cpu
```

The output CSV contains `id`, `prediction`, and one probability column per
class. Configurations with a context branch require DNABERT-2 weights to
process new sequences. Checkpoints from the baseline-comparison script use a
different format. Their test-set evaluation is handled by `run_fair_baselines.py`.

## Additional experiments

[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) provides commands and prerequisites
for component and teacher ablations, baseline comparisons, kernel and channel
analyses, genomic checks, sequence masking, and efficiency measurements.
CAMP-, TFBS-, and NeuGSE-style comparators are adaptations to these tasks.
The OpenSpliceAI comparison uses a candidate-window classification adapter.

## Repository structure

| Path | Contents |
|---|---|
| `src/models/` | BioQPred and baseline model definitions |
| `revision_20261022/` | Training, evaluation, and analysis scripts |
| `scripts/` | Data and weight preparation, prediction, and integrity checks |
| `models/dnabert2/` | DNABERT-2 configuration, tokenizer, and runtime code |
| `configs/` | Dataset versions, split hashes, and model metadata |
| `tests/` | CPU functional tests |
| `docs/EXPERIMENTS.md` | Detailed experiment protocols |
| `third_party/` | Third-party notices and license text |

## License and acknowledgments

A repository-wide license has not yet been specified. Third-party code and
dependencies retain their respective licenses. DNABERT-2 runtime code is
covered by Apache-2.0; the optional OpenSpliceAI dependency is GPL-3.0.
See [third_party/NOTICE.md](third_party/NOTICE.md) for source and license details.
