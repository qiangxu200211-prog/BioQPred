# Release Checks

The source-only archive was checked on 2026-10-30. These are software packaging
and reproducibility checks, **not new paper experiments**.

| Check | Outcome |
|---|---|
| Executed-source provenance | 26 scientific/runtime files compared with the executed source hashes; 25 unchanged, one declared management-only NeuGSE import adaptation |
| CPU unit tests | 8/8 passed on Python 3.9.23 / PyTorch 2.8.0 |
| Core configurations | Forward/backward with synthetic inputs for all six controlled variants |
| Prediction helper | Synthetic trusted local-student checkpoint produced ID-aligned probabilities |
| Dataset reconstruction | All 12 train/dev/test files matched the study SHA256 byte for byte, including order |
| External backbone | Every installed tensor matched the study checkpoint using the local safetensors input path |
| Full configuration | Real frozen-context extraction on two synthetic sequences and a finite full-model forward pass |
| Controlled runner | CPU smoke mode completed 64 train / 32 dev rows; `test=None` |
| Optional OpenSpliceAI | Pinned upstream module/license checks and original CPU preflight passed using an external checkout |
| CLI entry points | Baseline, teacher, specialist, NeuGSE, masking, genomic mapping and public benchmark help commands passed |
| Archive | Every member's payload hash checked against the release tree; ZIP CRC check passed |
| Private material scan | No supplied server password, server IP, private absolute home/work paths or private-key/token signatures found |

The weight-download network path was not re-downloaded as part of these checks;
the offline external-weight input path was tested. No new complete multi-seed
accuracy study, teacher training or efficiency comparison was run for packaging.
TensorFlow pretrained SpliceAI inference, a new clean-environment pip install,
and legacy PFM/result-archive workflows were not exercised. The exact scientific
environment versions are in `requirements.txt`; optional legacy dependencies
are documented separately.

Temporary reconstruction data and copied weights were kept outside the source
ZIP. Internal compute admission/recovery orchestration is intentionally not a
part of this public source release. The new benchmark wrapper only exposes the
original scientific measurement kernel and does not assert cluster isolation.
