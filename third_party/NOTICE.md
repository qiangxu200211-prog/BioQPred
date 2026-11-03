# Third-Party Notices

## DNABERT-2 Runtime

`models/dnabert2/bert_layers.py`, `bert_padding.py`, `configuration_bert.py`,
configuration and tokenizer assets originate from the DNABERT-2/MosaicBERT
runtime used in the study. Original copyright notices are retained. The
implementation already contains a documented local change in `bert_layers.py`
disabling the custom Triton attention path in favor of the existing PyTorch
implementation. This release does not add a new attention implementation.
`config.json` retains the study configuration, including its local relative
model name. These assets and their hashes are distinguished from author code.

Upstream: https://github.com/MAGICS-LAB/DNABERT_2

Model: https://huggingface.co/zhihan1996/DNABERT-2-117M

License: Apache-2.0; see `DNABERT2_APACHE_LICENSE.txt`. The pretrained tensors
are not in this repository. The download helper separately verifies the
official checkpoint and every tensor against the study payload.

## OpenSpliceAI

Upstream: https://github.com/Kuanhao-Chao/OpenSpliceAI

Pinned commit: `13ff432c211d0871eec27a1884d59df0a99df5eb`.

License: GPL-3.0 as declared by the pinned upstream LICENSE. This dependency is
not bundled or relicensed. The optional adapter dynamically loads a checked
upstream module. Review the applicable upstream obligations before distributing
a combined work; a license for BioQPred does not override third-party terms.

## Other Dependencies

PyTorch, Transformers, NumPy, scikit-learn, SciPy, einops, safetensors,
huggingface-hub and tqdm are separately installed packages under their own
licenses. No Python wheels or complete dependency environments are included.
