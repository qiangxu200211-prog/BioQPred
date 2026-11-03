"""Candidate-label adapter around a pinned, unchanged OpenSpliceAI module.

This is not dense genome-wide supervision or the published pretrained model.
"""

import hashlib
import importlib.util
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "research_dependencies/OpenSpliceAI"
COMMIT = "13ff432c211d0871eec27a1884d59df0a99df5eb"
FILES = {
    "openspliceai/train_base/openspliceai.py": "94fcb9aa732419865e18cf6b56ba0f73f191a854fc80d6c109f00ecba9fe51ef",
    "openspliceai/train/train.py": "cd3beb92ba6d8216dc0a7c8a8bfeaf497930eebc64d9b35c683e0d37933f0eed",
    "LICENSE": "230184f60bae2feaf244f10a8bac053c8ff33a183bcc365b4d8b876d2b7f4809",
}
TASKS = {"splice_sites_acceptor": ("AG", 1), "splice_sites_donor": ("GT", 2)}
RC = str.maketrans("ACGTN", "TGCAN")
WIDTHS = [11] * 8 + [21] * 4
DILATIONS = [1] * 4 + [4] * 4 + [10] * 4
CONTEXT = 2000


def official_class():
    for name, expected in FILES.items():
        if hashlib.sha256((UPSTREAM / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Pinned upstream source/license changed: " + name)
    spec = importlib.util.spec_from_file_location(
        "pinned_openspliceai", UPSTREAM / "openspliceai/train_base/openspliceai.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SpliceAI


def infer_candidate(train, motif):
    oriented = {}
    for label in (0, 1):
        seqs = [r["sequence"].upper() for r in train if r["label"] == label]
        if not seqs or any(len(s) != 600 or set(s) - set("ACGTN") for s in seqs):
            raise ValueError("Expected both binary classes and valid 600-base training sequences")
        oriented[label] = [(s, s.translate(RC)[::-1]) for s in seqs]
    peaks = []
    for strand in (0, 1):
        counts = Counter({p: sum(s[strand][p:p + 2] == motif for s in oriented[1])
                          for p in range(295, 305)})
        peaks.append(min(p for p, n in counts.items() if n == max(counts.values())))
    fractions = {label: sum(any(pair[j][peaks[j]:peaks[j] + 2] == motif for j in (0, 1))
                           for pair in pairs) / len(pairs) for label, pairs in oriented.items()}
    delta = {"AG": 2, "GT": -1}[motif]
    anchors = {"forward": peaks[0] + delta, "reverse": peaks[1] + delta}
    if (fractions[1] < 0.8 or fractions[1] <= fractions[0]
            or sum(anchors.values()) != 599):
        raise ValueError("Training-only motifs do not establish opposite-strand candidate anchors")
    return anchors, {"motif": motif, "core_starts": peaks,
                     "positive_fraction": fractions[1], "negative_fraction": fractions[0]}


def encode(sequences):
    lookup = np.zeros(256, dtype=np.int64)
    for i, base in enumerate("ACGT", 1):
        lookup[ord(base)] = i
    values = []
    for sequence in sequences:
        sequence = sequence.upper()
        if len(sequence) != 600 or set(sequence) - set("ACGTN"):
            raise ValueError("Only the unchanged 600 observed bases are accepted")
        values.append(lookup[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)])
    return torch.tensor(np.asarray(values), dtype=torch.long)


def candidate_logits(logits, anchors, channel, batch):
    if logits.shape != (2 * batch, 3, 600):
        raise ValueError("Unexpected official output shape")
    # A task-negative means either of the other two channels, not a dense label.
    candidates = torch.stack((logits[:batch, :, anchors["forward"]],
                              logits[batch:, :, anchors["reverse"]]), dim=1).float()
    other = [i for i in range(3) if i != channel]
    log_odds = candidates[:, :, channel] - torch.logsumexp(candidates[:, :, other], dim=-1)
    log_odds = log_odds.max(dim=1).values
    return torch.stack((torch.zeros_like(log_odds), log_odds), dim=1)


class CandidateOpenSpliceAI(nn.Module):
    def __init__(self, task, anchors):
        super().__init__()
        self.anchors, self.channel = dict(anchors), TASKS[task][1]
        self.network = official_class()(32, np.asarray(WIDTHS), np.asarray(DILATIONS),
                                        apply_softmax=False)
        if self.network.CL != CONTEXT:
            raise ValueError("Official receptive-context configuration differs")

    def forward(self, tokens):
        if tokens.ndim != 2 or tokens.shape[1] != 600:
            raise ValueError("Expected B x 600 observed-base tokens")
        x = F.one_hot(tokens, num_classes=5)[:, :, 1:].permute(0, 2, 1).float()
        reverse = x[:, [3, 2, 1, 0], :].flip(-1)
        both = F.pad(torch.cat((x, reverse), dim=0), (CONTEXT // 2, CONTEXT // 2))
        return candidate_logits(self.network(both), self.anchors, self.channel, len(tokens))
