"""Small CPU checks, not paper experiments or benchmark results."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "revision_20261022"))
sys.path.insert(0, str(ROOT / "scripts"))


class ReleaseTests(unittest.TestCase):
    def test_core_variants_forward_backward(self):
        import torch
        import run_controlled_experiments as control
        torch.set_num_threads(2)
        for variant in control.VARIANTS:
            with self.subTest(variant=variant):
                model = control.build_model(variant, 64, 3, 4, [3, 5])
                output = model({}, torch.ones(2, 64, dtype=torch.long), torch.zeros(2, 12), bert_pooled=torch.zeros(2, 1536))
                self.assertEqual(tuple(output.shape), (2, 3))
                self.assertTrue(torch.isfinite(output).all())
                output.sum().backward()
                self.assertTrue(any(p.grad is not None for p in model.parameters() if p.requires_grad))

    def test_source_manifest(self):
        manifest = json.loads((ROOT / "configs/source_provenance.json").read_text())
        for entry in manifest["files"]:
            self.assertEqual(hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest(), entry["release_sha256"])
            if entry["modification"] == "none":
                self.assertEqual(entry["release_sha256"], entry["executed_sha256"])

    def test_data_manifest(self):
        source = json.loads((ROOT / "configs/data_sources.json").read_text())
        expected = json.loads((ROOT / "configs/expected_splits.json").read_text())
        self.assertEqual(len(source["files"]), 8)
        self.assertEqual(len(expected["tasks"]), 4)
        for task, splits in expected["tasks"].items():
            self.assertEqual([splits[s]["rows"] for s in ("train", "dev", "test")], [27000, 3000, 3000])

    def test_fasta_validation(self):
        from verify_pinned_nt import read_fna
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "test.fna"
            file.write_text(">chr1:10-14|1\nACGT\n", encoding="ascii")
            self.assertEqual(read_fna(file)[0]["sequence"], "ACGT")
            file.write_text(">chr1:10-15|1\nACGT\n", encoding="ascii")
            with self.assertRaises(ValueError):
                read_fna(file)

    def test_label_and_sequence_encoding(self):
        from src.models.bioqpred_v2 import encode_base_ids, compute_composition_features
        self.assertEqual(encode_base_ids(["ACGTN"], 7).tolist(), [[1, 2, 3, 4, 0, 0, 0]])
        self.assertEqual(compute_composition_features(["ACGTN"]).shape, (1, 12))

    def test_refuse_conflicting_split(self):
        from prepare_data import save_split
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            path.write_text("existing", encoding="ascii")
            with self.assertRaises(ValueError):
                save_split(path, [{"id": "x"}], {"rows": 1, "sha256": "0" * 64})
            self.assertEqual(path.read_text(), "existing")

    def test_predict_local_checkpoint(self):
        import torch
        import run_controlled_experiments as control
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            model = control.build_model("local_comp", 600, 2, 4, [3])
            spec = dict(task="splice_sites_acceptor", variant="local_comp", channels=4, kernels=[3])
            torch.save(dict(model_state_dict=model.state_dict(), spec=spec), directory / "trusted.pt")
            (directory / "input.jsonl").write_text(json.dumps(dict(id="synthetic", sequence="ACGT" * 150)) + "\n")
            subprocess.run([sys.executable, str(ROOT / "scripts/predict.py"), "--checkpoint", str(directory / "trusted.pt"), "--input", str(directory / "input.jsonl"), "--output", str(directory / "output.csv")], check=True, capture_output=True, text=True, timeout=90)
            self.assertEqual(len((directory / "output.csv").read_text().splitlines()), 2)

    def test_runner_imports(self):
        import fair_baseline_models
        import run_fair_baselines
        import run_teacher_ablation
        import run_openspliceai_matched
        import run_neugse_matched
        self.assertTrue(callable(run_neugse_matched.main))


if __name__ == "__main__":
    unittest.main(verbosity=2)
