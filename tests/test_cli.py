"""CLI integration tests run the public entry point in real subprocesses."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aigc_clear.core import project_arrays
from aigc_clear.masks import build_masks


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = np.full((16, 16, 3), 20, dtype=np.uint8)
        self.proposal = np.full_like(self.source, 100)
        self.mask = np.zeros((16, 16), dtype=bool)
        self.mask[4:12, 4:12] = True
        self.input_path = self.directory / "source.png"
        self.proposal_path = self.directory / "proposal.png"
        self.mask_path = self.directory / "mask.png"
        self.alpha_path = self.directory / "alpha.png"
        Image.fromarray(self.source, "RGB").save(self.input_path)
        Image.fromarray(self.proposal, "RGB").save(self.proposal_path)
        Image.fromarray(self.mask.astype(np.uint8) * 255, "L").save(self.mask_path)
        Image.fromarray(self.mask.astype(np.uint8) * 255, "L").save(self.alpha_path)
        self.config_path = self.directory / "scope.json"
        self.config_path.write_text(json.dumps({
            "coordinate_size": [16, 16],
            "allow": [{"name": "material", "points": [[4, 4], [11, 4], [11, 11], [4, 11]]}],
            "protect": [], "feather_px": 2,
        }), encoding="utf-8")
        self.environment = os.environ.copy()
        self.environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + self.environment.get("PYTHONPATH", "")

    def run_cli(self, *arguments):
        return subprocess.run([sys.executable, "-m", "aigc_clear"] + [str(arg) for arg in arguments],
                              cwd=str(self.directory), env=self.environment,
                              text=True, capture_output=True, timeout=30)

    def assert_success(self, process):
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)

    def test_mask_creates_output_directory_and_refuses_second_write(self):
        output_dir = self.directory / "new" / "masks"
        args = ("mask", "--input", self.input_path, "--config", self.config_path,
                "--output-dir", output_dir)
        self.assert_success(self.run_cli(*args))
        mask_path, alpha_path = output_dir / "mask.png", output_dir / "alpha.png"
        self.assertTrue(mask_path.is_file())
        self.assertTrue(alpha_path.is_file())
        mask = np.asarray(Image.open(mask_path)) != 0
        alpha = np.asarray(Image.open(alpha_path))
        self.assertTrue(mask[8, 8])
        self.assertFalse(mask[0, 0])
        self.assertTrue(np.all(alpha[~mask] == 0))
        before = {path.name: path.read_bytes() for path in output_dir.iterdir() if path.is_file()}
        repeated = self.run_cli(*args)
        self.assertNotEqual(repeated.returncode, 0)
        after = {path.name: path.read_bytes() for path in output_dir.iterdir() if path.is_file()}
        self.assertEqual(before, after)

    def test_project_and_audit_preserve_protected_pixels(self):
        output = self.directory / "new" / "result.png"
        report_path = self.directory / "reports" / "project.json"
        args = ("project", "--input", self.input_path, "--proposal", self.proposal_path,
                "--mask", self.mask_path, "--alpha", self.alpha_path,
                "--output", output, "--report", report_path)
        self.assert_success(self.run_cli(*args))
        result = np.asarray(Image.open(output).convert("RGB"))
        np.testing.assert_array_equal(result[~self.mask], self.source[~self.mask])
        np.testing.assert_array_equal(result[self.mask], self.proposal[self.mask])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["protected_changed_pixels"], 0)
        audit = self.run_cli("audit", "--reference", self.input_path,
                             "--image", output, "--mask", self.mask_path)
        self.assert_success(audit)
        self.assertEqual(json.loads(audit.stdout)["protected_max_rgb_error"], 0)
        before = output.read_bytes(), report_path.read_bytes()
        self.assertNotEqual(self.run_cli(*args).returncode, 0)
        self.assertEqual(before, (output.read_bytes(), report_path.read_bytes()))

    def test_audit_returns_failure_and_json_for_protected_change(self):
        changed = self.source.copy()
        changed[0, 0] = [255, 21, 20]
        changed_path = self.directory / "changed.png"
        Image.fromarray(changed, "RGB").save(changed_path)
        audit = self.run_cli("audit", "--reference", self.input_path,
                             "--image", changed_path, "--mask", self.mask_path)
        self.assertEqual(audit.returncode, 1, audit.stdout + audit.stderr)
        report = json.loads(audit.stdout)
        self.assertEqual(report["protected_changed_pixels"], 1)
        self.assertEqual(report["protected_max_rgb_error"], 235)

    def test_project_rejects_nonzero_alpha_outside_mask_without_output(self):
        bad_alpha = self.mask.astype(np.uint8) * 255
        bad_alpha[0, 0] = 1
        Image.fromarray(bad_alpha, "L").save(self.alpha_path)
        output = self.directory / "invalid.png"
        report = self.directory / "invalid.json"
        process = self.run_cli("project", "--input", self.input_path, "--proposal", self.proposal_path,
                               "--mask", self.mask_path, "--alpha", self.alpha_path,
                               "--output", output, "--report", report)
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(output.exists())
        self.assertFalse(report.exists())

    def test_compare_creates_new_directory_and_refuses_overwrite(self):
        output = self.directory / "comparison" / "compare.png"
        args = ("compare", "--images", self.input_path, self.proposal_path, "--output", output)
        self.assert_success(self.run_cli(*args))
        with Image.open(output) as comparison:
            self.assertGreater(comparison.width, 16)
            self.assertGreaterEqual(comparison.height, 16)
        before = output.read_bytes()
        self.assertNotEqual(self.run_cli(*args).returncode, 0)
        self.assertEqual(before, output.read_bytes())

    def test_demo_is_synthetic_and_projection_recovers_background(self):
        output_dir = self.directory / "demo"
        command = [sys.executable, str(ROOT / "examples" / "make_demo.py"),
                   "--output-dir", str(output_dir)]
        process = subprocess.run(command, cwd=str(self.directory), env=self.environment,
                                 text=True, capture_output=True, timeout=30)
        self.assert_success(process)
        source = np.asarray(Image.open(output_dir / "source.png").convert("RGB"))
        proposal = np.asarray(Image.open(output_dir / "proposal.png").convert("RGB"))
        self.assertEqual(source.shape, (512, 512, 3))
        scope = json.loads((output_dir / "scope.json").read_text(encoding="utf-8"))
        mask, alpha, _ = build_masks(source, scope)
        self.assertTrue(np.any(source[mask] != proposal[mask]))
        self.assertTrue(np.any(source[~mask] != proposal[~mask]))
        np.testing.assert_array_equal(source[84:248, 182:332], proposal[84:248, 182:332])
        result = project_arrays(source, proposal, mask, alpha)
        np.testing.assert_array_equal(result[~mask], source[~mask])
        before = {path.name: path.read_bytes() for path in output_dir.iterdir()}
        repeated = subprocess.run(command, cwd=str(self.directory), env=self.environment,
                                  text=True, capture_output=True, timeout=30)
        self.assertNotEqual(repeated.returncode, 0)
        self.assertEqual(before, {path.name: path.read_bytes() for path in output_dir.iterdir()})


if __name__ == "__main__":
    unittest.main()
