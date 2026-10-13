"""Regression tests for rejected proposals and workflow preflight checks.

All editor calls are mocked.  These tests need no model, API, or credentials.
"""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image, PngImagePlugin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aigc_clear import backends, cli
from aigc_clear.core import read_rgb


class WorkflowEdgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = self.directory / "source.png"
        Image.new("RGB", (32, 48), (20, 40, 60)).save(self.source)
        self.config = self.directory / "scope.json"
        self.config.write_text(json.dumps({
            "coordinate_size": [32, 48],
            "allow": [{"points": [[2, 2], [29, 2], [29, 45], [2, 45]]}],
            "protect": [], "feather_px": 2,
        }), encoding="utf-8")

    def invoke(self, *arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr), mock.patch.dict(os.environ, {}, clear=True):
            status = cli.main([str(argument) for argument in arguments])
        return status, stdout.getvalue(), stderr.getvalue()

    def png_bytes(self, size, transparent=False):
        image = Image.new("RGBA" if transparent else "RGB", size, (170, 90, 50, 255) if transparent else (170, 90, 50))
        if transparent:
            image.putpixel((0, 0), (170, 90, 50, 0))
        info = PngImagePlugin.PngInfo()
        info.add_text("Description", "Mock editor raw bytes must survive rejection unchanged")
        stream = io.BytesIO()
        image.save(stream, format="PNG", pnginfo=info, compress_level=0)
        return stream.getvalue()

    def fake_editor(self, payload, backend):
        def generate(source, prompt, output, **kwargs):
            self.assertTrue(Path(source).is_file())
            self.assertEqual(prompt, "clean material")
            Path(output).write_bytes(payload)
            with Image.open(io.BytesIO(payload)) as image:
                size = list(image.size)
            return {
                "backend": backend, "model": "mock-model", "revision": None,
                "seed": None, "steps": None, "generation_seconds": 0,
                "output_size": size,
            }
        return generate

    def assert_rejected_raw_response(self, payload, backend):
        output = self.directory / backend / "proposal.png"
        report = output.with_suffix(".json")
        raw = output.with_name(output.stem + ".raw.png")
        chosen = "generate_" + backend
        other = "generate_klein" if backend == "openai" else "generate_openai"
        with mock.patch.object(backends, chosen, side_effect=self.fake_editor(payload, backend)) as editor, \
                mock.patch.object(backends, other) as unused:
            status, stdout, stderr = self.invoke(
                "propose", "--backend", backend, "--input", self.source,
                "--prompt", "clean material", "--output", output,
            )
        self.assertNotEqual(status, 0, stdout + stderr)
        editor.assert_called_once()
        unused.assert_not_called()
        self.assertTrue(raw.is_file(), stderr)
        self.assertEqual(raw.read_bytes(), payload)
        self.assertFalse(output.exists())
        self.assertFalse(report.exists())

    def test_propose_keeps_transparent_png_bytes_for_both_backends(self):
        payload = self.png_bytes((32, 48), transparent=True)
        for backend in ("openai", "klein"):
            with self.subTest(backend=backend):
                self.assert_rejected_raw_response(payload, backend)

    def test_propose_keeps_wrong_size_png_bytes_for_both_backends(self):
        payload = self.png_bytes((24, 16))
        for backend in ("openai", "klein"):
            with self.subTest(backend=backend):
                self.assert_rejected_raw_response(payload, backend)

    def test_raw_output_cannot_replace_prompt_file_even_with_overwrite(self):
        output = self.directory / "proposal.png"
        raw = self.directory / "proposal.raw.png"
        prompt_bytes = b"clean material\n"
        raw.write_bytes(prompt_bytes)
        with mock.patch.object(backends, "generate_openai") as openai_editor, \
                mock.patch.object(backends, "generate_klein") as klein_editor:
            status, stdout, stderr = self.invoke(
                "propose", "--backend", "openai", "--input", self.source,
                "--prompt-file", raw, "--output", output, "--overwrite",
            )
        self.assertNotEqual(status, 0, stdout + stderr)
        openai_editor.assert_not_called()
        klein_editor.assert_not_called()
        self.assertEqual(raw.read_bytes(), prompt_bytes)
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix(".json").exists())

    def test_raw_output_cannot_share_report_path_even_with_overwrite(self):
        output = self.directory / "proposal.png"
        raw = self.directory / "proposal.raw.png"
        report_bytes = b'{"existing_report": true}\n'
        raw.write_bytes(report_bytes)
        with mock.patch.object(backends, "generate_openai") as openai_editor, \
                mock.patch.object(backends, "generate_klein") as klein_editor:
            status, stdout, stderr = self.invoke(
                "propose", "--backend", "openai", "--input", self.source,
                "--prompt", "clean material", "--output", output,
                "--report", raw, "--overwrite",
            )
        self.assertNotEqual(status, 0, stdout + stderr)
        openai_editor.assert_not_called()
        klein_editor.assert_not_called()
        self.assertEqual(raw.read_bytes(), report_bytes)
        self.assertFalse(output.exists())

    def assert_run_rejects_without_partial_files(self, proposal):
        output_dir = self.directory / "run"
        with mock.patch.object(backends, "generate_openai") as openai_editor, \
                mock.patch.object(backends, "generate_klein") as klein_editor:
            status, stdout, stderr = self.invoke(
                "run", "--input", self.source, "--config", self.config,
                "--proposal", proposal, "--output-dir", output_dir,
            )
        self.assertNotEqual(status, 0, stdout + stderr)
        openai_editor.assert_not_called()
        klein_editor.assert_not_called()
        files = [path for path in output_dir.rglob("*") if path.is_file()] if output_dir.exists() else []
        self.assertEqual(files, [], "Rejected input left partial workflow files")

    def test_run_missing_existing_proposal_writes_no_partial_files(self):
        self.assert_run_rejects_without_partial_files(self.directory / "missing.png")

    def test_run_wrong_size_existing_proposal_writes_no_partial_files(self):
        proposal = self.directory / "wrong_size.png"
        Image.new("RGB", (24, 16), (90, 100, 110)).save(proposal)
        self.assert_run_rejects_without_partial_files(proposal)

    def test_read_rgb_rejects_palette_png_with_transparent_pixels(self):
        path = self.directory / "palette_transparent.png"
        image = Image.new("P", (2, 2))
        image.putpalette([10, 20, 30, 40, 50, 60] + [0] * (768 - 6))
        image.putdata([0, 1, 1, 1])
        image.save(path, transparency=0)
        with Image.open(path) as saved:
            self.assertEqual(saved.mode, "P")
            self.assertNotIn("A", saved.getbands())
        with self.assertRaises(ValueError):
            read_rgb(path)

    def test_read_rgb_accepts_palette_transparency_entry_when_no_pixel_uses_it(self):
        path = self.directory / "palette_opaque.png"
        image = Image.new("P", (2, 2), color=1)
        image.putpalette([10, 20, 30, 40, 50, 60] + [0] * (768 - 6))
        image.save(path, transparency=0)
        expected = np.full((2, 2, 3), [40, 50, 60], dtype=np.uint8)
        np.testing.assert_array_equal(read_rgb(path), expected)


if __name__ == "__main__":
    unittest.main()
