"""Synthetic pixel-level tests for projection, auditing, and file handling."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aigc_clear.core import (audit_arrays, project_arrays, read_alpha, read_mask,
                             read_rgb, save_rgb, write_json)


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.base = np.arange(6 * 7 * 3, dtype=np.uint8).reshape(6, 7, 3)
        self.proposal = 255 - self.base
        self.mask = np.zeros((6, 7), dtype=bool)
        self.mask[2:5, 2:6] = True

    def test_projection_preserves_every_protected_channel(self):
        result = project_arrays(self.base, self.proposal, self.mask)
        np.testing.assert_array_equal(result[~self.mask], self.base[~self.mask])
        np.testing.assert_array_equal(result[self.mask], self.proposal[self.mask])
        np.testing.assert_array_equal(self.base, np.arange(126, dtype=np.uint8).reshape(6, 7, 3))
        self.assertEqual(result.dtype, np.uint8)
        report = audit_arrays(self.base, result, self.mask)
        self.assertEqual(report["geometry"], [7, 6])
        self.assertEqual(report["protected_pixels"], 30)
        self.assertAlmostEqual(report["protected_fraction"], 30 / 42)
        self.assertEqual(report["protected_max_rgb_error"], 0)
        self.assertEqual(report["protected_changed_pixels"], 0)

    def test_alpha_blends_only_allowed_pixels(self):
        base = np.full((2, 3, 3), 20, dtype=np.uint8)
        proposal = np.full_like(base, 100)
        mask = np.array([[False, True, True], [False, True, True]])
        alpha = np.array([[0, 0, 0.25], [0, 0.5, 1]], dtype=np.float32)
        result = project_arrays(base, proposal, mask, alpha)
        expected = np.array([[20, 20, 40], [20, 60, 100]], dtype=np.uint8)
        np.testing.assert_array_equal(result, np.repeat(expected[..., None], 3, axis=2))

    def test_nonzero_alpha_outside_mask_is_rejected(self):
        alpha = self.mask.astype(np.float32)
        alpha[0, 0] = 0.01
        with self.assertRaises(ValueError):
            project_arrays(self.base, self.proposal, self.mask, alpha)

    def test_invalid_alpha_values_are_rejected(self):
        for value in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(value=value):
                alpha = self.mask.astype(np.float32)
                alpha[3, 3] = value
                with self.assertRaises(ValueError):
                    project_arrays(self.base, self.proposal, self.mask, alpha)

    def test_geometry_mismatches_are_rejected(self):
        cases = [
            (self.base, self.proposal[:-1], self.mask, None),
            (self.base, self.proposal, self.mask[:, :-1], None),
            (self.base, self.proposal, self.mask, np.zeros((5, 7), dtype=np.float32)),
            (self.base[..., 0], self.proposal, self.mask, None),
        ]
        for args in cases:
            with self.subTest(shapes=[None if item is None else item.shape for item in args]):
                with self.assertRaises(ValueError):
                    project_arrays(*args)
        with self.assertRaises(ValueError):
            audit_arrays(self.base, self.proposal[:-1], self.mask)
        with self.assertRaises(ValueError):
            audit_arrays(self.base, self.proposal, self.mask[:, :-1])

    def test_empty_and_full_masks_have_defined_audits(self):
        for active in (False, True):
            with self.subTest(active=active):
                mask = np.full(self.mask.shape, active, dtype=bool)
                result = project_arrays(self.base, self.proposal, mask)
                np.testing.assert_array_equal(result, self.proposal if active else self.base)
                report = audit_arrays(self.base, result, mask)
                self.assertEqual(report["protected_pixels"], 0 if active else 42)
                self.assertEqual(report["protected_fraction"], 0.0 if active else 1.0)
                self.assertEqual(report["protected_max_rgb_error"], 0)
                self.assertEqual(report["protected_changed_pixels"], 0)

    def test_audit_counts_pixels_instead_of_changed_channels(self):
        result = self.base.copy()
        result[0, 0] = [255, 255, 255]
        result[0, 1, 1] += 1
        result[3, 3] = [255, 255, 255]  # Allowed changes do not enter the audit.
        report = audit_arrays(self.base, result, self.mask)
        self.assertEqual(report["protected_changed_pixels"], 2)
        self.assertEqual(report["protected_max_rgb_error"], 255)


class FileTests(unittest.TestCase):
    def test_png_roundtrip_and_overwrite_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "image.png"
            original = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
            save_rgb(original, path)
            np.testing.assert_array_equal(read_rgb(path), original)
            with self.assertRaises(FileExistsError):
                save_rgb(np.zeros_like(original), path)
            np.testing.assert_array_equal(read_rgb(path), original)
            save_rgb(np.zeros_like(original), path, overwrite=True)
            np.testing.assert_array_equal(read_rgb(path), np.zeros_like(original))

    def test_mask_and_alpha_readers_check_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            pixels = np.array([[0, 255, 0], [255, 0, 255]], dtype=np.uint8)
            Image.fromarray(pixels, "L").save(path)
            mask = read_mask(path, size=(3, 2))
            self.assertEqual(mask.dtype, np.bool_)
            np.testing.assert_array_equal(mask, pixels != 0)
            alpha = read_alpha(path, size=(3, 2))
            self.assertEqual(alpha.dtype, np.float32)
            np.testing.assert_array_equal(alpha, pixels.astype(np.float32) / 255)
            for reader in (read_mask, read_alpha):
                with self.subTest(reader=reader.__name__):
                    with self.assertRaises(ValueError):
                        reader(path, size=(2, 3))

    def test_json_writes_new_directories_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "new" / "report.json"
            write_json({"protected_changed_pixels": 0}, path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")),
                             {"protected_changed_pixels": 0})
            with self.assertRaises(FileExistsError):
                write_json({"protected_changed_pixels": 2}, path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")),
                             {"protected_changed_pixels": 0})


if __name__ == "__main__":
    unittest.main()
