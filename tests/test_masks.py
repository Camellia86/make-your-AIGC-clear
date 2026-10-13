"""Tests for edit-scope composition and inward-only feathering."""

import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aigc_clear.masks import build_masks


class MaskTests(unittest.TestCase):
    def setUp(self):
        self.source = np.full((20, 20, 3), [180, 180, 205], dtype=np.uint8)
        self.config = {
            "coordinate_size": [20, 20],
            "allow": [{"name": "material", "points": [[2, 2], [17, 2], [17, 17], [2, 17]]}],
            "protect": [],
            "feather_px": 0,
        }

    def test_allow_scope_and_protect_priority(self):
        self.config["protect"] = [{"points": [[7, 7], [12, 7], [12, 12], [7, 12]]}]
        mask, alpha, metadata = build_masks(self.source, self.config)
        self.assertTrue(mask[4, 4])
        self.assertFalse(mask[0, 0])
        self.assertFalse(mask[9, 9])
        self.assertEqual(alpha[9, 9], 0)
        self.assertEqual(mask.dtype, np.bool_)
        self.assertEqual(alpha.dtype, np.float32)
        self.assertIsInstance(metadata, dict)
        np.testing.assert_array_equal(alpha, mask.astype(np.float32))

    def test_omitted_active_rects_keep_all_allow_regions(self):
        self.config["allow"] = [
            {"name": "left", "points": [[1, 2], [7, 2], [7, 15], [1, 15]]},
            {"name": "right", "points": [[12, 2], [18, 2], [18, 15], [12, 15]]},
        ]
        mask, _, _ = build_masks(self.source, self.config)
        self.assertTrue(mask[8, 4])
        self.assertTrue(mask[8, 15])
        self.config["active_rects"] = [[0, 0, 9, 20]]
        selected, alpha, _ = build_masks(self.source, self.config)
        self.assertTrue(selected[8, 4])
        self.assertTrue(selected[8, 15])
        self.assertEqual(alpha[8, 4], 1)
        self.assertEqual(alpha[8, 15], 0)

    def test_color_guard_uses_source_color_thresholds(self):
        self.config["allow"][0]["color_guard"] = {
            "min_rgb": [145, 145, 170], "b_minus_r_min": -8,
        }
        self.source[5, 5] = [144, 180, 200]
        self.source[5, 6] = [180, 144, 200]
        self.source[5, 7] = [180, 180, 169]
        self.source[5, 8] = [180, 180, 171]
        self.source[5, 9] = [180, 180, 172]
        mask, _, _ = build_masks(self.source, self.config)
        self.assertFalse(mask[5, 5])
        self.assertFalse(mask[5, 6])
        self.assertFalse(mask[5, 7])
        self.assertFalse(mask[5, 8])
        self.assertTrue(mask[5, 9])

    def test_feather_changes_alpha_without_expanding_mask(self):
        base_mask = np.zeros((20, 20), dtype=bool)
        base_mask[4:16, 4:16] = True
        config = {"coordinate_size": [20, 20], "allow": [], "protect": [], "feather_px": 3}
        mask, alpha, _ = build_masks(self.source, config, base_mask=base_mask)
        np.testing.assert_array_equal(mask, base_mask)
        np.testing.assert_array_equal(alpha[~base_mask], 0)
        self.assertTrue(np.all(np.isfinite(alpha)))
        self.assertTrue(np.all((alpha >= 0) & (alpha <= 1)))
        self.assertGreater(alpha[10, 10], alpha[4, 10])
        self.assertEqual(alpha[10, 10], 1)

    def test_protect_is_applied_to_imported_base_mask(self):
        base_mask = np.ones((20, 20), dtype=bool)
        config = {"coordinate_size": [20, 20], "allow": [], "feather_px": 2,
                  "protect": [{"points": [[7, 7], [12, 7], [12, 12], [7, 12]]}]}
        mask, alpha, _ = build_masks(self.source, config, base_mask=base_mask)
        self.assertTrue(mask[1, 1])
        self.assertFalse(mask[9, 9])
        self.assertEqual(alpha[9, 9], 0)
        self.assertLess(alpha[6, 9], alpha[3, 9])

    def test_coordinate_size_mismatch_is_rejected(self):
        config = {"coordinate_size": [10, 10], "allow": [
            {"points": [[5, 5], [9, 5], [9, 9], [5, 9]]}], "protect": [], "feather_px": 0}
        with self.assertRaises(ValueError):
            build_masks(self.source, config)

    def test_full_and_empty_masks(self):
        config = {"coordinate_size": [20, 20], "allow": [], "protect": [], "feather_px": 4}
        for active in (False, True):
            with self.subTest(active=active):
                base_mask = np.full((20, 20), active, dtype=bool)
                mask, alpha, _ = build_masks(self.source, config, base_mask=base_mask)
                np.testing.assert_array_equal(mask, base_mask)
                np.testing.assert_array_equal(alpha[~mask], 0)
                self.assertTrue(np.all(np.isfinite(alpha)))
                self.assertTrue(np.all((alpha >= 0) & (alpha <= 1)))
                if active:
                    self.assertEqual(alpha[10, 10], 1)
                else:
                    np.testing.assert_array_equal(alpha, 0)

    def test_missing_allow_and_base_mask_is_rejected(self):
        config = {"coordinate_size": [20, 20], "allow": [], "protect": [], "feather_px": 0}
        with self.assertRaises(ValueError):
            build_masks(self.source, config)

    def test_invalid_geometry_and_nonfinite_config_are_rejected(self):
        with self.assertRaises(ValueError):
            build_masks(self.source, self.config, base_mask=np.zeros((19, 20), dtype=bool))
        with self.assertRaises(ValueError):
            build_masks(self.source[..., 0], self.config)
        mutations = [
            ("coordinate_size", [0, 20]),
            ("feather_px", -1),
            ("feather_px", float("nan")),
            ("allow", [{"points": [[2, 2], [float("inf"), 3], [5, 5]]}]),
            ("allow", [{"points": [[2, 2], [5, 5]]}]),
            ("active_rects", [[0, 0, float("nan"), 20]]),
        ]
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                config = copy.deepcopy(self.config)
                config[key] = value
                with self.assertRaises(ValueError):
                    build_masks(self.source, config)


if __name__ == "__main__":
    unittest.main()
