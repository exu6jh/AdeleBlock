import unittest

import numpy as np

from painting_converter import blend_rgb, find_two_layer_matches, oklab_features


class ConverterTests(unittest.TestCase):
    def test_blend_uses_source_alpha(self):
        background = np.full((1, 1, 3), (10, 20, 30), dtype=np.uint8)
        overlay = np.array([[[110, 120, 130, 128]]], dtype=np.uint8)
        np.testing.assert_array_equal(blend_rgb(background, overlay), [[[60, 70, 80]]])

    def test_oklab_preserves_shape_and_black(self):
        pixels = np.zeros((2, 16, 16, 3), dtype=np.uint8)
        features = oklab_features(pixels)
        self.assertEqual(features.shape, (2, 16 * 16 * 3))
        self.assertFalse(features.any())

    def test_matcher_finds_exact_composite(self):
        backs = np.zeros((2, 16, 16, 4), dtype=np.uint8)
        backs[0, ..., :3] = 20
        backs[1, ..., :3] = 200
        backs[..., 3] = 255
        overlays = np.zeros((2, 16, 16, 4), dtype=np.uint8)
        overlays[0, ..., :3] = 50
        overlays[0, ..., 3] = 255
        overlays[1, ..., :3] = 100
        overlays[1, ..., 3] = 128
        target = blend_rgb(backs[1:2, ..., :3], overlays[1:2])

        pairs, scores = find_two_layer_matches(target, backs, overlays, 1)

        np.testing.assert_array_equal(pairs[:, 0], [[1, 1]])
        self.assertLess(float(scores[0, 0]), 1e-3)


if __name__ == "__main__":
    unittest.main()
