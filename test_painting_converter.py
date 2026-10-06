import unittest

import numpy as np

from painting_converter import blend_rgb, find_three_layer_matches, find_two_layer_matches, flatten_features, oklab_features


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

    def test_three_layer_matcher_logs_elapsed_time_for_each_step(self):
        backs = np.zeros((1, 16, 16, 4), dtype=np.uint8)
        backs[..., 3] = 255
        middles = np.zeros_like(backs)
        middles[..., 3] = 255
        fronts = np.zeros_like(backs)
        fronts[..., 3] = 255
        target = np.zeros((1, 16, 16, 3), dtype=np.uint8)
        pairs = np.array([[[0, 0]]])

        with self.assertLogs("painting_converter", level="INFO") as logs:
            matches = find_three_layer_matches(target, pairs, backs, middles, fronts, perceptual=False)

        np.testing.assert_array_equal(matches, [[0, 0, 0]])
        steps = (
            "target feature conversion",
            "front-layer preparation",
            "tile feature preparation",
            "shortlist retrieval",
            "middle-layer compositing",
            "middle-layer feature conversion",
            "transparent and opaque front scoring",
            "front-layer compositing",
            "candidate feature conversion",
            "candidate scoring",
            "winner selection and result storage",
        )
        for step in steps:
            self.assertTrue(any(f"{step} took" in message for message in logs.output), step)
        self.assertTrue(any("total took" in message for message in logs.output))

    def test_three_layer_matcher_agrees_with_brute_force(self):
        rng = np.random.default_rng(0)
        backs = rng.integers(0, 256, (3, 16, 16, 4), dtype=np.uint8)
        middles = rng.integers(0, 256, (4, 16, 16, 4), dtype=np.uint8)
        fronts = rng.integers(0, 256, (5, 16, 16, 4), dtype=np.uint8)
        # Mix fully transparent, fully opaque and partially transparent front pixels
        fronts[..., 3] = rng.choice(np.array([0, 255, 64, 200], dtype=np.uint8), (5, 16, 16))
        targets = rng.integers(0, 256, (6, 16, 16, 3), dtype=np.uint8)
        pairs = np.stack(np.meshgrid(np.arange(3), np.arange(4), indexing="ij"), axis=-1).reshape(1, -1, 2).repeat(len(targets), axis=0)

        for perceptual in (False, True):
            features_of = oklab_features if perceptual else flatten_features
            with self.assertLogs("painting_converter", level="INFO"):
                matches = find_three_layer_matches(targets, pairs, backs, middles, fronts, perceptual)
            for tile, match in enumerate(matches):
                best = min(
                    float(np.square(features_of(blend_rgb(blend_rgb(backs[b:b+1, ..., :3], middles[m:m+1]), fronts[f:f+1])) - features_of(targets[tile:tile+1])).sum())
                    for b, m in pairs[tile] for f in range(len(fronts))
                )
                found = np.square(features_of(blend_rgb(blend_rgb(backs[match[0:1], ..., :3], middles[match[1:2]]), fronts[match[2:3]])) - features_of(targets[tile:tile+1])).sum()
                self.assertAlmostEqual(float(found), best, places=4)


if __name__ == "__main__":
    unittest.main()
