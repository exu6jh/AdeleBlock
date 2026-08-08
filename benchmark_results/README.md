# Benchmark result

- Source: [Mona Lisa.jpg, Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Mona_Lisa.jpg)
- Source size: 960 × 1452 pixels
- Output: 20 × 20 blocks, 320 × 320 pixels, two layers
- Texture combinations per tile: 328,536
- Optimized full cold run: 73.342493 seconds
- Original cold-run projection: approximately 38 hours
- Original mean OKLab error: 0.0018159392
- Optimized mean OKLab error: 0.0017119189
- Perceptual error reduction: 5.73%
- Tiles with a changed match: 237 of 400

`before.png` uses the original RGB channel-RMS match metric. `after.png` uses the optimized OKLab match metric. Both images use the same source, dimensions, texture catalog, and alpha-composition operation.

The original program completed its 31-second combination-tile setup and 5 of 42 FFT chunks for the first tile in 41 seconds. The projected time uses that measured prefix. The original full run was stopped because it would take approximately 38 hours. Do not treat the projection as a completed-run measurement.
