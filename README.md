# AdeleBlock

AdeleBlock converts an image into a Minecraft block mosaic. Each source tile maps to one 16 × 16 block texture. The program can combine two or three block layers.

## Requirements

- Python 3.10 or later
- NumPy
- Pillow
- tqdm

Install the packages:

```console
python -m pip install -r requirements.txt
```

Run the converter:

```console
python painting_converter.py
```

Select an image. Then enter the required block width, block height, and layer count. The program writes the result, each separate layer, and a block palette to `outputs/<source filename>/`.

For three-layer images, `candidates` in `userpref.ini` controls the number of two-layer matches that the program tests with every third-layer texture. A larger value can improve the result, but it increases the third-layer run time and memory use.

## Algorithm

The program loads each texture one time. It composites candidate texture batches with the same integer alpha operation that Pillow uses. It converts the candidates and source tiles from sRGB to OKLab. OKLab distance gives a closer model of perceived color difference than distance in gamma-encoded RGB.

For each candidate batch, the program calculates squared Euclidean distance with this identity:

```text
distance²(x, y) = ||x||² + ||y||² - 2(x · y)
```

NumPy calculates the dot products for all source tiles in one matrix operation. This operation replaces the previous per-tile PNG, FFT, and cache-file loop. The program does not need the former multi-gigabyte FFT cache.

Two-layer mode searches all backing and overlay combinations. Three-layer mode searches all two-layer combinations, keeps the configured best candidates for each source tile, and tests each third-layer texture against that shortlist.

## Performance reference

The measured reference in [`benchmark_results`](benchmark_results) uses the 960 × 1452 public-domain Mona Lisa source and a 20 × 20-block, two-layer output. The optimized implementation completed the full cold run in 73.34 seconds on the development machine.

The original implementation completed its 31-second setup and 5 of 42 FFT chunks for the first tile in 41 seconds. That measured rate projects to approximately 38 hours for 400 tiles. The original run was stopped because a full result was not practical. This value is a projection, not a completed-run time.

The `before.png` comparison uses the original RGB channel-RMS selection metric with the same candidate catalog and exact alpha composition. The `after.png` comparison uses OKLab. Mean OKLab error decreased from 0.001816 to 0.001712, which is a 5.73% reduction.

## Name

The name combines *Portrait of Adele Bloch-Bauer I* by Gustav Klimt with Minecraft blocks.
