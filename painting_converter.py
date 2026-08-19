from __future__ import annotations

import configparser
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence
from tqdm import tqdm

TKINTER_FLAG=True
try:
    from tkinter.filedialog import askopenfilename
except ImportError:
    TKINTER_FLAG=False


PIXELS = 16
ROOT = Path(__file__).resolve().parent
TEXTURES = ROOT / "textures"
OUTPUTS = ROOT / "outputs"
SCORE_BATCH_SIZE = 1024


def texture_files(*folders: str) -> list[Path]:
    return sorted(path for folder in folders for path in (TEXTURES / folder).iterdir() if path.is_file())


def load_textures(paths: list[Path]) -> np.ndarray:
    textures = []
    for path in paths:
        with Image.open(path) as image:
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        if rgba.shape != (PIXELS, PIXELS, 4):
            raise ValueError(f"Texture must be {PIXELS} x {PIXELS} pixels: {path}")
        textures.append(rgba)
    return np.stack(textures)


def blend_rgb(background: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Composite RGBA overlays onto RGB backgrounds with integer alpha."""
    alpha = overlay[..., 3:4].astype(np.uint16)
    foreground = overlay[..., :3].astype(np.uint16)
    background = background.astype(np.uint16)
    return ((foreground * alpha + background * (255 - alpha) + 127) // 255).astype(np.uint8)


def oklab_features(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB pixels to perceptually uniform OKLab feature vectors."""
    values = rgb.astype(np.float32) / 255.0
    values = np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)
    red, green, blue = np.moveaxis(values, -1, 0)
    long = np.cbrt(0.41222147 * red + 0.53633254 * green + 0.05144599 * blue)
    medium = np.cbrt(0.21190350 * red + 0.68069955 * green + 0.10739696 * blue)
    short = np.cbrt(0.08830246 * red + 0.28171884 * green + 0.62997870 * blue)
    lab = np.stack(
        (
            0.21045426 * long + 0.79361779 * medium - 0.00407205 * short,
            1.97799850 * long - 2.42859221 * medium + 0.45059371 * short,
            0.02590404 * long + 0.78277177 * medium - 0.80867577 * short,
        ),
        axis=-1,
    )
    return lab.reshape(*rgb.shape[:-3], -1)


def _merge_shortlist(best_scores, best_indices, scores, offset):
    keep = best_scores.shape[1]
    count = min(keep, scores.shape[1])
    local = np.argpartition(scores, count - 1, axis=1)[:, :count]
    merged_scores = np.concatenate((best_scores, np.take_along_axis(scores, local, axis=1)), axis=1)
    merged_indices = np.concatenate((best_indices, local + offset), axis=1)
    selected = np.argpartition(merged_scores, keep - 1, axis=1)[:, :keep]
    return (
        np.take_along_axis(merged_scores, selected, axis=1),
        np.take_along_axis(merged_indices, selected, axis=1),
    )


def find_two_layer_matches(targets_rgb, backs_rgba, overlays_rgba, shortlist_size, *, perceptual=True):
    """Return each target's best backing/overlay pairs and their distances."""
    if perceptual:
        targets = oklab_features(targets_rgb)
        target_norms = np.einsum("ij,ij->i", targets, targets)
    else:
        targets = targets_rgb.reshape(len(targets_rgb), -1, 3).astype(np.float32) / 255.0
        target_norms = np.einsum("tpc,tpc->tc", targets, targets)
    combination_count = len(backs_rgba) * len(overlays_rgba)
    keep = min(shortlist_size, combination_count)
    best_scores = np.full((len(targets), keep), np.inf, dtype=np.float32)
    best_indices = np.full((len(targets), keep), -1, dtype=np.int64)
    batches = range(0, combination_count, SCORE_BATCH_SIZE)
    total = (combination_count + SCORE_BATCH_SIZE - 1) // SCORE_BATCH_SIZE

    for start in tqdm(batches, total=total, desc="Scoring two-layer combinations"):
        stop = min(start + SCORE_BATCH_SIZE, combination_count)
        indices = np.arange(start, stop)
        back_indices, overlay_indices = divmod(indices, len(overlays_rgba))
        composites = blend_rgb(backs_rgba[back_indices, ..., :3], overlays_rgba[overlay_indices])
        if perceptual:
            candidates = oklab_features(composites)
            candidate_norms = np.einsum("ij,ij->i", candidates, candidates)
            scores = target_norms[:, None] + candidate_norms[None, :] - 2.0 * targets @ candidates.T
        else:
            candidates = composites.reshape(len(composites), -1, 3).astype(np.float32) / 255.0
            candidate_norms = np.einsum("kpc,kpc->kc", candidates, candidates)
            scores = np.zeros((len(targets), len(candidates)), dtype=np.float32)
            for channel in range(3):
                channel_ssd = (
                    target_norms[:, channel, None]
                    + candidate_norms[None, :, channel]
                    - 2.0 * targets[:, :, channel] @ candidates[:, :, channel].T
                )
                scores += channel_ssd * channel_ssd
        best_scores, best_indices = _merge_shortlist(best_scores, best_indices, scores, start)

    order = np.argsort(best_scores, axis=1)
    best_scores = np.take_along_axis(best_scores, order, axis=1)
    best_indices = np.take_along_axis(best_indices, order, axis=1)
    pairs = np.stack(divmod(best_indices, len(overlays_rgba)), axis=-1)
    return pairs, np.maximum(best_scores, 0.0)


def find_three_layer_matches(targets_rgb, two_layer_pairs, backs_rgba, middles_rgba, fronts_rgba):
    """Refine each target's two-layer shortlist with every front texture."""
    target_features = oklab_features(targets_rgb)
    result = np.empty((len(targets_rgb), 3), dtype=np.int64)
    for tile in tqdm(range(len(targets_rgb)), desc="Scoring third-layer combinations"):
        pairs = two_layer_pairs[tile]
        middle_rgb = blend_rgb(backs_rgba[pairs[:, 0], ..., :3], middles_rgba[pairs[:, 1]])
        candidates = blend_rgb(
            np.repeat(middle_rgb, len(fronts_rgba), axis=0),
            np.tile(fronts_rgba, (len(pairs), 1, 1, 1)),
        )
        features = oklab_features(candidates)
        delta = features - target_features[tile]
        winner = np.argmin(np.einsum("ij,ij->i", delta, delta))
        pair_index, front_index = divmod(int(winner), len(fronts_rgba))
        result[tile] = (*pairs[pair_index], front_index)
    return result


def block_name(path: Path) -> str:
    return path.stem.replace("_", " ")


def tile_array(image: Image.Image, width: int, height: int) -> np.ndarray:
    resized = image.resize((width * PIXELS, height * PIXELS), Image.Resampling.LANCZOS)
    pixels = np.asarray(resized.convert("RGB"), dtype=np.uint8)
    return pixels.reshape(height, PIXELS, width, PIXELS, 3).transpose(2, 0, 1, 3, 4).reshape(-1, PIXELS, PIXELS, 3)


def tile_image(tiles: np.ndarray, width: int, height: int, mode: str) -> Image.Image:
    pixels = tiles.reshape(width, height, PIXELS, PIXELS, -1).transpose(1, 2, 0, 3, 4)
    return Image.fromarray(pixels.reshape(height * PIXELS, width * PIXELS, -1), mode)


def save_results(source, output_folder, width, height, layers, matches,
                 back_paths, middle_paths, front_paths, backs, middles, fronts, video_frame=0):
    if video_frame > 0:
        output_folder = output_folder / "frames" / str(video_frame)
    output_folder.mkdir(parents=True, exist_ok=True)
    source.resize((width * PIXELS, height * PIXELS), Image.Resampling.LANCZOS).save(output_folder / "original.png")
    back_tiles = backs[matches[:, 0]]
    middle_tiles = middles[matches[:, 1]]
    composed = blend_rgb(back_tiles[..., :3], middle_tiles)
    tile_image(composed, width, height, "RGB").save(output_folder / "output.png")
    tile_image(back_tiles, width, height, "RGBA").save(output_folder / "output_backing.png")
    tile_image(middle_tiles, width, height, "RGBA").save(output_folder / "output_overlay.png")
    if layers == 3:
        front_tiles = fronts[matches[:, 2]]
        composed = blend_rgb(composed, front_tiles)
        tile_image(composed, width, height, "RGB").save(output_folder / "output.png")
        tile_image(front_tiles, width, height, "RGBA").save(output_folder / "output_overlay2.png")
    else:
        (output_folder / "output_overlay2.png").unlink(missing_ok=True)

    with (output_folder / "palette.txt").open("w", encoding="utf-8") as palette:
        for index, match in enumerate(matches):
            column, row = divmod(index, height)
            paths = [back_paths[match[0]], middle_paths[match[1]]]
            labels = ["Back", "Middle" if layers == 3 else "Front"]
            if layers == 3:
                paths.append(front_paths[match[2]])
                labels.append("Front")
            palette.write(f"Column {column + 1}, row {row + 1}\n")
            for label, path in zip(labels, paths):
                palette.write(f"{label}: {block_name(path)} ({path})\n")
            palette.write("\n")
    return output_folder


def read_positive_int(prompt: str) -> int:
    while True:
        try:
            value = int(input(prompt))
            if value > 0:
                return value
        except ValueError:
            pass
        print("Enter a positive whole number.")


def process_visual(visual_file, width, height, layers, output_folder, video_frame=0) -> None:
    parser = configparser.ConfigParser()
    parser.read(ROOT / "userpref.ini")
    shortlist_size = max(1, parser.getint("DEFAULT", "candidates", fallback=25))

    # Get textures
    front_paths = texture_files("front", "front-side")
    middle_paths = front_paths + texture_files("mid", "mid-side")
    back_paths = texture_files("back", "back-side")
    print("Loading textures.")
    fronts = load_textures(front_paths)
    middles = load_textures(middle_paths)
    backs = load_textures(back_paths)

    # Get specified file tiles
    targets = tile_array(visual_file, width, height)

    # Find matches
    pairs, _ = find_two_layer_matches(targets, backs, middles, shortlist_size if layers == 3 else 1)
    matches = find_three_layer_matches(targets, pairs, backs, middles, fronts) if layers == 3 else pairs[:, 0]

    # Save
    output_folder = save_results(
        visual_file, output_folder, width, height, layers, matches,
        back_paths, middle_paths, front_paths, backs, middles, fronts,
        video_frame
    )
    print(f"Completed. Output: {output_folder}")

def main():
    # Specify file
    if TKINTER_FLAG:
        print("Select a file to convert.")
        visual_filename = askopenfilename()
    else:
        print("tkinter not detected. Please put an input file into the input_files folder, and type its name here to process it.")
        visual_filename = ROOT / "input_files" / input().strip()
    if not visual_filename:
        print("No file was selected.")
        return

    file_path = Path(visual_filename)
    output_folder = OUTPUTS / file_path.name

    # Specify dimensions and layers
    width = read_positive_int("Width in blocks: ")
    height = read_positive_int("Height in blocks: ")
    while (layer_choice := input("Layers (2 or 3): ").strip()) not in {"2", "3"}:
        print("Enter 2 or 3.")
    layers = int(layer_choice)
    
    # Get file extension
    ext = file_path.suffix
    if ext == '.gif':
        print("GIF detected")
        frame_index = 0
        output_imgs = []
        max_duration = 0
        with Image.open(visual_filename) as visual_file:
            for frame in ImageSequence.Iterator(visual_file):
                frame_index += 1
                max_duration = max(max_duration, frame.info.get("duration", 100))
                process_visual(frame, width, height, layers, output_folder, video_frame=frame_index)
        if frame_index == 0:
            # GIF has no frames
            return
        for i in range(frame_index):
            frame_output = output_folder / "frames" / str(i+1) / "output.png"
            output_imgs.append(Image.open(frame_output))
        output_imgs[0].save(output_folder / "output.gif", save_all=True, append_images=output_imgs[1:], duration=max_duration, loop=0)
        for i in range(frame_index):
            output_imgs[i].close()
    else:
        with Image.open(visual_filename) as visual_file:
            process_visual(visual_file, width, height, layers, output_folder)

if __name__ == "__main__":
    main()
