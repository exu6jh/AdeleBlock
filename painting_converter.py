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

# ===== Process image files into NP arrays ===== #

def texture_files(*folders: str) -> list[Path]:
    """Take a tuple of texture folder names and return a list of all texture paths contained within
    
    Args:
        folders: Strings corresponding to names of folders within the root textures folder
    
    Returns:
        out: The list of paths to all files within all of the specified folders
    """
    return sorted(path for folder in folders for path in (TEXTURES / folder).iterdir() if path.is_file())

def load_textures(paths: list[Path]) -> np.ndarray:
    """Take a list of texture file paths, ensure they are of right dimension / number of channels, then combine their channel values into one array
    
    Args:
        paths: A list of texture file paths, such as that returned by the texture_files function
    
    Returns:
        out: An NP array of stacked image channel values, dimension [len(paths), PIXELS, PIXELS, 4]
    
    Raises:
        ValueError: If the loaded texture does not have dimensions PIXELS by PIXELS, or does not have 4 channels
    """
    textures = []
    for path in paths:
        with Image.open(path) as image:
            # Read image as NP array
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        if rgba.shape != (PIXELS, PIXELS, 4):
            # Confirm dimension (PIXELS * PIXELS) and channels (4)
            raise ValueError(f"Texture must be {PIXELS} x {PIXELS} pixels: {path}")
        textures.append(rgba)
    # Stack arrays in the first dimension
    return np.stack(textures)

def tile_array(image: Image.Image, width: int, height: int) -> np.ndarray:
    """Create an NP array corresponding to a Pillow image resized to be a given number of blocks wide and tall
    
    Args:
        image: Pillow image to be reshaped
        width: Width of resized image in terms of blocks (which are PIXEL=16 pixels wide)
        height: Height of resized image in terms of blocks (which are PIXEL=16 pixels tall)
    
    Returns:
        tiles: An NP array of block-divided image values, dimension [height * width, PIXELS, PIXELS, 3]
        resized: the resized image, used only to be saved to outputs
    """
    resized = image.resize((width * PIXELS, height * PIXELS), Image.Resampling.LANCZOS)
    # Conversion from RGB image to array produces array of dimensions (height * PIXELS, width * PIXELS, 3)
    pixels = np.asarray(resized.convert("RGB"), dtype=np.uint8)
    # Convert from array of dimensions (height * PIXELS, width * PIXELS, 3) to array of dimensions (height * width, PIXELS, PIXELS, 3)
    # First, reshape the array so that each section of size PIXELS is its own subarray in both height and width
    # Then transpose so that height and width are indexed first before height pixels and width pixels
    # Then reshape the array to combine height and width into one index
    return pixels.reshape(height, PIXELS, width, PIXELS, 3).transpose(2, 0, 1, 3, 4).reshape(height * width, PIXELS, PIXELS, 3), resized

def blend_rgb(background: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Composite RGBA overlays onto RGB backgrounds (assuming number of backgrounds and overlays are equal)
    
    Args:
        background: An NP array of block-divided background image values, dimension [#background, PIXELS, PIXELS, 3]
        overlay: An NP array of block-divided overlay image values, dimension [#overlays, PIXELS, PIXELS, 4]
    
    Returns:
        out: An NP array corresponding to the composite RGB values, dimension [#combos, PIXELS, PIXELS, 3]
    """
    alpha = overlay[..., 3:4].astype(np.uint16)
    foreground = overlay[..., :3].astype(np.uint16)
    background = background.astype(np.uint16)
    return ((foreground * alpha + background * (255 - alpha) + 127) // 255).astype(np.uint8)

def oklab_features(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB pixels to OKLab feature vectors
    
    Args:
        rgb: An NP array corresponding to block-divided sRGB values, dimension [#images, PIXELS, PIXELS, 3]

    Returns:
        out: An NP array of block-divided flattened Oklab values, dimension [#images, 3 * PIXELS^2]
    """
    # Scale sRGB (0-255) to a 0-1 fractional value
    values = rgb.astype(np.float32) / 255.0
    # Convert sRGB to linear RGB using standard transfer function
    values = np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)
    # This makes it so that the channel index is top level, i.e. [0] corresponds to red over all pixels, [1] to green, etc.
    red, green, blue = np.moveaxis(values, -1, 0)
    # Apply linear transformation to get LMS-like space, then apply cube root
    cbrt_long = np.cbrt(0.41222147 * red + 0.53633254 * green + 0.05144599 * blue)
    cbrt_medium = np.cbrt(0.21190350 * red + 0.68069955 * green + 0.10739696 * blue)
    cbrt_short = np.cbrt(0.08830246 * red + 0.28171884 * green + 0.62997870 * blue)
    # Apply linear transformation and stack in new dimension to get Oklab values, same dimension as input
    lab = np.stack(
        (
            0.21045426 * cbrt_long + 0.79361779 * cbrt_medium - 0.00407205 * cbrt_short,
            1.97799850 * cbrt_long - 2.42859221 * cbrt_medium + 0.45059371 * cbrt_short,
            0.02590404 * cbrt_long + 0.78277177 * cbrt_medium - 0.80867577 * cbrt_short,
        ),
        axis=-1,
    )
    # Flatten RGB values (PIXELS * PIXELS pixels, 3 channels) to one subarray
    return lab.reshape(rgb.shape[0], -1)

# ===== Compare targets and images for the best matches ===== #

def _merge_shortlist(best_scores, best_indices, scores, offset):
    """Take the S (shortlist_size) current best scores and indices, then merge in new scores
    
    Args:
        best_scores: The S best scores at each block, dimension [width * height, S]
        best_indices: The S best backing-overlay flattened indices at each block, dimension [width * height, S]
        scores: New candidate scores to merge in at each block, dimension [width * height, #candidates]
        offset: The offset to add to the scores array index to get the combination index
    
    Returns:
        out: A tuple of new best scores and indices, same dimension as best_scores and best_indices
    """
    # Shortlist size
    keep = best_scores.shape[1]
    # Preemptively shorten incoming scores to at most shortlist size
    count = min(keep, scores.shape[1])
    local = np.argpartition(scores, count - 1, axis=1)[:, :count]
    # Concatenate shortened list of scores and indices
    merged_scores = np.concatenate((best_scores, np.take_along_axis(scores, local, axis=1)), axis=1)
    merged_indices = np.concatenate((best_indices, local + offset), axis=1)
    # Filter to best scores, shortlist size
    selected = np.argpartition(merged_scores, keep - 1, axis=1)[:, :keep]
    return (
        np.take_along_axis(merged_scores, selected, axis=1),
        np.take_along_axis(merged_indices, selected, axis=1),
    )

def find_two_layer_matches(targets_rgb, backs_rgba, overlays_rgba, shortlist_size, *, perceptual=True):
    """Return each target's best backing/overlay pairs and their distances
    
    Args:
        targets_rgb: Target image tiles represented as an array, dimension [width * height, PIXELS, PIXELS, 3]
        backs_rgba: Backing images represented as an array, dimension [#backing, PIXELS, PIXELS, 4]
        overlays_rgba: Overlay images represented as an array, dimension [#overlays, PIXELS, PIXELS, 4]
        shortlist_size: The number of best scores/indices to keep, hereon shortened to S

    Returns:
        candidate_pairs: An NP array of the best backing/overlay pairs at each block, dimension [width * height, S, 2]
        candidate_scores: The corresponding scores at each block, dimension [width * height, S]
    """
    if perceptual:
        # Dimension [width * height, 3 * PIXELS ^ 2]
        targets = oklab_features(targets_rgb)
        # Einstein summation notation; this goes through each block and sums the square of the Oklab value
        # Dimension [width * height], each value corresponds to the norm in each block
        target_norms = np.einsum("ij,ij->i", targets, targets)
    else:
        # Dimension [width * height, PIXELS ^ 2, 3]
        targets = targets_rgb.reshape(len(targets_rgb), -1, 3).astype(np.float32) / 255.0
        # Einstein summation notation; this goes through each block, and for each channel, sums the square of the value
        # Dimension [width * height, 3], each value corresponds to the norm in each block-channel
        target_norms = np.einsum("tpc,tpc->tc", targets, targets)
    # Number of backing block / overlay block combinations
    combination_count = len(backs_rgba) * len(overlays_rgba)
    keep = min(shortlist_size, combination_count)
    # The S closest-matching block combo scores and their block indices
    best_scores = np.full((len(targets), keep), np.inf, dtype=np.float32)
    best_indices = np.full((len(targets), keep), -1, dtype=np.int64)
    # The number of backing-overlay combos to score at once
    batches = range(0, combination_count, SCORE_BATCH_SIZE)
    total = (combination_count + SCORE_BATCH_SIZE - 1) // SCORE_BATCH_SIZE

    for start in tqdm(batches, total=total, desc=f"Scoring two-layer combinations, {SCORE_BATCH_SIZE} combinations at a time"):
        # Backing / overlay indices range
        stop = min(start + SCORE_BATCH_SIZE, combination_count)
        indices = np.arange(start, stop)
        back_indices, overlay_indices = divmod(indices, len(overlays_rgba))
        # Get composites of all of the backing block and overlay blocks to be tested
        composites = blend_rgb(backs_rgba[back_indices, ..., :3], overlays_rgba[overlay_indices])
        if perceptual:
            # Like above, get Oklab values for the candidates and then get the norm in each block
            # Dimension [#composites, 3 * PIXELS ^ 2]
            candidates = oklab_features(composites)
            # Dimension [#composites]
            candidate_norms = np.einsum("ij,ij->i", candidates, candidates)
            # Apply distance function (||x||^2 - 2(x * y) + ||y||^2)
            scores = target_norms[:, None] + candidate_norms[None, :] - 2.0 * targets @ candidates.T
        else:
            # Like above, get the norm for each block-channel
            # Dimension [width * height, PIXELS ^ 2, 3]
            candidates = composites.reshape(len(composites), -1, 3).astype(np.float32) / 255.0
            # Dimension [width * height, 3]
            candidate_norms = np.einsum("kpc,kpc->kc", candidates, candidates)
            scores = np.zeros((len(targets), len(candidates)), dtype=np.float32)
            # Apply distance function to each channel and add to score
            for channel in range(3):
                channel_ssd = (
                    target_norms[:, channel, None]
                    + candidate_norms[None, :, channel]
                    - 2.0 * targets[:, :, channel] @ candidates[:, :, channel].T
                )
                scores += channel_ssd * channel_ssd
        # Update best scores and corresponding indices
        best_scores, best_indices = _merge_shortlist(best_scores, best_indices, scores, start)
    # Sort the best scores and their corresponding indices
    order = np.argsort(best_scores, axis=1)
    best_scores = np.take_along_axis(best_scores, order, axis=1)
    best_indices = np.take_along_axis(best_indices, order, axis=1)
    # Unflatten indices into backing and overlay block indices
    pairs = np.stack(divmod(best_indices, len(overlays_rgba)), axis=-1)
    return pairs, np.maximum(best_scores, 0.0)

def find_three_layer_matches(targets_rgb, two_layer_pairs, backs_rgba, middles_rgba, fronts_rgba):
    """Refine each target's two-layer shortlist with every front texture.
    
    Args:
        targets_rgb: Target image tiles represented as an array, dimension [width * height, PIXELS, PIXELS, 3]
        two_layer_pairs: An NP array of the best two-layer pairs at each block, dimension [width * height, S, 2]
        backs_rgba: Backing images represented as an array, dimension [#backing, PIXELS, PIXELS, 4]
        middles_rgba: Middle layer images represented as an array, dimension [#middle, PIXELS, PIXELS, 4]
        fronts_rgba: Front layer images represented as an array, dimension [#front, PIXELS, PIXELS, 4]
    
    Returns:
        out: NP array of file indices in the best three-layer combination found, dimension [width * height, 3]
    """
    # Convert target imag tiles into Oklab
    target_features = oklab_features(targets_rgb)
    result = np.empty((len(targets_rgb), 3), dtype=np.int64)
    for tile in tqdm(range(len(targets_rgb)), desc="Scoring third-layer combinations"):
        # Get the S best pairs
        pairs = two_layer_pairs[tile]
        # Construct all corresponding pair images as arrays
        middle_rgb = blend_rgb(backs_rgba[pairs[:, 0], ..., :3], middles_rgba[pairs[:, 1]])
        # Construct all candidate pair, overlay image combos
        candidates = blend_rgb(
            np.repeat(middle_rgb, len(fronts_rgba), axis=0),
            np.tile(fronts_rgba, (len(pairs), 1, 1, 1)),
        )
        # Convert candidates into Oklab for comparison
        features = oklab_features(candidates)
        # Get best candidate-target distance directly
        delta = features - target_features[tile]
        winner = np.argmin(np.einsum("ij,ij->i", delta, delta))
        # Convert best candidate to indices
        pair_index, front_index = divmod(int(winner), len(fronts_rgba))
        result[tile] = (*pairs[pair_index], front_index)
    return result

# ===== Save results =====

def tile_image(tiles: np.ndarray, width: int, height: int, mode: str) -> Image.Image:
    """Convert
    
    Args:
        tiles: NP array of block-divided image information, dimension [width * height, PIXELS, PIXELS, channels]
        width: The width in blocks
        height: The height in blocks
        mode: Image modes (e.g. "RGB", "RGBA")
    
    Returns:
        out: Tiled image constructed from each image in tiles
    """
    # Reshape and transpose to [height, PIXELS, width, PIXELS, channels]
    pixels = tiles.reshape(width, height, PIXELS, PIXELS, -1).transpose(1, 2, 0, 3, 4)
    # Flatten to height * PIXELS, width * PIXELS, and channels, then produce image
    return Image.fromarray(pixels.reshape(height * PIXELS, width * PIXELS, -1), mode)

def save_results(resized, output_folder, width, height, layers, matches, back_paths, middle_paths, front_paths, backs, middles, fronts, video_frame=0):
    """Save all result images and palette info
    
    Args:
        resized: The resized image
        output_folder: The output folder to put all output images and palette information
        width: The width in blocks
        height: The height in blocks
        layers: Number of layers to be used
        matches: NP array of file indices in the best combination found, dimension [width * height, layers]
        back_paths: List of all backing image paths
        middle_paths: List of all middle image paths
        front_paths: List of all front image paths
        backs: NP array of all backing images, dimension [#background, PIXELS, PIXELS, 4]
        middles: NP array of all middle images, dimension [#middle, PIXELS, PIXELS, 4]
        fronts: NP array of all front images, dimension [#front, PIXELS, PIXELS, 4]
        video_frame: Optional argument for animated media (GIFs, videos, etc.)
    
    Returns:
        output_folder: In certain cases (e.g. video), the output folder may be modified
    """
    # Videos use specific frame folders
    if video_frame > 0:
        output_folder = output_folder / "frames" / str(video_frame)
    # Make new output folder
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Save resized image
    resized.save(output_folder / "original.png")
    # Get all backing and middle blocks as block-divided NP arrays
    back_tiles = backs[matches[:, 0]]
    middle_tiles = middles[matches[:, 1]]
    # Compose arrays to form backing-middle block comboss
    composed = blend_rgb(back_tiles[..., :3], middle_tiles)
    # Construct images from NP arrays and save
    tile_image(composed, width, height, "RGB").save(output_folder / "output.png")
    tile_image(back_tiles, width, height, "RGBA").save(output_folder / "output_backing.png")
    tile_image(middle_tiles, width, height, "RGBA").save(output_folder / "output_overlay.png")
    # Three-layer case
    if layers == 3:
        # Same process as above
        front_tiles = fronts[matches[:, 2]]
        composed = blend_rgb(composed, front_tiles)
        tile_image(composed, width, height, "RGB").save(output_folder / "output.png")
        tile_image(front_tiles, width, height, "RGBA").save(output_folder / "output_overlay2.png")
    else:
        # Delete front overlay image if present
        (output_folder / "output_overlay2.png").unlink(missing_ok=True)

    # Save palette info
    with (output_folder / "palette.txt").open("w", encoding="utf-8") as palette:
        for index, match in enumerate(matches):
            column, row = divmod(index, height)
            # Get path names and labels
            paths = [back_paths[match[0]], middle_paths[match[1]]]
            labels = ["Back", "Middle" if layers == 3 else "Front"]
            if layers == 3:
                paths.append(front_paths[match[2]])
                labels.append("Front")
            # Write info
            palette.write(f"Column {column + 1}, row {row + 1}\n")
            for label, path in zip(labels, paths):
                palette.write(f"{label}: {path.stem.replace("_", " ")} ({path})\n")
            palette.write("\n")
    return output_folder

# ===== Main processing functions =====

def read_positive_int(prompt: str) -> int:
    """Read positive integer from input

    Args:
        prompt: Prompt as written before input field
    
    Returns:
        out: Positive integer as input by user
    """
    while True:
        try:
            value = int(input(prompt))
            if value > 0:
                return value
        except ValueError:
            pass
        print("Enter a positive whole number.")

def process_image(visual_file, width, height, layers, output_folder, video_frame=0) -> None:
    """Main visual processing function, take in image, dimensions, layers, and output info, and then save info to outputs
    
    Args:
        visual_file: Pillow image file to process
        width: Width in blocks
        height: Height in blocks
        layers: Number of layers to use
        output_folder: Name of output folder
        video_frame: Optional argument for animated media (GIFs, videos, etc.)
    
    Returns:
        None, all outputs are saved
    """
    # Get candidate shortlist size
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
    targets, resized = tile_array(visual_file, width, height)

    # Find best matches
    pairs, _ = find_two_layer_matches(targets, backs, middles, shortlist_size if layers == 3 else 1)
    matches = find_three_layer_matches(targets, pairs, backs, middles, fronts) if layers == 3 else pairs[:, 0]

    # Save all output images and palette info
    output_folder = save_results(
        resized, output_folder, width, height, layers, matches, back_paths,
        middle_paths, front_paths, backs, middles, fronts, video_frame
    )
    print(f"Completed. Output: {output_folder}")

def main():
    """Main processing function; take user input on file, width, height, and layers, and then output to output folder"""
    # Specify file
    if TKINTER_FLAG:
        print("Select a file to convert.")
        visual_filename = askopenfilename()
    else:
        print("tkinter not detected.\nPlease put the file into the input_files folder, and type its name here (including the file extension) to process it.")
        visual_filename = ROOT / "input_files" / input().strip()
    if not visual_filename:
        print("No file was selected.")
        return

    # Get input file and output folder
    file_path = Path(visual_filename)
    output_folder = OUTPUTS / file_path.stem

    # Specify dimensions and layers
    width = read_positive_int("Width in blocks: ")
    height = read_positive_int("Height in blocks: ")
    while (layer_choice := input("Layers (2 or 3): ").strip()) not in {"2", "3"}:
        print("Please specify either 2 or 3 layers.")
    layers = int(layer_choice)
    
    # Get file extension
    ext = file_path.suffix
    # Handle processing of GIFs differently
    if ext == '.gif':
        print("GIF detected.")
        # Process all frames, keeping track of number of frames and max duration
        frame_index, max_duration = 0, 0
        with Image.open(visual_filename) as visual_file:
            for frame in ImageSequence.Iterator(visual_file):
                frame_index += 1
                max_duration = max(max_duration, frame.info.get("duration", 100))
                process_image(frame, width, height, layers, output_folder, video_frame=frame_index)

        if frame_index == 0:
            # GIF has no frames
            return
        # Open all output images
        output_imgs = []
        for i in range(frame_index):
            frame_output = output_folder / "frames" / str(i+1) / "output.png"
            output_imgs.append(Image.open(frame_output))
        # Save to a GIF
        output_imgs[0].save(output_folder / "output.gif", save_all=True, append_images=output_imgs[1:], duration=max_duration, loop=0)
        # Close all output images
        for i in range(frame_index):
            output_imgs[i].close()
    else:
        # Simple image processing
        with Image.open(visual_filename) as visual_file:
            process_image(visual_file, width, height, layers, output_folder)

if __name__ == "__main__":
    main()
