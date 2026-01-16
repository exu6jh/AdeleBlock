from PIL import Image
import numpy as np
from tkinter.filedialog import askopenfilename
import math, os, sys, time
import image_correlator
import itertools
from tqdm import tqdm

def preface(pre, post):
    return pre + "/" + post

def preface_tuple(pre, tup):
    return tuple(map(lambda x : preface(pre, x), tup))

def get_filenames(file):
    return [filename for filename in os.listdir(file) if os.path.isfile(preface(file, filename))]

def preface_files(file):
    return preface_tuple(file, get_filenames(file))

def rms(a):
    return np.sqrt(np.mean(a*a))

def get_pos_in_array(val, arr):
    i = 0
    while(val > arr[i]):
        i = i + 1
    return i-1

def get_block_name(filename):
    return ' '.join(filename.split('/')[-1][:-4].split('_'))

# Every file used should be a square of the same size, as shown below.
COMMON_PIXEL_SIZE = 16
# If third layer mode is active, we reduce the number of base two-block candidates searched
CANDIDATES = 25
# If third layer mode is active, we construct a square out of available files
THIRD_SQ_SIZE = -1
THIRD_LAYER = None
# There are too many file combinations for the FFT info to be comfortably stored in memory
# We chunk into sections (dynamically set below)
BACK_CHUNK = -1
MIDFRONT_CHUNK = -1

# Store filenames for different sections
FRONT_FILENAMES = ()
MID_FILENAMES = ()
MIDFRONT_FILENAMES = ()
BACK_FILENAMES = ()

WIDTH = -1
HEIGHT = -1

USE_CACHING = False
LAYERS = 3

def process_tile(a, b):
    print(f"\nConstructing painting block at column {a+1}/{WIDTH}, row {b+1}/{HEIGHT} (#{a*HEIGHT+b+1} out of {WIDTH*HEIGHT} total)")

    comparison_file = f"paintingtiles/tile_{a}_{b}.png"
    
    # Find the closest block combo
    blockdists = np.zeros((len(MIDFRONT_FILENAMES), len(BACK_FILENAMES)))
    for x, y in tqdm(itertools.product(range(BACK_CHUNK), range(MIDFRONT_CHUNK)), desc="Searching for candidates in combination tiles", total=BACK_CHUNK*MIDFRONT_CHUNK):
        blocktile_file = f"blocktiles/tile_{x}_{y}.png"
        # Get distance between painting tile and combination chunk by offset
        distances_rgb = image_correlator.distance_by_offset(blocktile_file, comparison_file, cache=USE_CACHING).real
        # Since distance is on a per-pixel basis, while blocks are COMMON_PIXEL_SIZE pixels,
        # we scale down by that factor to find distances on a per-block basis
        blockdists_tile = distances_rgb[::COMMON_PIXEL_SIZE,::COMMON_PIXEL_SIZE]
        # Also, since distance is on a per-color basis, we combine distances over colors through RMS
        # to get a total block distance. The method can be changed as needed based on results
        blockdists_tile_rms = np.apply_along_axis(rms, 2, blockdists_tile)
        # Add distances for combination chunk to overall distance array
        blockdists[midfront_chunk_indices[y]:midfront_chunk_indices[y+1], back_chunk_indices[x]:back_chunk_indices[x+1]] = blockdists_tile_rms
    
    # Third layer processing
    if LAYERS == 3:
        # Get the top {CANDIDATES} best two-layer block combos
        blockdists_array = []
        for x, y in itertools.product(range(len(BACK_FILENAMES)), range(len(MIDFRONT_FILENAMES))):
            blockdists_array.append((blockdists[y,x], x, y))
        blockdists_array = sorted(blockdists_array, key=lambda dist: dist[0])[:CANDIDATES]

        # Get all three-layer distances from those two-layer block combos
        thirdlayer_dists = []
        for (_, x, y) in tqdm(blockdists_array, desc="Searching for third layer candidates"):
            # Generate a grid of the current two-layer block combo...
            combo_thirdlayer = Image.new('RGBA', (THIRD_SQ_SIZE * COMMON_PIXEL_SIZE, THIRD_SQ_SIZE * COMMON_PIXEL_SIZE))
            combo = Image.open(BACK_FILENAMES[x]).convert('RGBA')
            combo_front = Image.open(MIDFRONT_FILENAMES[y]).convert('RGBA')
            combo.paste(combo_front,(0,0),combo_front)
            combo_front.close()
            for i in range(THIRD_SQ_SIZE):
                for j in range(THIRD_SQ_SIZE):
                    combo_thirdlayer.paste(combo,(i * COMMON_PIXEL_SIZE,j * COMMON_PIXEL_SIZE))
            combo.close()
            #... so that the third layer image can be pasted on top.
            combo_thirdlayer.paste(THIRD_LAYER,(0,0),THIRD_LAYER)
            combo_thirdlayer_file = f"candidates_thirdlayer/tile_{a}_{b}_block_{x}_{y}.png"
            combo_thirdlayer.save(f"cache_images/{combo_thirdlayer_file}")
            combo_thirdlayer.close()
            
            # Get all distances by pixel, and once again scale down to get distances by block instead
            distances_rgb = image_correlator.distance_by_offset(combo_thirdlayer_file, comparison_file, cache=False).real
            thirdlayer_combodists = distances_rgb[::COMMON_PIXEL_SIZE,::COMMON_PIXEL_SIZE]
            thirdlayer_combodistsrms = np.apply_along_axis(rms, 2, thirdlayer_combodists)
            # Add to list of distances
            for i in range(len(FRONT_FILENAMES)):
                col, row = i // THIRD_SQ_SIZE, i % THIRD_SQ_SIZE
                thirdlayer_dists.append((x, y, col, row, thirdlayer_combodistsrms[row, col]))
        
        # Once again get the top {CANDIDATES} best distances, these time for three-layer block combos.
        # TODO: use these top {CANDIDATES} to provide alternate options
        lowest_dists = sorted(thirdlayer_dists, key=lambda dist: dist[4])[:CANDIDATES]
        
        # Get the best three-layer block combo found
        final_block_info = lowest_dists[0]

        # Final backing block image
        final_back = BACK_FILENAMES[final_block_info[0]]
        # Final second block
        final_midfront = MIDFRONT_FILENAMES[final_block_info[1]]
        # Final third block
        final_front_index = final_block_info[2] * THIRD_SQ_SIZE + final_block_info[3]
        final_front = FRONT_FILENAMES[final_front_index]

        return (final_back, final_midfront, final_front)
    else:
        final_block_index = np.unravel_index(np.argmin(blockdists), blockdists.shape)

        # Final backing block image
        final_back = BACK_FILENAMES[final_block_index[1]]
        # Final overlay block
        final_front = MIDFRONT_FILENAMES[final_block_index[0]]
        
        return (final_back, final_front)

if __name__ == "__main__":
    # Get the painting and set up the output folder
    print("Please select a painting you would like blockified.")
    painting_filename = askopenfilename()
    print(f"Selected {painting_filename}\n")
    painting_im = Image.open(painting_filename).convert('RGB')
    output_folder = f"{os.getcwd()}/outputs/{os.path.basename(painting_filename)}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get blockified image width/height info, resize for reference
    print("Please enter a width.")
    WIDTH = int(input())
    print("Please enter a height.")
    HEIGHT = int(input())
    painting_im = painting_im.resize((WIDTH*COMMON_PIXEL_SIZE,HEIGHT*COMMON_PIXEL_SIZE))
    painting_im.save(f"{output_folder}/original.png")

    # Create palette information file
    palette_path = f"{output_folder}/palette.txt"
    # Clears file if it already exists, creates it if it doesn't
    f = open(palette_path, "w")
    f.close()

    layer_specified = False
    print("\nWould you like your image to have two or three layers? Three layers takes a few seconds longer per block. Type '2' or 'two' for the former, '3' or 'three' for the latter.")
    while(not layer_specified):
        layer_choice = input()
        if layer_choice.strip().lower() in ['2', 'two']:
            LAYERS = 2
            layer_specified = True
            print("Two-layer mode selected.")
        elif layer_choice.strip().lower() in ['3', 'three']:
            LAYERS = 3
            layer_specified = True
            print("Three-layer mode selected.")
        else:
            print("Sorry, could not understand your input. Type '2' or 'two' for two-layer mode, '3' or 'three' for three-layer mode.")

    # Ask about caching
    print("\nDo you want to cache new Fourier transform coefficient data? This will speed up calculations but also can take up significant amounts of storage (~9GB). Type 'yes' if so.")
    # We want the user to be very explicit about actually wanting this
    if(input().strip().lower()) == 'yes':
        print('Enabling caching of new coefficient data.\n')
        USE_CACHING = True
    else:
        print('Keeping new data caching disabled.\nNote that existing cache data will still be used.\n')
    
    # Clear previous cached information
    print("Clearing any cached data that may have been left over from previous session.")
    clear_folders = ["cache_images/paintingtiles","cache_fft/paintingtiles","cache_images/blocktiles","cache_fft/blocktiles","cache_fft/masks","cache_images/candidates_thirdlayer"]
    [os.remove(cached_info) for cached_info in list(itertools.accumulate([preface_files(f"{os.getcwd()}/{folder}") for folder in clear_folders]))[-1] if not ".gitignore" in cached_info]
    print("Cleared.\n")
    # Block files
    ## Front blocks
    front = f"{os.getcwd()}/textures/front"
    front_side = f"{os.getcwd()}/textures/front-side"
    ## Middle blocks
    mid = f"{os.getcwd()}/textures/mid"
    mid_side = f"{os.getcwd()}/textures/mid-side"
    ## Back blocks
    back = f"{os.getcwd()}/textures/back"
    back_side = f"{os.getcwd()}/textures/back-side"

    FRONT_FILENAMES = preface_files(front) + preface_files(front_side)
    MID_FILENAMES = preface_files(mid) + preface_files(mid_side)
    BACK_FILENAMES = preface_files(back) + preface_files(back_side)

    MIDFRONT_FILENAMES = FRONT_FILENAMES+MID_FILENAMES

    # Get third layer square size
    THIRD_SQ_SIZE = math.ceil(math.sqrt(len(FRONT_FILENAMES) + 1))

    # Dynamic number of block-overlay combo chunks based on how many blocks there are
    BACK_CHUNK = math.ceil(len(BACK_FILENAMES) / 100)
    MIDFRONT_CHUNK = math.ceil(len(MIDFRONT_FILENAMES) / 100)

    # Make the sizes of combo chunks relatively consistent
    midfront_chunk_indices = [round(len(MIDFRONT_FILENAMES) * i / MIDFRONT_CHUNK) for i in range(MIDFRONT_CHUNK + 1)]
    back_chunk_indices = [round(len(BACK_FILENAMES) * i / BACK_CHUNK) for i in range(BACK_CHUNK + 1)]

    # Save painting tiles
    for a, b in itertools.product(range(WIDTH), range(HEIGHT)):
        painting_crop = painting_im.crop((COMMON_PIXEL_SIZE * a, COMMON_PIXEL_SIZE * b, COMMON_PIXEL_SIZE * (a + 1), COMMON_PIXEL_SIZE * (b + 1)))
        painting_crop.save(f"cache_images/paintingtiles/tile_{a}_{b}.png")
    painting_im.close()
    
    # Check for block combo tiles, otherwise recreate
    if not all(os.path.isfile(f"{os.getcwd()}/cache_images/blocktiles/tile_{x}_{y}.png") for x, y in itertools.product(range(BACK_CHUNK),range(MIDFRONT_CHUNK))):
        for x, y in tqdm(itertools.product(range(BACK_CHUNK), range(MIDFRONT_CHUNK)), desc="Creating block combination tiles", total=BACK_CHUNK*MIDFRONT_CHUNK):
            # Generate row of backing blocks
            back_row = Image.new('RGBA', ((back_chunk_indices[x+1] - back_chunk_indices[x]) * COMMON_PIXEL_SIZE, COMMON_PIXEL_SIZE))
            for i in range(back_chunk_indices[x], back_chunk_indices[x+1]):
                back_row.paste(Image.open(BACK_FILENAMES[i]).convert('RGBA'), (COMMON_PIXEL_SIZE * (i - back_chunk_indices[x]), 0))
            # Copy rows vertically
            backing = Image.new('RGBA', ((back_chunk_indices[x+1] - back_chunk_indices[x]) * COMMON_PIXEL_SIZE, (midfront_chunk_indices[y+1] - midfront_chunk_indices[y]) * COMMON_PIXEL_SIZE))
            for j in range(midfront_chunk_indices[y], midfront_chunk_indices[y+1]):
                backing.paste(back_row, (0, COMMON_PIXEL_SIZE * (j - midfront_chunk_indices[y])))
            back_row.close()

            # Generate column of overlay blocks
            overlay_col = Image.new('RGBA', (COMMON_PIXEL_SIZE, (midfront_chunk_indices[y+1] - midfront_chunk_indices[y]) * COMMON_PIXEL_SIZE))
            for j in range(midfront_chunk_indices[y], midfront_chunk_indices[y+1]):
                overlay_j = Image.open(MIDFRONT_FILENAMES[j]).convert('RGBA')
                overlay_col.paste(overlay_j, (0,COMMON_PIXEL_SIZE*(j - midfront_chunk_indices[y])))
                overlay_j.close()
            # Copy columns horizonally
            overlay = Image.new('RGBA', ((back_chunk_indices[x+1] - back_chunk_indices[x]) * COMMON_PIXEL_SIZE, (midfront_chunk_indices[y+1] - midfront_chunk_indices[y]) * COMMON_PIXEL_SIZE))
            for i in range(back_chunk_indices[x], back_chunk_indices[x+1]):
                overlay.paste(overlay_col, (COMMON_PIXEL_SIZE*(i - back_chunk_indices[x]), 0))
            overlay_col.close()

            # Paste overlay tiles over backing tiles to get combination tiles
            backing.paste(overlay, (0, 0), overlay)
            overlay.close()
            backing.convert('RGB').save(f"cache_images/blocktiles/tile_{x}_{y}.png")
            backing.close()

    THIRD_LAYER = Image.new('RGBA', (THIRD_SQ_SIZE * COMMON_PIXEL_SIZE, THIRD_SQ_SIZE * COMMON_PIXEL_SIZE))
    for i in range(len(FRONT_FILENAMES)):
        front_i = Image.open(FRONT_FILENAMES[i]).convert('RGBA')
        THIRD_LAYER.paste(front_i, ((i // THIRD_SQ_SIZE) * COMMON_PIXEL_SIZE, (i % THIRD_SQ_SIZE) * COMMON_PIXEL_SIZE))
        front_i.close()

    # Actual main conversion section
    paintified = Image.new('RGB', (COMMON_PIXEL_SIZE * WIDTH, COMMON_PIXEL_SIZE * HEIGHT))
    paintified_backing = Image.new('RGB', (COMMON_PIXEL_SIZE * WIDTH, COMMON_PIXEL_SIZE * HEIGHT))
    paintified_overlay = Image.new('RGBA', (COMMON_PIXEL_SIZE * WIDTH, COMMON_PIXEL_SIZE * HEIGHT))
    paintified_overlay2 = Image.new('RGBA', (COMMON_PIXEL_SIZE * WIDTH, COMMON_PIXEL_SIZE * HEIGHT))
    for a, b in itertools.product(range(WIDTH), range(HEIGHT)):
        if LAYERS == 3:
            (final_back_file, final_midfront_file, final_front_file) = process_tile(a,b)

            # Process final block image for painting tile
            final_im = Image.new('RGB', (COMMON_PIXEL_SIZE, COMMON_PIXEL_SIZE))
            # Final backing block image
            final_back = Image.open(final_back_file).convert('RGBA')
            final_im.paste(final_back,(0,0))
            paintified_backing.paste(final_back,(COMMON_PIXEL_SIZE*a,COMMON_PIXEL_SIZE*b))
            final_back.close()
            # Final second block
            final_midfront = Image.open(final_midfront_file).convert('RGBA')
            final_im.paste(final_midfront,(0,0),final_midfront)
            paintified_overlay.paste(final_midfront,(COMMON_PIXEL_SIZE*a,COMMON_PIXEL_SIZE*b),final_midfront)
            final_midfront.close()
            # Final third block
            final_front = Image.open(final_front_file).convert('RGBA')
            final_im.paste(final_front,(0,0),final_front)
            paintified_overlay2.paste(final_front,(COMMON_PIXEL_SIZE*a,COMMON_PIXEL_SIZE*b),final_front)
            final_front.close()
            # Final combination block
            paintified.paste(final_im,(COMMON_PIXEL_SIZE*a,COMMON_PIXEL_SIZE*b))
            final_im.close()
            paintified.save(f"{output_folder}/progress.png")

            # Save information to palette
            backstring = f"Back: {get_block_name(final_back_file)} ({final_back_file})"
            midstring = f"Middle: {get_block_name(final_midfront_file)} ({final_midfront_file})"
            frontstring = f"Front: {get_block_name(final_front_file)} ({final_front_file})"
            print(f"Block combination found.\n{backstring}\n{midstring}\n{frontstring}")
            with open(palette_path, "a") as f:
                f.write(f"Column {a+1}, row {b+1}\n{backstring}\n{midstring}\n{frontstring}\n\n")
        elif LAYERS == 2:
            (final_back_file, final_front_file) = process_tile(a,b)

            # Process final block image for painting tile
            final_im = Image.new('RGB', (COMMON_PIXEL_SIZE, COMMON_PIXEL_SIZE))
            # Final backing block image
            final_back = Image.open(final_back_file).convert('RGBA')
            final_im.paste(final_back,(0,0))
            paintified_backing.paste(final_back,(COMMON_PIXEL_SIZE*a,COMMON_PIXEL_SIZE*b))
            final_back.close()
            # Final overlay block
            final_front = Image.open(final_front_file).convert('RGBA')
            final_im.paste(final_front,(0,0),final_front)
            paintified_overlay.paste(final_front,(COMMON_PIXEL_SIZE*a,COMMON_PIXEL_SIZE*b),final_front)
            final_front.close()
            # Final combination block
            paintified.paste(final_im,(COMMON_PIXEL_SIZE*a,COMMON_PIXEL_SIZE*b))
            final_im.close()
            paintified.save(f"{output_folder}/progress.png")

            # Save information to palette
            backstring = f"Back: {get_block_name(final_back_file)} ({final_back_file})"
            frontstring = f"Front: {get_block_name(final_front_file)} ({final_front_file})"
            print(f"Block combination found.\n{backstring}\n{frontstring}")
            with open(palette_path, "a") as f:
                f.write(f"Column {a+1}, row {b+1}\n{backstring}\n{frontstring}\n\n")

        # Clear cache info
        paintingtile_fftcache = preface_files(f"{os.getcwd()}/cache_fft/paintingtiles")
        thirdlayer_imagecache = preface_files(f"{os.getcwd()}/cache_images/candidates_thirdlayer")
        [os.remove(cache_file) for cache_file in paintingtile_fftcache + thirdlayer_imagecache if f"tile_{a}_{b}" in cache_file and not ".gitignore" in cache_file]
    THIRD_LAYER.close()

    # Process final block image for whole painting
    print(f"\nImage processed, saving output to {output_folder}/")
    paintified.save(f"{output_folder}/output.png")
    paintified.close()
    paintified_backing.save(f"{output_folder}/output_backing.png")
    paintified_backing.close()
    paintified_overlay.save(f"{output_folder}/output_overlay.png")
    paintified_overlay.close()
    if LAYERS == 3:
        paintified_overlay2.save(f"{output_folder}/output_overlay2.png")
    elif os.path.isfile(preface(output_folder, 'output_overlay2.png')):
        os.remove(f"{output_folder}/output_overlay2.png")
    paintified_overlay2.close()

    # Clear unnecessary cached information
    print("Doing a final clean on some cached data.")
    if os.path.isfile(preface(output_folder, 'progress.png')):
        os.remove(f"{output_folder}/progress.png")
    [os.remove(cached_info) for cached_info in list(itertools.accumulate([preface_files(f"{os.getcwd()}/{folder}") for folder in clear_folders]))[-1] if not ".gitignore" in cached_info]
    print(f"Completed. Go to {output_folder}/ for the output.\n")