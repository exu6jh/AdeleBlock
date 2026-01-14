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
# There are too many file combinations for the FFT info to be comfortably stored in memory
# We chunk into sections (dynamically set below)
BACK_CHUNK = -1
MIDFRONT_CHUNK = -1

WIDTH = -1
HEIGHT = -1

USE_CACHING = False
LAYERS = 3

if __name__ == "__main__":
    # Get the painting and set up the output folder
    print("Please select a painting you would like blockified.")
    painting_filename = askopenfilename()
    print(str.format("Selected {0}\n", painting_filename))
    painting_im = Image.open(painting_filename).convert('RGB')
    output_folder = str.format(os.getcwd() + "/outputs/{0}", os.path.basename(painting_filename))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get blockified image width/height info, resize for reference
    print("Please enter a width.")
    WIDTH = int(input())
    print("Please enter a height.")
    HEIGHT = int(input())
    painting_im = painting_im.resize((WIDTH*COMMON_PIXEL_SIZE,HEIGHT*COMMON_PIXEL_SIZE))
    painting_im.save(str.format('{0}/original.png', output_folder))

    # Create palette information file
    palette_path = output_folder + "/palette.txt"
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
    paintingtile_fftcache = preface_files(os.getcwd() + "/cache_fft/paintingtiles")
    paintingtile_cache = preface_files(os.getcwd() + "/cache_images/paintingtiles")
    blocktile_cache = preface_files(os.getcwd() + "/cache_images/blocktiles")
    blocktile_fftcache = preface_files(os.getcwd() + "/cache_fft/blocktiles")
    mask_fftcache = preface_files(os.getcwd() + "/cache_fft/masks")
    candidates_cache = preface_files(os.getcwd() + "/cache_images/candidates_thirdlayer")
    for cached_info in paintingtile_fftcache + paintingtile_cache + blocktile_cache + blocktile_fftcache + mask_fftcache + candidates_cache:
        if not ".gitignore" in cached_info:
            os.remove(cached_info)
    print("Cleared.\n")

    # Block files
    ## Front blocks
    front = os.getcwd() + "/textures/front"
    front_side = os.getcwd() + "/textures/front-side"
    ## Middle blocks
    mid = os.getcwd() + "/textures/mid"
    mid_side = os.getcwd() + "/textures/mid-side"
    ## Back blocks
    back = os.getcwd() + "/textures/back"
    back_side = os.getcwd() + "/textures/back-side"

    front_filenames = preface_files(front) + preface_files(front_side)
    mid_filenames = preface_files(mid) + preface_files(mid_side)
    back_filenames = preface_files(back) + preface_files(back_side)

    midfront_filenames = front_filenames+mid_filenames

    # Dynamic number of block-overlay combo chunks based on how many blocks there are
    BACK_CHUNK = math.ceil(len(back_filenames) / 100)
    MIDFRONT_CHUNK = math.ceil(len(midfront_filenames) / 100)

    # Make the sizes of combo chunks relatively consistent
    midfront_chunk_indices = [round(len(midfront_filenames) * i / MIDFRONT_CHUNK) for i in range(MIDFRONT_CHUNK + 1)]
    back_chunk_indices = [round(len(back_filenames) * i / BACK_CHUNK) for i in range(BACK_CHUNK + 1)]

    # Save painting tiles
    for a, b in itertools.product(range(WIDTH), range(HEIGHT)):
        painting_crop = painting_im.crop((COMMON_PIXEL_SIZE * a, COMMON_PIXEL_SIZE * b, COMMON_PIXEL_SIZE * (a + 1), COMMON_PIXEL_SIZE * (b + 1)))
        painting_crop.save(str.format("cache_images/paintingtiles/tile{0}_{1}.png",a,b))
    
    # Check for block combo tiles, otherwise recreate
    if not all(os.path.isfile(os.getcwd() + str.format('cache_images/blocktiles/tile{0}_{1}.png', x, y)) for x, y in itertools.product(range(BACK_CHUNK),range(MIDFRONT_CHUNK))):
        for x, y in tqdm(itertools.product(range(BACK_CHUNK), range(MIDFRONT_CHUNK)), desc="Creating block combination tiles", total=BACK_CHUNK*MIDFRONT_CHUNK):
            # Generate row of backing blocks
            back_row = Image.new('RGBA', ((back_chunk_indices[x+1] - back_chunk_indices[x]) * COMMON_PIXEL_SIZE, COMMON_PIXEL_SIZE))
            for i in range(back_chunk_indices[x], back_chunk_indices[x+1]):
                back_row.paste(Image.open(back_filenames[i]).convert('RGBA'), (COMMON_PIXEL_SIZE * (i - back_chunk_indices[x]), 0))
            # Copy rows vertically
            backing = Image.new('RGBA', ((back_chunk_indices[x+1] - back_chunk_indices[x]) * COMMON_PIXEL_SIZE, (midfront_chunk_indices[y+1] - midfront_chunk_indices[y]) * COMMON_PIXEL_SIZE))
            for j in range(midfront_chunk_indices[y], midfront_chunk_indices[y+1]):
                backing.paste(back_row, (0, COMMON_PIXEL_SIZE * (j - midfront_chunk_indices[y])))

            # Generate column of overlay blocks
            overlay_col = Image.new('RGBA', (COMMON_PIXEL_SIZE, (midfront_chunk_indices[y+1] - midfront_chunk_indices[y]) * COMMON_PIXEL_SIZE))
            for j in range(midfront_chunk_indices[y], midfront_chunk_indices[y+1]):
                overlay_col.paste(Image.open(midfront_filenames[j]).convert('RGBA'), (0,COMMON_PIXEL_SIZE*(j - midfront_chunk_indices[y])))
            # Copy columns horizonally
            overlay = Image.new('RGBA', ((back_chunk_indices[x+1] - back_chunk_indices[x]) * COMMON_PIXEL_SIZE, (midfront_chunk_indices[y+1] - midfront_chunk_indices[y]) * COMMON_PIXEL_SIZE))
            for i in range(back_chunk_indices[x], back_chunk_indices[x+1]):
                overlay.paste(overlay_col, (COMMON_PIXEL_SIZE*(i - back_chunk_indices[x]), 0))

            # Paste overlay tiles over backing tiles to get combination tiles
            final = backing.copy()
            final.paste(overlay, (0, 0), overlay)
            final.convert('RGB').save(str.format('cache_images/blocktiles/tile{0}_{1}.png',x,y))

    third_sq_size = math.ceil(math.sqrt(len(front_filenames) + 1))
    third_layer = Image.new('RGBA', (third_sq_size * COMMON_PIXEL_SIZE, third_sq_size * COMMON_PIXEL_SIZE))
    for i in range(len(front_filenames)):
        front_i = Image.open(front_filenames[i]).convert('RGBA')
        third_layer.paste(front_i, ((i // third_sq_size) * COMMON_PIXEL_SIZE, (i % third_sq_size) * COMMON_PIXEL_SIZE))

    # Actual main conversion section
    paintified = Image.new('RGB', (COMMON_PIXEL_SIZE * WIDTH, COMMON_PIXEL_SIZE * HEIGHT))
    paintified_backing = Image.new('RGB', (COMMON_PIXEL_SIZE * WIDTH, COMMON_PIXEL_SIZE * HEIGHT))
    paintified_overlay = Image.new('RGBA', (COMMON_PIXEL_SIZE * WIDTH, COMMON_PIXEL_SIZE * HEIGHT))
    paintified_overlay2 = Image.new('RGBA', (COMMON_PIXEL_SIZE * WIDTH, COMMON_PIXEL_SIZE * HEIGHT))
    for a, b in itertools.product(range(WIDTH), range(HEIGHT)):
        print(str.format("\nConstructing painting block at column {0}/{1}, row {2}/{3} (#{4} out of {5} total)", a+1, WIDTH, b+1, HEIGHT, a * HEIGHT + b + 1, WIDTH * HEIGHT))

        comparison_file = str.format("paintingtiles/tile{0}_{1}.png",a,b)
        
        # Find the closest block combo
        blockdists = np.zeros((len(midfront_filenames), len(back_filenames)))
        for x, y in tqdm(itertools.product(range(BACK_CHUNK), range(MIDFRONT_CHUNK)), desc="Searching for candidates in combination tiles", total=BACK_CHUNK*MIDFRONT_CHUNK):
            blocktile_file = str.format('blocktiles/tile{0}_{1}.png',x,y)
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
            # Get the top 25 best two-layer block combos
            blockdists_array = []
            for x in range(len(back_filenames)):
                for y in range(len(midfront_filenames)):
                    blockdists_array.append((blockdists[y,x], x, y))
            blockdists_array = sorted(blockdists_array, key=lambda dist: dist[0])[:25]

            # Get all three-layer distances from those two-layer block combos
            thirdlayer_dists = []
            for (_, x, y) in tqdm(blockdists_array, desc="Searching for third layer candidates"):
                # Generate a grid of the current two-layer block combo...
                combo_thirdlayer = Image.new('RGBA', (third_sq_size * COMMON_PIXEL_SIZE, third_sq_size * COMMON_PIXEL_SIZE))
                combo = Image.open(back_filenames[x]).convert('RGBA')
                combo_front = Image.open(midfront_filenames[y]).convert('RGBA')
                combo.paste(combo_front,(0,0),combo_front)
                for i in range(third_sq_size):
                    for j in range(third_sq_size):
                        combo_thirdlayer.paste(combo,(i * COMMON_PIXEL_SIZE,j * COMMON_PIXEL_SIZE))
                #... so that the third layer image can be pasted on top.
                combo_thirdlayer.paste(third_layer,(0,0),third_layer)
                combo_thirdlayer_file = str.format("candidates_thirdlayer/{0}_{1}.png",x,y)
                combo_thirdlayer.save(str.format('cache_images/{0}', combo_thirdlayer_file))
                
                # Get all distances by pixel, and once again scale down to get distances by block instead
                distances_rgb = image_correlator.distance_by_offset(combo_thirdlayer_file, comparison_file, cache=False).real
                thirdlayer_combodists = distances_rgb[::COMMON_PIXEL_SIZE,::COMMON_PIXEL_SIZE]
                thirdlayer_combodistsrms = np.apply_along_axis(rms, 2, thirdlayer_combodists)
                # Add to list of distances
                for i in range(len(front_filenames) + 1):
                    col, row = i // third_sq_size, i % third_sq_size
                    thirdlayer_dists.append((x, y, col, row, thirdlayer_combodistsrms[row, col]))
            
            # Once again get the top 25 best distances, these time for three-layer block combos.
            # TODO: use these top 25 to provide alternate options
            lowest_dists = sorted(thirdlayer_dists, key=lambda dist: dist[4])[:25]
            
            # Get the best three-layer block combo found
            final_block_info = lowest_dists[0]
            final_im = Image.new('RGB', (COMMON_PIXEL_SIZE, COMMON_PIXEL_SIZE))
            # Final backing block image
            final_back_filename = back_filenames[final_block_info[0]]
            final_back = Image.open(final_back_filename).convert('RGBA')
            final_im.paste(final_back,(0,0))
            paintified_backing.paste(final_back,(COMMON_PIXEL_SIZE*a,COMMON_PIXEL_SIZE*b))
            # Final second block
            final_midfront_filename = midfront_filenames[final_block_info[1]]
            final_midfront = Image.open(final_midfront_filename).convert('RGBA')
            final_im.paste(final_midfront,(0,0),final_midfront)
            paintified_overlay.paste(final_midfront,(COMMON_PIXEL_SIZE*a,COMMON_PIXEL_SIZE*b),final_midfront)
            # Final third block
            final_front_index = final_block_info[2] * third_sq_size + final_block_info[3]
            final_front_filename = 'none'
            if final_front_index < len(front_filenames):
                final_front_filename = front_filenames[final_front_index]
                final_front = Image.open(final_front_filename).convert('RGBA')
                final_im.paste(final_front,(0,0),final_front)
                paintified_overlay2.paste(final_front,(COMMON_PIXEL_SIZE*a,COMMON_PIXEL_SIZE*b),final_front)
            # Final combination block
            paintified.paste(final_im,(COMMON_PIXEL_SIZE*a,COMMON_PIXEL_SIZE*b))
            paintified.save(str.format("{0}/progress.png", output_folder))

            # Save information to palette
            print("Block combination found.")
            backstring = str.format("Back: {0} ({1})", get_block_name(final_back_filename), final_back_filename)
            midstring = str.format("Middle: {0} ({1})", get_block_name(final_midfront_filename), final_midfront_filename)
            frontstring = str.format("Front: {0} ({1})", get_block_name(final_front_filename), final_front_filename)
            print(backstring)
            print(midstring)
            print(frontstring)
            with open(palette_path, "a") as f:
                f.write(str.format("Column {0}, row {1}\n", a+1, b+1))
                f.write(backstring + "\n")
                f.write(midstring + "\n")
                f.write(frontstring + "\n\n")
        else:
            final_block_index = np.unravel_index(np.argmin(blockdists), blockdists.shape)

            # Process final block image for painting tile
            final_im = Image.new('RGB', (COMMON_PIXEL_SIZE, COMMON_PIXEL_SIZE))
            # Final backing block image
            final_back_filename = back_filenames[final_block_index[1]]
            final_back = Image.open(final_back_filename).convert('RGBA')
            final_im.paste(final_back,(0,0))
            paintified_backing.paste(final_back,(COMMON_PIXEL_SIZE*a,COMMON_PIXEL_SIZE*b))
            # Final overlay block
            final_midfront_filename = midfront_filenames[final_block_index[0]]
            final_midfront = Image.open(final_midfront_filename).convert('RGBA')
            final_im.paste(final_midfront,(0,0),final_midfront)
            paintified_overlay.paste(final_midfront,(COMMON_PIXEL_SIZE*a,COMMON_PIXEL_SIZE*b),final_midfront)
            # Final combination block
            paintified.paste(final_im,(COMMON_PIXEL_SIZE*a,COMMON_PIXEL_SIZE*b))
            paintified.save(str.format("{0}/progress.png", output_folder))

            # Save information to palette
            print("Block combination found.")
            backstring = str.format("Back: {0} ({1})", get_block_name(final_back_filename), final_back_filename)
            frontstring = str.format("Front: {0} ({1})", get_block_name(final_midfront_filename), final_midfront_filename)
            print(backstring)
            print(frontstring)
            with open(palette_path, "a") as f:
                f.write(str.format("Column {0}, row {1}\n", a+1, b+1))
                f.write(backstring + "\n")
                f.write(frontstring + "\n\n")
        
        # Clear cache info
        paintingtile_fftcache = preface_files(os.getcwd() + "/cache_fft/paintingtiles")
        thirdlayer_candidates = preface_files(os.getcwd() + "/cache_images/candidates_thirdlayer")
        for cache_file in paintingtile_fftcache + thirdlayer_candidates:
            if not ".gitignore" in cache_file:
                os.remove(cache_file)
    
    # Process final block image for whole painting
    print(str.format("\nImage processed, saving output to {0}/", output_folder))
    paintified.save(str.format("{0}/output.png", output_folder))
    paintified_backing.save(str.format("{0}/output_backing.png", output_folder))
    paintified_overlay.save(str.format("{0}/output_overlay.png", output_folder))
    if LAYERS == 3:
        paintified_overlay2.save(str.format("{0}/output_overlay2.png", output_folder))
    elif os.path.isfile(preface(output_folder, 'output_overlay2.png')):
        os.remove(str.format("{0}/output_overlay2.png", output_folder))

    # Clear unnecessary cached information
    print("Doing a final clean on some cached data.")
    if os.path.isfile(preface(output_folder, 'progress.png')):
        os.remove(str.format("{0}/progress.png", output_folder))
    paintingtile_cache = preface_files(os.getcwd() + "/cache_images/paintingtiles")
    blocktile_cache = preface_files(os.getcwd() + "/cache_images/blocktiles")
    blocktile_fftcache = preface_files(os.getcwd() + "/cache_fft/blocktiles")
    mask_fftcache = preface_files(os.getcwd() + "/cache_fft/masks")
    candidates_cache = preface_files(os.getcwd() + "/cache_images/candidates_thirdlayer")
    for cache_file in paintingtile_cache + blocktile_cache + blocktile_fftcache + mask_fftcache + candidates_cache:
        if not ".gitignore" in cache_file:
            os.remove(cache_file)
    print(str.format("Completed. Go to {0}/ for the output.\n", output_folder))