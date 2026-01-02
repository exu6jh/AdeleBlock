from PIL import Image
import numpy as np
from tkinter.filedialog import askopenfilename
import math, os, sys, time
import image_correlator
import itertools

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

# Every file used should be a square of the same size, as shown below.
COMMON_PIXEL_SIZE = 16
# There are too many file combinations for the FFT info to be comfortably stored in memory
# We chunk into sections (dynamically set below)
BACK_CHUNK = -1
MIDFRONT_CHUNK = -1

WIDTH = -1
HEIGHT = -1

CACHING = False

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

    # Ask about caching
    print("\nDo you want to cache new Fourier transform coefficient data? This will speed up calculations but also can take up significant amounts of storage (~9GB). Type 'yes' if so.")
    # We want the user to be very explicit about actually wanting this
    if(input().strip().lower()) == 'yes':
        print('Enabling caching of new coefficient data.\n')
        CACHING = True
    else:
        print('Keeping new data caching disabled.\nNote that existing cache data will still be used.\n')
    
    # Clear previous cached information
    print("Clearing any cached image data that may have been left over from previous session.")
    paintingtile_fftcache = preface_files(os.getcwd() + "/cache/paintingtiles")
    paintingtile_cache = preface_files(os.getcwd() + "/paintingtiles")
    for cached_info in paintingtile_fftcache + paintingtile_cache:
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
    BACK_CHUNK = len(back_filenames) // 100 + 1
    MIDFRONT_CHUNK = len(midfront_filenames) // 100 + 1

    # Make the sizes of combo chunks relatively consistent
    midfront_chunk_indices = [round(len(midfront_filenames) * i / MIDFRONT_CHUNK) for i in range(MIDFRONT_CHUNK + 1)]
    back_chunk_indices = [round(len(back_filenames) * i / BACK_CHUNK) for i in range(BACK_CHUNK + 1)]

    # Save painting tiles
    for a in range(WIDTH):
        for b in range(HEIGHT):
            painting_crop = painting_im.crop((COMMON_PIXEL_SIZE * a, COMMON_PIXEL_SIZE * b, COMMON_PIXEL_SIZE * (a + 1), COMMON_PIXEL_SIZE * (b + 1)))
            painting_crop.save(str.format("paintingtiles/tile{0}_{1}.png",a,b))
    
    # Check for block combo tiles, otherwise recreate
    # TODO: delete block tiles after processing? (see below)
    if not all(os.path.isfile(os.getcwd() + str.format('/blocktiles/tile{0}_{1}.png', x, y)) for (x, y) in list(itertools.product(range(BACK_CHUNK),range(MIDFRONT_CHUNK)))):
        print("Block combination tiles missing, recreating all.")
        for x in range(BACK_CHUNK):
            for y in range(MIDFRONT_CHUNK):
                print(str.format("Chunking block combination tiles, column {0}/{1}, row {2}/{3}", x+1, BACK_CHUNK, y+1, MIDFRONT_CHUNK))

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
                final.convert('RGB').save(str.format('blocktiles/tile{0}_{1}.png',x,y))

    # Actual main conversion section
    paintified = Image.new('RGB', (COMMON_PIXEL_SIZE * WIDTH, COMMON_PIXEL_SIZE * HEIGHT))
    paintified_backing = Image.new('RGB', (COMMON_PIXEL_SIZE * WIDTH, COMMON_PIXEL_SIZE * HEIGHT))
    paintified_overlay = Image.new('RGBA', (COMMON_PIXEL_SIZE * WIDTH, COMMON_PIXEL_SIZE * HEIGHT))
    for a in range(WIDTH):
        for b in range(HEIGHT):
            print(str.format("Constructing painting block, column {0}/{1}, row {2}/{3} ({4}/{5} total)", a+1, WIDTH, b+1, HEIGHT, a * HEIGHT + b + 1, WIDTH * HEIGHT))

            comparison_file = str.format("paintingtiles/tile{0}_{1}.png",a,b)
            
            # Find the closest block combo
            blockdists = np.zeros((len(midfront_filenames), len(back_filenames)))
            for x in range(BACK_CHUNK):
                for y in range(MIDFRONT_CHUNK):
                    print(str.format("Searching for block candidates ({0} / {1})", x * MIDFRONT_CHUNK + y + 1, BACK_CHUNK * MIDFRONT_CHUNK))
                    blocktile_file = str.format('blocktiles/tile{0}_{1}.png',x,y)
                    # Get distance between painting tile and combination chunk by offset
                    distances_rgb = image_correlator.distance_by_offset(blocktile_file, comparison_file, cache=CACHING).real
                    # Since distance is on a per-pixel basis, while blocks are COMMON_PIXEL_SIZE pixels,
                    # we scale down by that factor to find distances on a per-block basis
                    blockdists_tile = distances_rgb[::COMMON_PIXEL_SIZE,::COMMON_PIXEL_SIZE]
                    # Also, since distance is on a per-color basis, we combine distances over colors through RMS
                    # to get a total block distance. The method can be changed as needed based on results
                    blockdists_tile_rms = np.apply_along_axis(rms, 2, blockdists_tile)
                    # Add distances for combination chunk to overall distance array
                    blockdists[midfront_chunk_indices[y]:midfront_chunk_indices[y+1], back_chunk_indices[x]:back_chunk_indices[x+1]] = blockdists_tile_rms
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
            print(str.format("\nBlock combination found. Back: {0}, front: {1}", final_back_filename, final_midfront_filename))
            with open(palette_path, "a") as f:
                f.write(str.format("Column {0}, row {1}\nBack: {2}\nFront: {3}\n\n", a, b, final_back_filename, final_midfront_filename))
            
            # Clear painting tile FFT cache info
            print("Clearing some cache files to free up space.")
            paintingtile_fftcache = preface_files(os.getcwd() + "/cache/paintingtiles")
            for cache_file in paintingtile_fftcache:
                if not ".gitignore" in cache_file:
                    os.remove(cache_file)
            print("Cleared.\n")
    
    # Process final block image for whole painting
    print(str.format("Image processed, saving output to {0}/", output_folder))
    paintified.save(str.format("{0}/output.png", output_folder))
    paintified_backing.save(str.format("{0}/output_backing.png", output_folder))
    paintified_overlay.save(str.format("{0}/output_overlay.png", output_folder))

    # Clear unnecessary saved information
    # TODO: delete blocktiles and FFT caches by default after processing?
    print("Doing a final clean on some cached data.")
    os.remove(str.format("{0}/progress.png", output_folder))
    paintingtile_cache = preface_files(os.getcwd() + "/paintingtiles")
    for cache_file in paintingtile_cache:
        if not ".gitignore" in cache_file:
            os.remove(cache_file)
    print("Completed.\n")