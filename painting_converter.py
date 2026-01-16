from PIL import Image
import numpy as np
from tqdm import tqdm
from tkinter.filedialog import askopenfilename
import configparser, itertools, json, math, os
import image_correlator
from multiprocessing import Pool

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

# Block files
## Front blocks
FRONT = f"{os.getcwd()}/textures/front"
FRONT_SIDE = f"{os.getcwd()}/textures/front-side"
## Middle blocks
MID = f"{os.getcwd()}/textures/mid"
MID_SIDE = f"{os.getcwd()}/textures/mid-side"
## Back blocks
BACK = f"{os.getcwd()}/textures/back"
BACK_SIDE = f"{os.getcwd()}/textures/back-side"

def get_files_for_tile(a, b, use_tqdm):
    parser = configparser.ConfigParser()
    parser.read('userpref.ini')
    runtime_config = parser['RUNTIME']
    WIDTH = int(runtime_config['WIDTH'])
    HEIGHT = int(runtime_config['HEIGHT'])
    LAYERS = int(runtime_config['LAYERS'])
    USE_CACHING = bool(runtime_config['USE_CACHING'])
    THIRD_SQ_SIZE = int(runtime_config['THIRD_SQ_SIZE'])
    BACK_CHUNK = int(runtime_config['BACK_CHUNK'])
    MIDFRONT_CHUNK = int(runtime_config['MIDFRONT_CHUNK'])
    BACK_CHUNK_INDICES = json.loads(runtime_config['BACK_CHUNK_INDICES'])
    MIDFRONT_CHUNK_INDICES = json.loads(runtime_config['MIDFRONT_CHUNK_INDICES'])
    CANDIDATES = int(parser['DEFAULT']['CANDIDATES'])

    FRONT_FILENAMES = preface_files(FRONT) + preface_files(FRONT_SIDE)
    MID_FILENAMES = preface_files(MID) + preface_files(MID_SIDE)
    BACK_FILENAMES = preface_files(BACK) + preface_files(BACK_SIDE)
    MIDFRONT_FILENAMES = FRONT_FILENAMES+MID_FILENAMES

    print(f"Constructing painting block at column {a+1}/{WIDTH}, row {b+1}/{HEIGHT} (#{a*HEIGHT+b+1} out of {WIDTH*HEIGHT} total)")

    comparison_file = f"paintingtiles/tile_{a}_{b}.png"
    
    # Find the closest block combo
    blockdists = np.zeros((len(MIDFRONT_FILENAMES), len(BACK_FILENAMES)))
    it = itertools.product(range(BACK_CHUNK), range(MIDFRONT_CHUNK))
    if use_tqdm:
        it = tqdm(it, desc="Searching for candidates in combination tiles", total=BACK_CHUNK*MIDFRONT_CHUNK)
    for x, y in it:
        if not use_tqdm:
            print(f"Comparing tile {a*HEIGHT+b+1}/{WIDTH*HEIGHT} to combination file {x*MIDFRONT_CHUNK+y+1}/{BACK_CHUNK*MIDFRONT_CHUNK}")
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
        blockdists[MIDFRONT_CHUNK_INDICES[y]:MIDFRONT_CHUNK_INDICES[y+1], BACK_CHUNK_INDICES[x]:BACK_CHUNK_INDICES[x+1]] = blockdists_tile_rms
    
    # Third layer processing
    if LAYERS == 3:
        # Get the top {CANDIDATES} best two-layer block combos
        blockdists_array = []
        for x, y in itertools.product(range(len(BACK_FILENAMES)), range(len(MIDFRONT_FILENAMES))):
            blockdists_array.append((blockdists[y,x], x, y))
        blockdists_array = sorted(blockdists_array, key=lambda dist: dist[0])[:CANDIDATES]

        # Get all three-layer distances from those two-layer block combos
        thirdlayer_dists = []
        it = blockdists_array
        if use_tqdm:
            it = tqdm(blockdists_array, desc="Searching for third layer candidates")
        else:
            print(f"Searching for third layer candidates for tile {a*HEIGHT+b+1}/{BACK_CHUNK*MIDFRONT_CHUNK}")
        for (_, x, y) in it:
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
            thirdlayer = Image.open(os.getcwd() + "/cache_images/thirdlayer.png")
            combo_thirdlayer.paste(thirdlayer,(0,0),thirdlayer)
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

def process_tile(a, b, progress_image, palette_path, use_tqdm):
    parser = configparser.ConfigParser()
    parser.read('userpref.ini')
    runtime_config = parser['RUNTIME']
    WIDTH = int(runtime_config['WIDTH'])
    HEIGHT = int(runtime_config['HEIGHT'])
    LAYERS = int(runtime_config['LAYERS'])
    N_PROCESSES = int(parser['DEFAULT']['N_PROCESSES'])
    if LAYERS == 3:
        (final_back_file, final_midfront_file, final_front_file) = get_files_for_tile(a,b,use_tqdm)

        # Process final block image for painting tile
        final_im = Image.new('RGB', (COMMON_PIXEL_SIZE, COMMON_PIXEL_SIZE))
        # Final backing block image
        final_back = Image.open(final_back_file).convert('RGBA')
        final_im.paste(final_back,(0,0))
        # Final second block
        final_midfront = Image.open(final_midfront_file).convert('RGBA')
        final_im.paste(final_midfront,(0,0),final_midfront)
        # Final third block
        final_front = Image.open(final_front_file).convert('RGBA')
        final_im.paste(final_front,(0,0),final_front)
        # Save images to progress if not multiprocessing:
        if N_PROCESSES == 1:
            progress_im = Image.open(progress_image)
            progress_im.paste(final_im,(COMMON_PIXEL_SIZE*a,COMMON_PIXEL_SIZE*b))
            progress_im.save(progress_image)
            progress_im.close()

        # Save information to palette
        backstring = f"Back: {get_block_name(final_back_file)} ({final_back_file})"
        midstring = f"Middle: {get_block_name(final_midfront_file)} ({final_midfront_file})"
        frontstring = f"Front: {get_block_name(final_front_file)} ({final_front_file})"
        print(f"Block combination found for tile {a*HEIGHT+b+1}/{WIDTH*HEIGHT}.\n{backstring}\n{midstring}\n{frontstring}")
        if N_PROCESSES == 1:
            print()
        with open(palette_path, "a") as f:
            f.write(f"Column {a+1}, row {b+1}\n{backstring}\n{midstring}\n{frontstring}\n\n")
        
        return((a, b, final_im, final_back, final_midfront, final_front))
    elif LAYERS == 2:
        (final_back_file, final_front_file) = get_files_for_tile(a,b,use_tqdm)

        # Process final block image for painting tile
        final_im = Image.new('RGB', (COMMON_PIXEL_SIZE, COMMON_PIXEL_SIZE))
        # Final backing block image
        final_back = Image.open(final_back_file).convert('RGBA')
        final_im.paste(final_back,(0,0))
        # Final overlay block
        final_front = Image.open(final_front_file).convert('RGBA')
        final_im.paste(final_front,(0,0),final_front)
        # Save images to progress if not multiprocessing
        if(N_PROCESSES == 1):
            progress_im = Image.open(progress_image)
            progress_im.paste(final_im,(COMMON_PIXEL_SIZE*a,COMMON_PIXEL_SIZE*b))
            progress_im.save(progress_image)
            progress_im.close()

        # Save information to palette
        backstring = f"Back: {get_block_name(final_back_file)} ({final_back_file})"
        frontstring = f"Front: {get_block_name(final_front_file)} ({final_front_file})"
        print(f"Block combination found for tile {a*HEIGHT+b+1}/{WIDTH*HEIGHT}.\n{backstring}\n{frontstring}")
        if N_PROCESSES == 1:
            print()
        with open(palette_path, "a") as f:
            f.write(f"Column {a+1}, row {b+1}\n{backstring}\n{frontstring}\n\n")
        
        return ((a, b, final_im, final_back, final_front))

    # Clear cache info
    paintingtile_fftcache = preface_files(f"{os.getcwd()}/cache_fft/paintingtiles")
    thirdlayer_imagecache = preface_files(f"{os.getcwd()}/cache_images/candidates_thirdlayer")
    [os.remove(cache_file) for cache_file in paintingtile_fftcache + thirdlayer_imagecache if f"tile_{a}_{b}" in cache_file and not ".gitignore" in cache_file]

if __name__ == "__main__":
    parser = configparser.ConfigParser()
    parser.read('userpref.ini')
    N_PROCESSES = int(parser['DEFAULT']['N_PROCESSES'])
    runtime_config = {}

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
    runtime_config['WIDTH'] = str(WIDTH)
    print("Please enter a height.")
    HEIGHT = int(input())
    runtime_config['HEIGHT'] = str(HEIGHT)
    painting_im = painting_im.resize((WIDTH*COMMON_PIXEL_SIZE,HEIGHT*COMMON_PIXEL_SIZE))
    painting_im.save(f"{output_folder}/original.png")

    # Progress image
    progress_image = f"{output_folder}/progress.png"
    if N_PROCESSES == 1:
        painting_im.save(progress_image)
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
    runtime_config['LAYERS'] = str(LAYERS)

    # Ask about caching
    print("\nDo you want to cache new Fourier transform coefficient data? This will speed up calculations but also can take up significant amounts of storage (~9GB). Type 'yes' if so.")
    # We want the user to be very explicit about actually wanting this
    if(input().strip().lower()) == 'yes':
        print('Enabling caching of new coefficient data.\n')
        USE_CACHING = True
    else:
        print('Keeping new data caching disabled.\nNote that existing cache data will still be used.\n')
        USE_CACHING = False
    runtime_config['USE_CACHING'] = str(USE_CACHING)
    
    # Clear previous cached information
    print("Clearing any cached data that may have been left over from previous session.")
    clear_folders = ["cache_images/paintingtiles","cache_fft/paintingtiles","cache_images/blocktiles","cache_fft/blocktiles","cache_fft/masks","cache_images/candidates_thirdlayer"]
    [os.remove(cached_info) for cached_info in list(itertools.accumulate([preface_files(f"{os.getcwd()}/{folder}") for folder in clear_folders]))[-1] if not ".gitignore" in cached_info]
    if os.path.isfile(f"{os.getcwd()}/cache_images/thirdlayer.png"):
        os.remove(f"{os.getcwd()}/cache_images/thirdlayer.png")
    print("Cleared.\n")

    FRONT_FILENAMES = preface_files(FRONT) + preface_files(FRONT_SIDE)
    MID_FILENAMES = preface_files(MID) + preface_files(MID_SIDE)
    BACK_FILENAMES = preface_files(BACK) + preface_files(BACK_SIDE)
    MIDFRONT_FILENAMES = FRONT_FILENAMES+MID_FILENAMES

    # Get third layer square size
    THIRD_SQ_SIZE = math.ceil(math.sqrt(len(FRONT_FILENAMES) + 1))
    runtime_config['THIRD_SQ_SIZE'] = str(THIRD_SQ_SIZE)

    # There are too many file combinations for the FFT info to be comfortably stored in memory
    # We use a dynamic number of block-overlay combo chunks based on how many blocks there are
    BACK_CHUNK = math.ceil(len(BACK_FILENAMES) / 100)
    runtime_config['BACK_CHUNK'] = str(BACK_CHUNK)
    MIDFRONT_CHUNK = math.ceil(len(MIDFRONT_FILENAMES) / 100)
    runtime_config['MIDFRONT_CHUNK'] = str(MIDFRONT_CHUNK)

    # Make the sizes of combo chunks relatively consistent
    BACK_CHUNK_INDICES = [round(len(BACK_FILENAMES) * i / BACK_CHUNK) for i in range(BACK_CHUNK + 1)]
    runtime_config['BACK_CHUNK_INDICES'] = json.dumps(BACK_CHUNK_INDICES)
    MIDFRONT_CHUNK_INDICES = [round(len(MIDFRONT_FILENAMES) * i / MIDFRONT_CHUNK) for i in range(MIDFRONT_CHUNK + 1)]
    runtime_config['MIDFRONT_CHUNK_INDICES'] = json.dumps(MIDFRONT_CHUNK_INDICES)
    
    parser['RUNTIME'] = runtime_config
    with open('userpref.ini', 'w') as configfile:
        parser.write(configfile)

    # Save painting tiles
    for a, b in itertools.product(range(WIDTH), range(HEIGHT)):
        painting_crop = painting_im.crop((COMMON_PIXEL_SIZE * a, COMMON_PIXEL_SIZE * b, COMMON_PIXEL_SIZE * (a + 1), COMMON_PIXEL_SIZE * (b + 1)))
        painting_crop.save(f"cache_images/paintingtiles/tile_{a}_{b}.png")
    painting_im.close()
    
    # Check for block combo tiles, otherwise recreate
    if not all(os.path.isfile(f"{os.getcwd()}/cache_images/blocktiles/tile_{x}_{y}.png") for x, y in itertools.product(range(BACK_CHUNK),range(MIDFRONT_CHUNK))):
        for x, y in tqdm(itertools.product(range(BACK_CHUNK), range(MIDFRONT_CHUNK)), desc="Creating block combination tiles", total=BACK_CHUNK*MIDFRONT_CHUNK):
            # Generate row of backing blocks
            back_row = Image.new('RGBA', ((BACK_CHUNK_INDICES[x+1] - BACK_CHUNK_INDICES[x]) * COMMON_PIXEL_SIZE, COMMON_PIXEL_SIZE))
            for i in range(BACK_CHUNK_INDICES[x], BACK_CHUNK_INDICES[x+1]):
                back_row.paste(Image.open(BACK_FILENAMES[i]).convert('RGBA'), (COMMON_PIXEL_SIZE * (i - BACK_CHUNK_INDICES[x]), 0))
            # Copy rows vertically
            backing = Image.new('RGBA', ((BACK_CHUNK_INDICES[x+1] - BACK_CHUNK_INDICES[x]) * COMMON_PIXEL_SIZE, (MIDFRONT_CHUNK_INDICES[y+1] - MIDFRONT_CHUNK_INDICES[y]) * COMMON_PIXEL_SIZE))
            for j in range(MIDFRONT_CHUNK_INDICES[y], MIDFRONT_CHUNK_INDICES[y+1]):
                backing.paste(back_row, (0, COMMON_PIXEL_SIZE * (j - MIDFRONT_CHUNK_INDICES[y])))
            back_row.close()

            # Generate column of overlay blocks
            overlay_col = Image.new('RGBA', (COMMON_PIXEL_SIZE, (MIDFRONT_CHUNK_INDICES[y+1] - MIDFRONT_CHUNK_INDICES[y]) * COMMON_PIXEL_SIZE))
            for j in range(MIDFRONT_CHUNK_INDICES[y], MIDFRONT_CHUNK_INDICES[y+1]):
                overlay_j = Image.open(MIDFRONT_FILENAMES[j]).convert('RGBA')
                overlay_col.paste(overlay_j, (0,COMMON_PIXEL_SIZE*(j - MIDFRONT_CHUNK_INDICES[y])))
                overlay_j.close()
            # Copy columns horizonally
            overlay = Image.new('RGBA', ((BACK_CHUNK_INDICES[x+1] - BACK_CHUNK_INDICES[x]) * COMMON_PIXEL_SIZE, (MIDFRONT_CHUNK_INDICES[y+1] - MIDFRONT_CHUNK_INDICES[y]) * COMMON_PIXEL_SIZE))
            for i in range(BACK_CHUNK_INDICES[x], BACK_CHUNK_INDICES[x+1]):
                overlay.paste(overlay_col, (COMMON_PIXEL_SIZE*(i - BACK_CHUNK_INDICES[x]), 0))
            overlay_col.close()

            # Paste overlay tiles over backing tiles to get combination tiles
            backing.paste(overlay, (0, 0), overlay)
            overlay.close()
            backing.convert('RGB').save(f"cache_images/blocktiles/tile_{x}_{y}.png")
            backing.close()
    print()

    thirdlayer = Image.new('RGBA', (THIRD_SQ_SIZE * COMMON_PIXEL_SIZE, THIRD_SQ_SIZE * COMMON_PIXEL_SIZE))
    for i in range(len(FRONT_FILENAMES)):
        front_i = Image.open(FRONT_FILENAMES[i]).convert('RGBA')
        thirdlayer.paste(front_i, ((i // THIRD_SQ_SIZE) * COMMON_PIXEL_SIZE, (i % THIRD_SQ_SIZE) * COMMON_PIXEL_SIZE))
        front_i.close()
    thirdlayer.save(f"cache_images/thirdlayer.png")

    # Actual main conversion section
    paintified = Image.new('RGB', (COMMON_PIXEL_SIZE * WIDTH, COMMON_PIXEL_SIZE * HEIGHT))
    paintified_backing = Image.new('RGB', (COMMON_PIXEL_SIZE * WIDTH, COMMON_PIXEL_SIZE * HEIGHT))
    paintified_overlay = Image.new('RGBA', (COMMON_PIXEL_SIZE * WIDTH, COMMON_PIXEL_SIZE * HEIGHT))
    paintified_overlay2 = Image.new('RGBA', (COMMON_PIXEL_SIZE * WIDTH, COMMON_PIXEL_SIZE * HEIGHT))

    if N_PROCESSES > 1:
        print("Performing an initial iteration to cache FFT data")
    # Run an initial iteration to get cache files working
    alltile00 = process_tile(0, 0, progress_image, palette_path, True)
    alltiles = []
    # Then multiprocess
    if N_PROCESSES > 1:
        with Pool(N_PROCESSES) as p:
            alltiles = p.starmap(process_tile, [(a, b, progress_image, palette_path, False) for a, b in itertools.product(range(WIDTH), range(HEIGHT)) if a > 0 or b > 0])
    else:
        alltiles = [process_tile(a, b, progress_image, palette_path, True) for a, b in itertools.product(range(WIDTH), range(HEIGHT)) if a > 0 or b > 0]
    alltiles = [alltile00] + alltiles

    # Combine all info into block image files
    for final_info in alltiles:
        paintified.paste(final_info[2],(COMMON_PIXEL_SIZE*final_info[0],COMMON_PIXEL_SIZE*final_info[1]))
        final_info[2].close()
        paintified_backing.paste(final_info[3],(COMMON_PIXEL_SIZE*final_info[0],COMMON_PIXEL_SIZE*final_info[1]))
        final_info[3].close()
        paintified_overlay.paste(final_info[4],(COMMON_PIXEL_SIZE*final_info[0],COMMON_PIXEL_SIZE*final_info[1]),final_info[4])
        final_info[4].close()
        if LAYERS == 3:
            paintified_overlay2.paste(final_info[5],(COMMON_PIXEL_SIZE*final_info[0],COMMON_PIXEL_SIZE*final_info[1]),final_info[5])
            final_info[5].close()

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
    os.remove(f"{os.getcwd()}/cache_images/thirdlayer.png")
    parser['RUNTIME'] = {}
    with open('userpref.ini', 'w') as configfile:
        parser.write(configfile)
    print(f"Completed. Go to {output_folder}/ for the output.\n")