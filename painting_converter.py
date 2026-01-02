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
BACK_CHUNK = 5
MIDFRONT_CHUNK = 6

WIDTH = -1
HEIGHT = -1

CACHING = False

if __name__ == "__main__":
    print("Please select a painting you would like blockified.")
    painting_filename = askopenfilename()
    print(str.format("Selected {0}", painting_filename))
    painting_im = Image.open(painting_filename).convert('RGBA')
    output_folder = str.format(os.getcwd() + "/outputs/{0}", os.path.basename(painting_filename))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    palette_path = output_folder + "/palette.txt"
    if not os.path.isfile(palette_path):
        open(palette_path, "x")
        
    print("Please enter a width.")
    WIDTH = int(input())
    print("Please enter a height.")
    HEIGHT = int(input())
    painting_im = painting_im.resize((WIDTH*COMMON_PIXEL_SIZE,HEIGHT*COMMON_PIXEL_SIZE))
    painting_im.save(str.format('{0}/resized.png', output_folder))

    print("Do you want to cache Fourier transform coefficientt data? This will speed up calculations but also takes up significant amounts of storage (~9GB). Type 'yes' if so.")
    if(input().strip().lower()) == 'yes':
        print('Enabling caching.')
        CACHING = True
    else:
        print('Keeping caching disabled.')
    
    print("Clearing any cached image data that may have been left over from previous session.")
    paintingtile_fftcache = preface_files(os.getcwd() + "/cache/paintingtiles")
    paintingtile_cache = preface_files(os.getcwd() + "/paintingtiles")
    candidates_cache = preface_files(os.getcwd() + "/candidates")
    for cached_info in paintingtile_fftcache + paintingtile_cache + candidates_cache:
        if not ".gitignore" in cached_info:
            os.remove(cached_info)
    print("Cleared.\n")

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

    front_chunk_indices = [round(len(midfront_filenames) * i / MIDFRONT_CHUNK) for i in range(MIDFRONT_CHUNK + 1)]
    back_chunk_indices = [round(len(back_filenames) * i / BACK_CHUNK) for i in range(BACK_CHUNK + 1)]

    for a in range(WIDTH):
        for b in range(HEIGHT):
            painting_crop = painting_im.crop((COMMON_PIXEL_SIZE * a, COMMON_PIXEL_SIZE * b, COMMON_PIXEL_SIZE * (a + 1), COMMON_PIXEL_SIZE * (b + 1)))
            painting_crop.save(str.format("paintingtiles/tile{0}_{1}.png",a,b))
    
    if not all(os.path.isfile(os.getcwd() + str.format('/blocktiles/tile{0}_{1}.png', x, y)) for (x, y) in list(itertools.product(range(BACK_CHUNK),range(MIDFRONT_CHUNK)))):
        print("Block combination tiles missing, recreating all.")
        for x in range(BACK_CHUNK):
            for y in range(MIDFRONT_CHUNK):
                print(str.format("Chunking block combination tiles, column {0}/{1}, row {2}/{3}", x+1, BACK_CHUNK, y+1, MIDFRONT_CHUNK))
                back_row = Image.new('RGBA', ((back_chunk_indices[x+1] - back_chunk_indices[x]) * COMMON_PIXEL_SIZE, COMMON_PIXEL_SIZE))
                for i in range(back_chunk_indices[x], back_chunk_indices[x+1]):
                    back_row.paste(Image.open(back_filenames[i]).convert('RGBA'), (COMMON_PIXEL_SIZE * (i - back_chunk_indices[x]), 0))
                backing = Image.new('RGBA', ((back_chunk_indices[x+1] - back_chunk_indices[x]) * COMMON_PIXEL_SIZE, (front_chunk_indices[y+1] - front_chunk_indices[y]) * COMMON_PIXEL_SIZE))
                for j in range(front_chunk_indices[y], front_chunk_indices[y+1]):
                    backing.paste(back_row, (0, COMMON_PIXEL_SIZE * (j - front_chunk_indices[y])))

                overlay_col = Image.new('RGBA', (COMMON_PIXEL_SIZE, (front_chunk_indices[y+1] - front_chunk_indices[y]) * COMMON_PIXEL_SIZE))
                for j in range(front_chunk_indices[y], front_chunk_indices[y+1]):
                    overlay_col.paste(Image.open(midfront_filenames[j]).convert('RGBA'), (0,COMMON_PIXEL_SIZE*(j - front_chunk_indices[y])))
                overlay = Image.new('RGBA', ((back_chunk_indices[x+1] - back_chunk_indices[x]) * COMMON_PIXEL_SIZE, (front_chunk_indices[y+1] - front_chunk_indices[y]) * COMMON_PIXEL_SIZE))
                for i in range(back_chunk_indices[x], back_chunk_indices[x+1]):
                    overlay.paste(overlay_col, (COMMON_PIXEL_SIZE*(i - back_chunk_indices[x]), 0))

                final = backing.copy()
                final.paste(overlay, (0, 0), overlay)
                final.convert('RGB').save(str.format('blocktiles/tile{0}_{1}.png',x,y))

    paintified = Image.new('RGB', (COMMON_PIXEL_SIZE * WIDTH, COMMON_PIXEL_SIZE * HEIGHT))
    for a in range(WIDTH):
        for b in range(HEIGHT):
            print(str.format("Constructing painting block, column {0}/{1}, row {2}/{3} ({4}/{5} total)", a+1, WIDTH, b+1, HEIGHT, a * HEIGHT + b + 1, WIDTH * HEIGHT))
            candidates = []
            candidates_im = Image.new('RGB', (COMMON_PIXEL_SIZE * BACK_CHUNK, COMMON_PIXEL_SIZE * MIDFRONT_CHUNK))
            comparison_file = str.format("paintingtiles/tile{0}_{1}.png",a,b)
            
            for x in range(BACK_CHUNK):
                for y in range(MIDFRONT_CHUNK):
                    print(str.format("Searching for block candidates ({0} / {1})", x * MIDFRONT_CHUNK + y + 1, BACK_CHUNK * MIDFRONT_CHUNK))
                    blocktile_file = str.format('blocktiles/tile{0}_{1}.png',x,y)
                    # Default value, just in case
                    correlations_rgb = image_correlator.correlate(blocktile_file, comparison_file, cache=CACHING)
                    blockcorrs_tile = correlations_rgb[::COMMON_PIXEL_SIZE,::COMMON_PIXEL_SIZE]
                    blockcorrs_tile_rms = np.apply_along_axis(rms, 2, blockcorrs_tile)
                    tile_candidate = np.unravel_index(np.argmin(blockcorrs_tile_rms), blockcorrs_tile_rms.shape)
                    candidates.append((front_chunk_indices[y] + tile_candidate[0], back_chunk_indices[x] + tile_candidate[1]))
            
            for i in range(BACK_CHUNK * MIDFRONT_CHUNK):
                (x, y) = (i // MIDFRONT_CHUNK, i % MIDFRONT_CHUNK)
                (candidate_y, candidate_x) = candidates[i]
                candidate_back = Image.open(back_filenames[candidate_x]).convert('RGBA')
                candidates_im.paste(candidate_back,(COMMON_PIXEL_SIZE*x,COMMON_PIXEL_SIZE*y))
                candidate_front = Image.open(midfront_filenames[candidate_y]).convert('RGBA')
                candidates_im.paste(candidate_front,(COMMON_PIXEL_SIZE*x,COMMON_PIXEL_SIZE*y),candidate_front)
            candidates_filename = str.format("candidates/candidate{0}_{1}.png",a,b)
            candidates_im.save(candidates_filename)
            candidates_corr = image_correlator.correlate(candidates_filename, comparison_file, cache=False)
            candidates_blockcorr_rms = np.apply_along_axis(rms, 2, candidates_corr[::COMMON_PIXEL_SIZE,::COMMON_PIXEL_SIZE])
            final_candidate_index = np.unravel_index(np.argmin(candidates_blockcorr_rms), candidates_blockcorr_rms.shape)
            final_block_index = candidates[final_candidate_index[1]*MIDFRONT_CHUNK + final_candidate_index[0]]

            final_im = Image.new('RGB', (COMMON_PIXEL_SIZE, COMMON_PIXEL_SIZE))
            final_back_filename = back_filenames[final_block_index[1]]
            final_back = Image.open(final_back_filename).convert('RGBA')
            final_im.paste(final_back,(0,0))
            final_front_filename = midfront_filenames[final_block_index[0]]
            final_front = Image.open(final_front_filename).convert('RGBA')
            final_im.paste(final_front,(0,0),final_front)
            paintified.paste(final_im,(COMMON_PIXEL_SIZE*a,COMMON_PIXEL_SIZE*b))
            paintified.save(str.format("{0}/progress.png", output_folder))
            print(str.format("Block combination found. Back: {0}, front: {1}", final_back_filename, final_front_filename))
            with open(palette_path, "a") as f:
                f.write(str.format("Column {0}, row {1}\nBack: {2}\nFront: {3}\n\n", a, b, final_back_filename, final_front_filename))
            print("Clearing some cache files to free up space.")
            paintingtile_fftcache = preface_files(os.getcwd() + "/cache/paintingtiles")
            for cache_file in paintingtile_fftcache:
                if not ".gitignore" in cache_file:
                    os.remove(cache_file)
            print("Cleared.\n")
    paintified.save(str.format("{0}/output.png", output_folder))