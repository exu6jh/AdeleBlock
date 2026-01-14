from PIL import Image
import numpy as np
import pyfftw
import os

# Produce distances between main image and pattern shifted by position
def distance_by_offset(main_im_file, pattern_im_file, cache=True):
    im_main = Image.open(os.getcwd() + "/cache_images/" + main_im_file).convert("RGB")
    im_array = np.asarray(im_main) / 255

    im_pattern = Image.open(os.getcwd() + "/cache_images/" + pattern_im_file).convert("RGB")
    im_pattern_arr = np.asarray(im_pattern)
    padshape = [(0, im_array.shape[0] - im_pattern_arr.shape[0]), (0, im_array.shape[1] - im_pattern_arr.shape[1]), (0,0)]
    pattern_array = np.pad(im_pattern_arr, padshape) / 255

    mask = np.pad(np.ones(im_pattern_arr.shape[:2]), [(0, im_array.shape[0] - im_pattern_arr.shape[0]), (0, im_array.shape[1] - im_pattern_arr.shape[1])])
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    im_squared_array = im_array * im_array

    a = pyfftw.empty_aligned(im_array.shape[:2], dtype='complex128')
    b = pyfftw.empty_aligned(im_array.shape[:2], dtype='complex128')
    c = pyfftw.empty_aligned(im_array.shape[:2], dtype='complex128')

    fft_obj = pyfftw.FFTW(a, b, ortho=True, normalise_idft=False, axes=(0,1))
    ifft_obj = pyfftw.FFTW(b, c, ortho=True, normalise_idft=False, axes=(0,1), direction='FFTW_BACKWARD')

    distances_rgb = np.zeros(im_array.shape, dtype='complex128')
    f2 = []

    for color in range(3):
        f2_color_arr = pattern_array[:,:,color]
        f2.append(np.sum(f2_color_arr*f2_color_arr)/np.sqrt(f2_color_arr.size))

        im_filename = str.format('cache_fft/{0}_color_{1}_imfft.npy', main_im_file, color)
        im_fft = np.zeros(a.shape,dtype=np.complex128)
        if os.path.isfile(os.getcwd() + '/' + im_filename):
            im_fft = np.load(im_filename)
        else:
            a[:] = im_array[:,:,color]
            fft_forward = fft_obj()
            im_fft[:] = b[:]
            if cache:
                np.save(im_filename, im_fft)

        pattern_filename = str.format('cache_fft/{0}_color_{1}_size_{2}_{3}.npy', pattern_im_file, color, a.shape[0], a.shape[1])
        ptrn_fft = np.zeros(a.shape,dtype=np.complex128)
        if os.path.isfile(os.getcwd() + '/' + pattern_filename):
            ptrn_fft = np.load(pattern_filename)
        else:
            a[:] = pattern_array[:,:,color]
            fft_forward = fft_obj()
            ptrn_fft[:] = b[:]
            if cache:
                np.save(pattern_filename, ptrn_fft)

        main_ptrn_corr_fft = np.zeros(a.shape,dtype=np.complex128)
        main_ptrn_corr_fft[:] = ptrn_fft[:].conjugate() * im_fft[:]

        im_sq_filename = str.format('cache_fft/{0}_color_{1}_imsqfft.npy', main_im_file, color)
        im_sq_fft = np.zeros(a.shape,dtype=np.complex128)
        if os.path.isfile(os.getcwd() + '/' + im_sq_filename):
            im_sq_fft = np.load(im_sq_filename)
        else:
            a[:] = im_squared_array[:,:,color]
            fft_forward = fft_obj()
            im_sq_fft[:] = b[:]
            if cache:
                np.save(im_sq_filename, im_sq_fft)

        mask_filename = str.format('cache_fft/masks/size_{0}_{1}_mask_{2}_{3}.npy', a.shape[0], a.shape[1], im_pattern_arr.shape[0], im_pattern_arr.shape[1])
        mask_fft = np.zeros(a.shape,dtype=np.complex128)
        if os.path.isfile(os.getcwd() + '/' + mask_filename):
            mask_fft = np.load(mask_filename)
        else:
            a[:] = mask[:,:,color]
            fft_forward = fft_obj()
            mask_fft[:] = b[:]
            if cache:
                np.save(mask_filename, mask_fft)

        im_sq_mask_corr_fft = np.zeros(a.shape,dtype=np.complex128)
        im_sq_mask_corr_fft[:] = mask_fft[:].conjugate() * im_sq_fft[:]

        b[:] = 2 * main_ptrn_corr_fft[:] - im_sq_mask_corr_fft[:]
        fft_backward = ifft_obj()
        normdiff_corr = np.zeros(a.shape,dtype=np.complex128)
        normdiff_corr[:] = fft_backward[:]

        normdiff = f2[color] - normdiff_corr
        distances_rgb[:,:,color] = normdiff

    return distances_rgb * np.sqrt(a.size)