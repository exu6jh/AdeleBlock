from PIL import Image
import numpy as np
import pyfftw
import os

def normalize_arr_range(arr, min, max):
    return (arr.real - min)/(max - min)

def prettify_arr_fullrange(arr):
    return 255 * normalize_arr_range(arr, arr.real.min(), arr.real.max())

# Produce correlations between main image and pattern shifted by position
def correlate(main_im_file, pattern_im_file, cache=True):
    im_main = Image.open(os.getcwd() + "/" + main_im_file).convert("RGB")
    im_array = np.asarray(im_main) / 255

    im_pattern = Image.open(os.getcwd() + "/" + pattern_im_file).convert("RGB")
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

    correlations_rgb = np.zeros(im_array.shape, dtype='complex128')
    f2 = []

    for color in range(3):
        im_filename = str.format('cache/{0}_imfft_{1}.npy', main_im_file, color)
        im_fft = np.zeros(a.shape,dtype=np.complex_)
        if os.path.isfile(os.getcwd() + '/' + im_filename):
            im_fft = np.load(im_filename)
        else:
            a[:] = im_array[:,:,color]
            fft_forward = fft_obj()
            im_fft[:] = b[:]
            if cache:
                np.save(im_filename, im_fft)

        a[:] = pattern_array[:,:,color]
        f2.append(np.sum(a*a)/np.sqrt(a.size))
        fft_forward = fft_obj()
        ptrn_fft = np.zeros(a.shape,dtype=np.complex_)
        ptrn_fft[:] = b[:]

        main_ptrn_corr_fft = np.zeros(a.shape,dtype=np.complex_)
        main_ptrn_corr_fft[:] = ptrn_fft[:].conjugate() * im_fft[:]

        im_sq_filename = str.format('cache/{0}_imsqfft_{1}.npy', main_im_file, color)
        im_sq_fft = np.zeros(a.shape,dtype=np.complex_)
        if os.path.isfile(os.getcwd() + '/' + im_sq_filename):
            im_sq_fft = np.load(im_sq_filename)
        else:
            a[:] = im_squared_array[:,:,color]
            fft_forward = fft_obj()
            im_sq_fft[:] = b[:]
            if cache:
                np.save(im_sq_filename, im_sq_fft)

        mask_filename = str.format('cache/{0}_mask_{1}_{2}.npy', main_im_file, im_pattern_arr.shape[0], im_pattern_arr.shape[1])
        mask_fft = np.zeros(a.shape,dtype=np.complex_)
        if os.path.isfile(os.getcwd() + '/' + mask_filename):
            mask_fft = np.load(mask_filename)
        else:
            a[:] = mask[:,:,color]
            fft_forward = fft_obj()
            mask_fft[:] = b[:]
            if cache:
                np.save(mask_filename, mask_fft)

        im_sq_mask_corr_fft = np.zeros(a.shape,dtype=np.complex_)
        im_sq_mask_corr_fft[:] = mask_fft[:].conjugate() * im_sq_fft[:]

        b[:] = main_ptrn_corr_fft[:]
        fft_backward = ifft_obj()
        main_ptrn_corr = np.zeros(a.shape,dtype=np.complex_)
        main_ptrn_corr[:] = fft_backward[:]

        b[:] = im_sq_mask_corr_fft[:]
        fft_backward = ifft_obj()
        im_sq_mask_corr = np.zeros(a.shape,dtype=np.complex_)
        im_sq_mask_corr[:] = fft_backward[:]

        # Should go from 0 to im_pattern_arr[:,:,0].size/np.sqrt(a.size)
        normdiff = f2[color] - 2*main_ptrn_corr + im_sq_mask_corr
        correlations_rgb[:,:,color] = normdiff

    # Normalize to min and max possible
    correlations_rgb = normalize_arr_range(correlations_rgb, 0, im_pattern_arr[:,:,0].size/np.sqrt(a.size))

    return correlations_rgb