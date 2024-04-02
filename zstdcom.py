import cv2
import os
import numpy as np
from skimage import io, img_as_uint
import tifffile
from natsort import natsorted
import time
import zstandard as zstd
from multiprocessing import Pool, cpu_count
import logging

# Setup logging
logging.basicConfig(filename='zstd_output.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load(folder_path):
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.tif', '.tiff'))]
    images = np.array([cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths])
    return images, image_paths

def compress_tiff(input_file_path):
    cctx = zstd.ZstdCompressor(level=22)
    with open(input_file_path, "rb") as tiff_input_file:
        tiff_data = tiff_input_file.read()
    compressed_data = cctx.compress(tiff_data)
    compressed_file_path = input_file_path + ".zst"
    with open(compressed_file_path, "wb") as compressed_output_file:
        compressed_output_file.write(compressed_data)
    logging.info(f"Compressed and saved: {compressed_file_path}")
    return compressed_file_path

def compute_and_save_differences(images, reference_image_path, output_folder_diff):
    ref_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)  
    if ref_image is None:
        logging.error("Reference image could not be loaded.")
        return
    for i, image in enumerate(images):
        difference = cv2.absdiff(image, ref_image)
        output_path = os.path.join(output_folder_diff, f"diff_{i:04d}.tif")
        difference_16bit = img_as_uint(difference)
        tifffile.imwrite(output_path, difference_16bit)
        logging.info(f"Difference image saved: {output_path}")


def gen_sinograms(images, output_folder):
    print("Shape of images array before transposing:", images.shape)
    sinograms = np.transpose(images, axes=(1, 0, 2))  
    for i, sinogram in enumerate(sinograms):
        output_path = os.path.join(output_folder, f"combined_row_{i}.tif")
        tifffile.imwrite(output_path, img_as_uint(sinogram)) 

def main():
    start_time = time.time()
    folder_path = "/home/amarjitsingh/imgrec/XCT4K_tiff_0_0_8"
    reference_image_path = "/home/amarjitsingh/imgrec/xct/q0001.tiff"
    output_folder_diff = '/home/amarjitsingh/imgrec/xct/p1/diff1/'
    output_folder_sinogram = '/home/amarjitsingh/imgrec/xct/p1/sub'
    #output_zstd_file = "/home/amarjitsingh/imgrec/p1/all.zst"

    os.makedirs(output_folder_diff, exist_ok=True)
    os.makedirs(output_folder_sinogram, exist_ok=True)

    load_time_start = time.time()
    images, image_paths = load(folder_path)
    load_time_end = time.time()
    #logging.info(f"Load times {load_time_start - load_time_end :.2f} seconds")

    diff_time_start = time.time()
    compute_and_save_differences(images, reference_image_path, output_folder_diff)
    diff_time_end = time.time()
    logging.info(f"compute difference time is{diff_time_end - diff_time_start:.2f} seconds")

    sino_load_start_time = time.time()
    diff_images, _ = load(output_folder_diff)
    sino_load_end_time = time.time()
    logging.info(f"sino load time is {sino_load_end_time - sino_load_start_time:.2f} seconds")

    sino_start_time = time.time()
    gen_sinograms(diff_images, output_folder_sinogram)
    sino_end_time = time.time()
    logging.info(f"generate sinogram time is {sino_end_time - sino_start_time:.2f} seconds")

    num_cores = cpu_count()
    with Pool(processes=num_cores) as pool:
        pool.map(compress_tiff, image_paths)

    #logging.info(f"All TIFF images in '{folder_path}' compressed to '{output_zstd_file}' with compression level 22")
    logging.info(f"Total process time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
