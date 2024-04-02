import cv2
import os
import numpy as np
from skimage import io, img_as_uint
import tifffile
from natsort import natsorted
import time
import snappy
from multiprocessing import Pool, cpu_count
import logging

# Setup logging
logging.basicConfig(filename='gzip_comp.log', level=logging.INFO, format='%(asctime)s: %(message)s')

def load(folder_path):
    image_paths = natsorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.tif', '.tiff'))])
    images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths if cv2.imread(path, cv2.IMREAD_GRAYSCALE) is not None]
    return np.array(images), image_paths

def compress_file(input_file_path):
    with open(input_file_path, "rb") as file:
        tiff_data = file.read()
    compressed_data = snappy.compress(tiff_data)
    output_file_path = input_file_path + ".snappy"
    with open(output_file_path, "wb") as output_file:
        output_file.write(compressed_data)
    #logging.info(f"Compressed TIFF file: {output_file_path}")
    return output_file_path

def compute_diff(images, reference_image_path, output_folder_diff):
    ref_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    if ref_image is None:
        logging.error("Reference image could not be loaded.")
        return
    for i, image in enumerate(images):
        difference = cv2.absdiff(image, ref_image)
        output_path = os.path.join(output_folder_diff, f"diff_{i:04d}.tif")
        tifffile.imwrite(output_path, img_as_uint(difference))
        logging.info(f"Difference image saved: {output_path}")

def gen_sinograms(images, output_folder):
    if not images.size:
        logging.error("No images to generate sinograms from.")
        return
    sinograms = np.transpose(images, axes=(1, 0, 2))
    for i, sinogram in enumerate(sinograms):
        output_path = os.path.join(output_folder, f"combined_row_{i}.tif")
        tifffile.imwrite(output_path, img_as_uint(sinogram))
        logging.info(f"Sinogram saved: {output_path}")

def main():
    start_time = time.time()

    folder_path = '/home/amarjitsingh/imgrec/XCT4K_tiff_0_0_8'
    reference_image_path = '/home/amarjitsingh/imgrec/q0001.tiff'
    output_folder_diff = '/home/amarjitsingh/imgrec/p1/diff1/'
    output_folder_sinogram = '/home/amarjitsingh/imgrec/p1/sub'

    os.makedirs(output_folder_diff, exist_ok=True)
    os.makedirs(output_folder_sinogram, exist_ok=True)

    images, image_paths = load(folder_path)

    # Computing and saving differences
    compute_diff(images, reference_image_path, output_folder_diff)

    # Loading difference images
    diff_images, _ = load(output_folder_diff)

    # Generating and saving sinograms
    gen_sinograms(diff_images, output_folder_sinogram)

    # Compressing sinogram images
    tiff_files_to_compress = [os.path.join(output_folder_sinogram, f) for f in os.listdir(output_folder_sinogram) if f.lower().endswith((".tif", ".tiff"))]

    with Pool(processes=cpu_count()) as pool:
        pool.map(compress_file, tiff_files_to_compress)

    logging.info(f"Total process time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
