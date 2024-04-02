import cv2
import os
import numpy as np
from skimage import io, img_as_uint
from natsort import natsorted
import tifffile
import gzip
import glob
import time
from multiprocessing import Pool, cpu_count
import logging

# Setup log > decompression_output.log for decompression
logging.basicConfig(filename='gzip_de.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def decompress_gzip_file(file_info):
    input_file_path, output_directory = file_info
    output_file_name = os.path.basename(input_file_path).replace('.gz', '.tif')
    output_file_path = os.path.join(output_directory, output_file_name)
    # Use gzip.open to read and automatically decompress the .gz file
    with gzip.open(input_file_path, 'rb') as compressed_file, \
         open(output_file_path, 'wb') as decompressed_file:
        # This will copy the decompressed content directly to the output file
        decompressed_file.writelines(compressed_file)
    #print(f"Decompressed {input_file_path} to {output_file_path}")

def generate_sinograms(image_folder, sinogram_folder):
    """
    Generate sinograms from a stack of images and save them to a sinogram folder, maintaining their original bit depth.
    """
    if not os.path.exists(sinogram_folder):
        os.makedirs(sinogram_folder)

    # Load images while preserving their original bit depth
    image_paths = natsorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.tif')])
    images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    # Stack images into a 3D array for efficient processing
    #images_stack = np.stack(images, axis=0)
    
    # Generate sinograms by transposing the axes
    sinograms = np.transpose(images, axes=(1, 0, 2))

    # Save sinograms without converting to a specific bit depth
    for i, sinogram in enumerate(sinograms):
        output_path = os.path.join(sinogram_folder, f"sinogram_{i:04d}.tif")
        tifffile.imwrite(output_path, img_as_uint(sinogram))

    #logging.info("Sinograms generated and saved, maintaining the original bit depth of the images.")

def compute_diff(images, reference_image_path, output_folder_diff):

    ref_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    if ref_image is None:
        logging.error("Reference image could not be loaded.")
        return
    for i, image in enumerate(images):
        if image is None:
            logging.warning(f"Skipping image {i} due to loading failure.")
            continue
        difference = cv2.absdiff(image, ref_image)
        output_path = os.path.join(output_folder_diff, f"diff_{i:04d}.tif")
        
        tifffile.imwrite(output_path, img_as_uint(difference))
        #logging.info(f"Difference image saved: {output_path}")


def main():
    start_time = time.time()

    compressed_directory = '/home/amarjitsingh/imgrec/p1/sub'  
    output_directory_decompressed = '/home/amarjitsingh/imgrec/decompressed'
    output_folder_sinogram = '/home/amarjitsingh/imgrec/sinograms'
    output_folder_diff = '/home/amarjitsingh/imgrec/diff'
    reference_image_path = "/home/amarjitsingh/imgrec/q001.tif" 

    # Create directories
    os.makedirs(output_directory_decompressed, exist_ok=True)
    os.makedirs(output_folder_sinogram, exist_ok=True)
    os.makedirs(output_folder_diff, exist_ok=True)

    logging.info(f"Decompress folder:{output_directory_decompressed}")
    logging.info(f"Sinogram folder:{output_folder_sinogram}")
    logging.info(f"Difference folder:{output_folder_diff}")

    decom_start_time = time.time()
    # Decompress .gz files
    compressed_files = [(os.path.join(compressed_directory, f), output_directory_decompressed) for f in os.listdir(compressed_directory) if f.endswith('.gz')]
    with Pool(cpu_count()) as pool:
        pool.map(decompress_gzip_file, compressed_files)
    decom_end_time = time.time()
    logging.info(f"Decompressed time {decom_end_time - decom_start_time:.2f} seconds")

    sino_start_time = time.time()
    # Generate sinograms
    generate_sinograms(output_directory_decompressed, output_folder_sinogram)
    sino_end_time = time.time()
    logging.info(f"Generate sinogram time {sino_end_time - sino_start_time:.2f} seconds")


    sinogram_paths = natsorted([os.path.join(output_folder_sinogram, f) for f in os.listdir(output_folder_sinogram) if f.endswith('.tif')])
    sinogram_images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in sinogram_paths]

    diff_start_time = time.time()
    # Calculate absolute difference 
    compute_diff(sinogram_images, reference_image_path, output_folder_diff)
    diff_end_time = time.time()
    logging.info(f"Diffference time is {diff_end_time - diff_start_time:.2f}seconds")
    
    print(f"All processes complete. Total processing time: {time.time() - start_time:.2f} seconds.")
    end_time = time.time()
    logging.info(f"Total processing time:{end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()