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


# Setup log > decompression_output.log for decompression
logging.basicConfig(filename='decomp_zstd.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def decompress_zstd_file(input_file_path, output_folder):
    """
    Decompresses a Zstandard compressed file to the specified output folder.
    Assumes the .zst file contains multiple sequentially compressed TIFF images.
    """
    dctx = zstd.ZstdDecompressor()
    image_number = 0 
    with open(input_file_path, "rb") as zstd_input_file: 
        while True:
            size_bytes = zstd_input_file.read(4)  # Corrected method call
            if not size_bytes:
                break
            size = int.from_bytes(size_bytes, 'big')
            compressed_data = zstd_input_file.read(size)
            decompressed_data = dctx.decompress(compressed_data)

            output_file_path = os.path.join(output_folder, f"decompressed_image_{image_number}.tif")
            with open(output_file_path, "wb") as tiff_output_file:
                tiff_output_file.write(decompressed_data)
            image_number += 1

def generate_sinograms(image_folder, sinogram_folder):
  
    if not os.path.exists(sinogram_folder):
        os.makedirs(sinogram_folder)
    # Load images while preserving their original bit depth and ensuring they are sorted
    image_paths = natsorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.startswith('decompressed_image_') and f.endswith('.tif')])
    images = np.array([cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in image_paths])  # Use cv2.IMREAD_UNCHANGED to load images in their original bit depth
    
    # Generate sinograms by transposing the axes
    sinograms = np.transpose(images, axes=(1, 0, 2))
    
    # Save sinograms in 16-bit format if needed
    for i, sinogram in enumerate(sinograms):
        output_path = os.path.join(sinogram_folder, f"sinogram_{i:04d}.tif")
        # Convert sinogram to 16-bit if necessary and save
        tifffile.imwrite(output_path, img_as_uint(sinogram))  # img_as_uint ensures conversion to 16-bit unsigned integer format

    logging.info("Sinograms generated and saved, maintaining the original bit depth of the images.")


def calculate_absolute_difference(reference_image_path, input_folder, output_folder_diff):
    """
    Calculates the absolute difference between a reference image and each image in the input folder.
    """
    img1 = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    if img1 is None:
        print("Failed to load the reference image.")
        return

    image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.tif')]
    for image_path in image_paths:
        img2 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img2 is not None:
            diff = cv2.absdiff(img1, img2)
            output_path = os.path.join(output_folder_diff, os.path.basename(image_path))
            io.imsave(output_path, img_as_uint(diff))

def main():
    start_time = time.time()

    input_file_path = '/home/amarjitsingh/imgrec/p1/all.zst'
    output_folder_decompressed = 'zstd'
    output_folder_sinogram = 'zstdsinograms'
    output_folder_diff = 'absdifference'
    reference_image_path = '/home/amarjitsingh/imgrec/q001.tif'

    os.makedirs(output_folder_decompressed, exist_ok=True)
    os.makedirs(output_folder_sinogram, exist_ok=True)
    os.makedirs(output_folder_diff, exist_ok=True)
    
    # Decompress all TIFF files
    decompress_zstd_file(input_file_path, output_folder_decompressed)
    
    # Generate sinograms
    generate_sinograms(output_folder_decompressed, output_folder_sinogram)
    
    # Calculate absolute difference
    calculate_absolute_difference(reference_image_path, output_folder_sinogram, output_folder_diff)

    print(f"All processes complete. Total processing time: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
