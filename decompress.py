import cv2
import os
import numpy as np
import tifffile
from natsort import natsorted
import time
import ffmpeg
from skimage import img_as_uint
import logging

# Setup logging
logging.basicConfig(filename='decompression_output.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract(video_file, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    (
        ffmpeg
        .input(video_file)
        .output(os.path.join(output_folder, 'frame_%04d.tif'), pix_fmt='gray16le')
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )
    logging.info("16-bit TIFF frames extracted from video.")

def gen_sinograms(image_folder, sinogram_folder):
    if not os.path.exists(sinogram_folder):
        os.makedirs(sinogram_folder)
    image_paths = natsorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.tif')])
    images = np.array([cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths])
    sinograms = np.transpose(images, axes=(1, 0, 2))
    for i, sinogram in enumerate(sinograms):
        output_path = os.path.join(sinogram_folder, f"sinogram_{i:04d}.tif")
        # img_as_uint
        tifffile.imwrite(output_path, img_as_uint(sinogram))
    logging.info("Sinograms generated and saved as 16-bit TIFF images.")

def calculate_diff(sinogram_folder, reference_image_path, restored_folder):
    if not os.path.exists(restored_folder):
        os.makedirs(restored_folder)
    ref_image = cv2.imread(reference_image_path, cv2.IMREAD_UNCHANGED)
    sinogram_paths = natsorted([os.path.join(sinogram_folder, filename) for filename in os.listdir(sinogram_folder) if filename.endswith('.tif')])
    for path in sinogram_paths:
        sinogram = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        difference_image = cv2.absdiff(sinogram, ref_image)
        output_path = os.path.join(restored_folder, os.path.basename(path))
        # img_as_uint
        tifffile.imwrite(output_path, img_as_uint(difference_image))
    logging.info("Restored images from sinogram differences and saved as 16-bit TIFF images.")

def main():
    start_time = time.time()
    logging.info("Decompression process started")
    video_file = '/home/amarjitsingh/imgrec/xct/p1/sinograms_video.mkv'
    output_frames_folder = 'output'
    sinogram_folder = 'sino'
    restored_folder = 'restore'
    reference_image_path = '/home/amarjitsingh/imgrec/xct/q0001.tiff'
    
    os.makedirs(output_frames_folder, exist_ok=True)
    os.makedirs(sinogram_folder, exist_ok=True)
    os.makedirs(restored_folder, exist_ok=True)


    logging.info(f"Input extract video to images folder: {extract}")
    logging.info(f"image to sinogram folder: {gen_sinograms}")
    logging.info(f"calculate difference folder: {calculate_diff}")

    diff_start_time = time.time()
    extract(video_file, output_frames_folder)
    diff_end_time = time.time()
    logging.info(f"Computed differences in {diff_end_time - diff_start_time:.2f} seconds")

    sino_start_time = time.time()
    gen_sinograms(output_frames_folder, sinogram_folder)
    sino_end_time = time.time()
    logging.info(f"Generated sinograms in {sino_end_time - sino_start_time:.2f} seconds")

    diff_start_time = time.time()
    calculate_diff(sinogram_folder, reference_image_path, restored_folder)
    diff_end_time = time.time()
    logging.info(f"Difference in {sino_end_time - sino_start_time:.2f} seconds")

    total_time = time.time() - start_time
    logging.info(f"Decompression process completed successfully. Total processing time: {total_time:.2f} seconds.")
    print(f"All processes complete. Total processing time: {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()
