import cv2
import os
import numpy as np
import tifffile
from natsort import natsorted
import time
import ffmpeg
from skimage import img_as_ubyte, img_as_uint
import logging


# Setup log > Output.log for compression 
logging.basicConfig(filename='output.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load(image_paths):
    """Load images"""
    images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
    return np.array(images)

def compute_diff(images, reference_image_path):
    """compute absolute differences"""
    ref_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    # initialize
    differences = np.empty_like(images)
    # Compute the absolute difference
    for i in range(images.shape[0]):
        differences[i] = cv2.absdiff(ref_image, images[i])
    return differences

def save_img(images, output_folder, prefix):
    """save images"""
    for i, img in enumerate(images):
        output_path = os.path.join(output_folder, f"{prefix}{i}.tif")
        tifffile.imwrite(output_path, img_as_uint(img), compression=None)

def gen_sinograms(images, output_folder):
    """generate sinograms"""
    sinograms = np.transpose(images, axes=(1, 0, 2))
    save_img(sinograms, output_folder, prefix='combined_row_')

def compress_ffmpeg(input_folder, output_video_file):
    print("compress_images_to_video function is called")
    input_pattern = os.path.join(input_folder, 'combined_row_%d.tif')
    try:
        (
            ffmpeg
            .input(input_pattern, format='image2', start_number=0)
            .output(output_video_file, vcodec='magicyuv', y=None)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"Video compressed successfully: {output_video_file}")
    except ffmpeg.Error as e:
        print(f"Failed to compress video. FFmpeg error: {e.stderr}")

def main():
    start_time = time.time()
    logging.info("Process started")
    
    folder_path = '/home/amarjitsingh/imgrec/XCT4K_tiff_0_0_8'
    output_folder_diff = '/home/amarjitsingh/imgrec/xct/p1/diff1/'
    output_folder_sinogram = '/home/amarjitsingh/imgrec/xct/p1/sinograms/'
    reference_image_path = "/home/amarjitsingh/imgrec/q0001.tiff"

    os.makedirs(output_folder_diff, exist_ok=True)
    os.makedirs(output_folder_sinogram, exist_ok=True)
    
    logging.info(f"Input folder: {folder_path}")
    logging.info(f"Output difference folder: {output_folder_diff}")
    logging.info(f"Output sinogram folder: {output_folder_sinogram}")
    logging.info(f"Reference image: {reference_image_path}")

    # Load images
    image_paths = natsorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.tiff')])
    images = load(image_paths)

    # Compute differences
    diff_start_time = time.time()
    differences = compute_diff(images, reference_image_path)
    diff_end_time = time.time()
    logging.info(f"Computed differences in {diff_end_time - diff_start_time:.2f} seconds")

    # Save the difference images
    save_img(differences, output_folder_diff, prefix='diff_')

    # Generate and save sinograms
    sino_start_time = time.time()
    gen_sinograms(differences, output_folder_sinogram)
    sino_end_time = time.time()
    logging.info(f"Generated sinograms in {sino_end_time - sino_start_time:.2f} seconds")

    # compress the sinograms into a video
    compress_start_time = time.time()
    output_video_file = '/home/amarjitsingh/imgrec/xct/p1/sinograms_video.mkv'
    compress_ffmpeg(output_folder_sinogram, output_video_file)
    compress_end_time = time.time()
    logging.info(f"Compressed to video in {compress_end_time - compress_start_time:.2f} seconds")
    logging.info(f"Video file: {output_video_file}")

    total_time = time.time() - start_time
    logging.info(f"All processes complete. Total processing time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()