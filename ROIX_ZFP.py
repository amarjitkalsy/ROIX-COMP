import os
import cv2
import numpy as np
import time
from multiprocessing import Pool, cpu_count
from skimage.filters import threshold_multiotsu
import libpressio
import json
import sys

def load_image(image_path):
    """Load an image from a path and normalize it based on its max value."""
    start_time = time.time()
    print(f"Processing file {image_path}...")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to load image {image_path}.")
        elapsed_time = time.time() - start_time
        return (None, None), elapsed_time
    x_max = np.max(image)
    if x_max > 0:
        img_nor = image * (255 / x_max)
        img_nor = np.clip(img_nor, 0, 255).astype(np.uint8)
    else:
        img_nor = image
    elapsed_time = time.time() - start_time
    return (img_nor, x_max), elapsed_time

def multi_otsu(img_nor, classes=3):
    """Apply Multi-Otsu Thresholding."""
    start_time = time.time()
    th = threshold_multiotsu(img_nor, classes=classes)
    elapsed_time = time.time() - start_time
    return th, elapsed_time

def binary(img_nor, th):
    """Binarize the image based on the calculated thresholds."""
    start_time = time.time()
    bin_img = np.zeros_like(img_nor)
    bin_img[img_nor > th[0]] = 255
    elapsed_time = time.time() - start_time
    return bin_img, elapsed_time

def find_contours_directly(orig_img, bin_img):
    """Find contours and extract contour data."""
    start_time = time.time()
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_data = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        for i in range(y, y + h):
            row_slice = bin_img[i, x:x + w]
            xs = np.where(row_slice == 255)[0] + x
            if xs.size > 0:
                row_x_start = xs.min()
                row_x_end = xs.max()
                pixel_values = orig_img[i, row_x_start:row_x_end + 1].astype(np.int32)  # Use float32 for SZ3
                contours_data.append({
                    'row_index': i,
                    'x_start': row_x_start,
                    'x_end': row_x_end,
                    'inside_pixel_values': pixel_values
                })
    elapsed_time = time.time() - start_time
    return None, contours_data, contours, elapsed_time

def process_image(image_path):
    """Process a single image and return contour data."""
    timings = {}
    total_start_time = time.time()
    (image, x_max), elapsed_time = load_image(image_path)
    timings['load_image'] = elapsed_time

    if image is None:
        return None

    try:
        image_shape = image.shape

        th, elapsed_time = multi_otsu(image, classes=3)
        timings['multi_otsu'] = elapsed_time

        bin_img, elapsed_time = binary(image, th)
        timings['binary'] = elapsed_time

        _, contours_data, contours, elapsed_time = find_contours_directly(image, bin_img)
        timings['find_contours_directly'] = elapsed_time

        total_end_time = time.time()
        timings['process_image'] = total_end_time - total_start_time

        # Add filename and image_shape to each contour
        for contour in contours_data:
            contour['filename'] = image_path
            contour['image_shape'] = image_shape

        # Prepare data to return
        data_to_return = {
            'filename': image_path,
            'image_shape': image_shape,
            'x_max': x_max,
            'contours_data': contours_data,
            'original_image': image  # For reconstruction
        }

        return (data_to_return, timings)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def calculate_total_original_size(folder_path):
    """Calculate the total size of all image files in the folder."""
    total_size = 0
    for dirpath, _, filenames in os.walk(folder_path):
        for f in filenames:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
    return total_size

def process_folder(folder_path):
    """Process all images in a folder and compress contour data."""
    start_time = time.time()
    if not os.path.exists(folder_path):
        print(f"Folder path does not exist: {folder_path}")
        return
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    if not image_files:
        print(f"No image files found in the folder: {folder_path}")
        return

    # Calculate total original size
    total_original_size = calculate_total_original_size(folder_path)

    # Use multiprocessing to process images in parallel
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_image, image_files)

    total_timings = {}
    processed_data = []
    for res in results:
        if res is not None:
            data, timings = res
            processed_data.append(data)
            for key, value in timings.items():
                total_timings[key] = total_timings.get(key, 0) + value

    # Collect all contours_data
    all_contours_data = []
    for data in processed_data:
        contours_data = data['contours_data']
        all_contours_data.extend(contours_data)

    # Prepare data for compression
    # Since the contour data has variable lengths, we need to handle this
    # We'll create a list of NumPy arrays for 'inside_pixel_values'

    # Collect all pixel values into a single NumPy array
    pixel_values_list = [cd['inside_pixel_values'] for cd in all_contours_data]
    total_pixel_values = np.concatenate(pixel_values_list)
    total_pixel_values = total_pixel_values.astype(np.float32)

    # Store indices for reconstructing the contours
    indices = []
    start_idx = 0
    for pv in pixel_values_list:
        length = len(pv)
        indices.append((start_idx, length))
        start_idx += length

    # Now, compress the total_pixel_values array using libpressio
    decomp_pixel_values = np.empty_like(total_pixel_values)

    # Create directory for reconstructed images
    reconstructed_dir = 'reconstructed_images'
    if not os.path.exists(reconstructed_dir):
        os.makedirs(reconstructed_dir)

    # The compression loop
    for bound in [0, 1e-1, 1e0, 5e0, 1e1, 1.5e1]:
        compressor = libpressio.PressioCompressor.from_config({
            # Configure which compressor to use
            "compressor_id": "zfp",
            # Configure the set of metrics to be gathered
            "early_config": {
                "pressio:metric": "composite",
                "composite:plugins": ["size", "time", "error_stat"]
            },
            # Configure SZ3
            "compressor_config": {
                "pressio:abs": int(bound),
            }
        })

        try:
            # Compress the data
            compressed_data = compressor.encode(total_pixel_values)

            # Calculate compressed data size
            compressed_size = compressed_data.size * compressed_data.itemsize
            original_size = total_pixel_values.size * total_pixel_values.itemsize
            compression_ratio = original_size / compressed_size

            print(f"\nCompression with bound={bound:1.0e}")
            print(f"Compressed data size: {compressed_size / (1024 * 1024):.2f} MB")
            print(f"Compression ratio (original data size / compressed size): {compression_ratio:.2f}")

            # Decompress the data
            decompressed_data = compressor.decode(compressed_data, decomp_pixel_values)

            # Verify the decompressed data
            max_error = np.max(np.abs(total_pixel_values - decompressed_data))
            print(f"Maximum reconstruction error: {max_error}")

            # Check if maximum error is within the specified bound
            if max_error > bound:
                print(f"Warning: Maximum error {max_error} exceeds bound {bound}")

            # Retrieve and print metrics
            metrics = compressor.get_metrics()
            print(f"Metrics: {json.dumps(metrics, indent=4)}")

            # Update 'inside_pixel_values' with decompressed data
            start_idx = 0
            for idx, cd in enumerate(all_contours_data):
                length = cd['inside_pixel_values'].size
                cd['inside_pixel_values'] = decompressed_data[start_idx:start_idx+length]
                start_idx += length

            # Group contours by filename for reconstruction
            contours_by_image = {}
            for cd in all_contours_data:
                filename = cd['filename']
                if filename not in contours_by_image:
                    contours_by_image[filename] = []
                contours_by_image[filename].append(cd)

            # Reconstruct images and save them
            for data in processed_data:
                filename = data['filename']
                image_shape = data['image_shape']
                reconstructed_image = np.zeros(image_shape, dtype=np.uint8)
                contours_data = contours_by_image.get(filename, [])
                for cd in contours_data:
                    row_index = cd['row_index']
                    x_start = cd['x_start']
                    x_end = cd['x_end']
                    pixel_values = cd['inside_pixel_values'].astype(np.uint8)
                    reconstructed_image[row_index, x_start:x_end+1] = pixel_values
                # Save the reconstructed image
                output_filename = os.path.join(reconstructed_dir, f"{os.path.basename(filename)}_bound_{bound:1.0e}.png")
                cv2.imwrite(output_filename, reconstructed_image)

                # Optional: Calculate difference between original and reconstructed images
                original_image = data['original_image']
                difference = np.abs(original_image.astype(np.int16) - reconstructed_image.astype(np.int16))
                max_difference = np.max(difference)
                print(f"Image: {filename}, Maximum pixel difference: {max_difference}")

        except Exception as e:
            print(f"Error during compression with bound={bound}: {e}")
            continue

    total_end_time = time.time()
    total_timings['total_time'] = total_end_time - start_time
    print("\nProcessing complete. Timings:", total_timings)

if __name__ == '__main__':
    import multiprocessing
    import faulthandler
    faulthandler.enable()
    # Optionally redirect stdout to a file
    # sys.stdout = open('sz3_diff1.txt', 'w')

    # Set the start method to 'fork' to help with module imports (Unix-based systems)
    process_folder('diff11')

    # Close the output file if redirected
    # sys.stdout.close() 
