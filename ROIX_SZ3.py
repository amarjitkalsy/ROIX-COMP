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

def multi_otsu(img_nor, classes=5):
    start_time = time.time()
    th = threshold_multiotsu(img_nor, classes=classes)
    elapsed_time = time.time() - start_time
    return th, elapsed_time

def binary(img_nor, th):
    start_time = time.time()
    bin_img = np.zeros_like(img_nor)
    bin_img[img_nor > th[0]] = 255
    elapsed_time = time.time() - start_time
    return bin_img, elapsed_time

def find_contours_directly(orig_img, bin_img):
    """Find contours and calculate x-boundaries directly from the binary image."""
    start_time = time.time()
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
    contours_data = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        for i in range(y, y + h):
            row_slice = bin_img[i, x:x + w]
            xs = np.where(row_slice == 255)[0] + x
            if xs.size > 0:
                row_x_start = xs.min()
                row_x_end = xs.max()
                pixel_values = orig_img[i, row_x_start:row_x_end + 1]
                contours_data.append({
                    'row_index': i,
                    'x_start': row_x_start,
                    'x_end': row_x_end,
                    'inside_pixel_values': pixel_values.tolist()
                })
    elapsed_time = time.time() - start_time
    return cont_img, contours_data, contours, elapsed_time  # Return contours

def process_image(image_path):
    # Re-import necessary modules in the child process
    import os
    import sys
    import cv2
    import numpy as np
    from skimage.filters import threshold_multiotsu

    timings = {}
    total_start_time = time.time()
    (image, x_max), elapsed_time = load_image(image_path)
    timings['load_image'] = elapsed_time

    if image is None:
        return None

    try:
        image_shape = image.shape

        th, elapsed_time = multi_otsu(image, classes=5)
        timings['multi_otsu'] = elapsed_time

        bin_img, elapsed_time = binary(image, th)
        timings['binary'] = elapsed_time

        # Corrected function call
        _, contours_data, contours, elapsed_time = find_contours_directly(image, bin_img)
        timings['find_contours_directly'] = elapsed_time

        total_end_time = time.time()
        timings['process_image'] = total_end_time - total_start_time

        # Prepare data to return
        data_to_return = {
            'filename': image_path,
            'image_shape': image_shape,
            'x_max': x_max,
            'contours_data': contours_data,  # Include contours_data with inside_pixel_values
            'original_image': image  # Add the original image for reconstruction
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

    # Now, collect all contours_data and prepare for compression
    all_contours_data = []
    for data in processed_data:
        contours_data = data['contours_data']
        for contour_info in contours_data:
            all_contours_data.append(contour_info)

    # Determine the maximum length
    max_length = max(len(ci['inside_pixel_values']) for ci in all_contours_data)
    max_pixel_values_length = max_length
    print(f"Maximum 'inside_pixel_values' length: {max_pixel_values_length}")

    num_contours = len(all_contours_data)
    # Define data types
    index_dtype = np.int32
    pixel_dtype = np.uint8

    # Create structured array with appropriate data types
    structured_array = np.zeros((num_contours, 3 + max_pixel_values_length), dtype=np.int32)

    for idx, contour_info in enumerate(all_contours_data):
        # Store indices as int32
        structured_array[idx, 0] = contour_info['row_index']
        structured_array[idx, 1] = contour_info['x_start']
        structured_array[idx, 2] = contour_info['x_end']
        pixel_values = np.array(contour_info['inside_pixel_values'], dtype=np.int32)
        length = min(len(pixel_values), max_pixel_values_length)
        structured_array[idx, 3:3+length] = pixel_values[:length]
        # Handle padding if necessary (already zeros by default)

    # Now, compress the structured_array
    decomp_structured_array = np.empty_like(structured_array)

    # Print original data size (now using total_original_size)
    print(f"Total original images size: {total_original_size / (1024 * 1024):.2f} MB")

    # The compression loop
    for bound in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        compressor = libpressio.PressioCompressor.from_config({
            # Configure which compressor to use
            "compressor_id": "zfp",
            # Configure the set of metrics to be gathered
            "early_config": {
                "pressio:metric": "composite",
                "composite:plugins": ["time", "size", "error_stat"]
            },
            # Configure SZ3
            "compressor_config": {
                "pressio:abs": float(bound),
            }})
        # Run compressor to determine metrics
        input_data = structured_array
        comp_data = compressor.encode(input_data)
        decomp_structured_array = compressor.decode(comp_data, decomp_structured_array)
        metrics = compressor.get_metrics()

        # Calculate compressed data size
        compressed_size = len(comp_data)
        compression_ratio = total_original_size / compressed_size
        print(f"\nCompression with bound={bound:1.0e}")
        print(f"Compressed data size: {compressed_size / (1024 * 1024):.2f} MB")
        print(f"Compression ratio (original images size / compressed size): {compression_ratio:.2f}")

        # Verify the decompressed data
        reconstruction_error = np.abs(structured_array - decomp_structured_array)
        max_error = np.max(reconstruction_error)
        print(f"Maximum reconstruction error: {max_error}")

        # Check if maximum error is within the specified bound
        if max_error > bound:
            print(f"Warning: Maximum error {max_error} exceeds bound {bound}")

        # Print metrics
        print(f"Metrics: {json.dumps(metrics, indent=4)}")

        # Reconstruct an image for verification
        image_shape = processed_data[0]['image_shape']
        reconstructed_image = reconstruct_image(decomp_structured_array, image_shape)
        original_image = processed_data[0]['original_image']

        # Calculate the difference
        difference = np.abs(original_image.astype(np.int16) - reconstructed_image.astype(np.int16))
        max_difference = np.max(difference)
        print(f"Maximum pixel difference between original and reconstructed image: {max_difference}")

    total_end_time = time.time()
    total_timings['total_time'] = total_end_time - start_time
    print("\nProcessing complete. Timings:", total_timings)

def reconstruct_image(decomp_structured_array, image_shape):
    # Create an empty image
    reconstructed_image = np.zeros(image_shape, dtype=np.uint8)

    for row in decomp_structured_array:
        row_index = int(row[0])
        x_start = int(row[1])
        x_end = int(row[2])
        pixel_values = row[3:]
        # Remove padding zeros
        pixel_values = pixel_values[:x_end - x_start + 1]
        pixel_values = pixel_values.astype(np.uint8)
        if len(pixel_values) > 0:
            reconstructed_image[row_index, x_start:x_end + 1] = pixel_values
    return reconstructed_image

if __name__ == '__main__':
    import multiprocessing

    # Redirect stdout to a file
    sys.stdout = open('sz3_diff1.txt', 'w')

    # Set the start method to 'fork' to help with module imports (Unix-based systems)
    multiprocessing.set_start_method('fork', force=True)
    process_folder('diff1')

    # Close the output file
    sys.stdout.close()
