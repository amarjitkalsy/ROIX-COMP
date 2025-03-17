import cv2
import numpy as np
import time
import os
import gzip  # Using gzip for compression
import pickle
from skimage.filters import threshold_multiotsu
import glob
import multiprocessing

def load_image(image_path):
    """Load an image from a path and normalize it based on its max value."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None, None
    x_max = np.max(image)
    if x_max > 0:
        img_nor = image * (255 / x_max)
        img_nor = np.clip(img_nor, 0, 255).astype(np.uint8)
    else:
        img_nor = image
    return img_nor, x_max

def multi_otsu(image, classes=3):
    """Applies Multi-Otsu Thresholding to refine the thresholds."""
    th = threshold_multiotsu(image, classes=classes)
    return th

def binary(image, th):
    """Binarize the image based on the calculated Multi-Otsu thresholds."""
    bin_img = np.zeros_like(image, dtype=np.uint8)
    bin_img[image > th[0]] = 255
    return bin_img

def find_contours_directly(orig_img, bin_img):
    """Find contours and draw them on the original image, calculate x-boundaries directly from the binary image."""
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
                contours_data.append({'row_index': i, 'x_start': row_x_start, 'x_end': row_x_end, 'inside_pixel_values': pixel_values.tolist()})
    return cont_img, contours_data

def calculate_error_bounds(original, mode, value):
    """Calculate the upper and lower error bounds based on the mode."""
    if mode == "abs":
        E = np.full_like(original, np.abs(value), dtype=np.int32)
    elif mode == "rel":
        E = (np.abs(original) * value).astype(np.int32)
    elif mode == "absrel":
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("For 'absrel' mode, 'value' must be a tuple/list of two elements (abs_value, rel_value).")
        abs_value = np.abs(value[0])
        rel_value = (np.abs(original) * value[1]).astype(np.int32)
        E = np.minimum(abs_value, rel_value).astype(np.int32)
    elif mode == "pwrel":
        E = (np.abs(original) * value).astype(np.int32)  # Point-wise relative tolerance
    else:
        raise ValueError("Invalid mode for error bound.")

    # Calculate the upper and lower bounds
    upper_limit = original + E
    lower_limit = original - E
    return upper_limit, lower_limit

def quantize_with_error_bounds(original, upper_limit, lower_limit):
    """Apply quantization within error bounds."""
    # Flatten the arrays for sequential processing
    original_flat = original.flatten()
    upper_limit_flat = upper_limit.flatten()
    lower_limit_flat = lower_limit.flatten()

    quantized_flat = np.zeros_like(original_flat, dtype=np.int32)

    # Initialize the overlapping area
    u = float('inf')
    l = -float('inf')
    head = 0  # Start index of the current segment

    for i in range(len(original_flat)):
        # Update the overlapping area
        l_new = max(l, lower_limit_flat[i])
        u_new = min(u, upper_limit_flat[i])

        if u_new < l_new:
            # The overlapping area is empty
            # Set the values up to this point to (u + l) / 2
            quantized_value = int((u + l) / 2)
            quantized_flat[head:i] = quantized_value
            # Reset the overlapping area
            l = lower_limit_flat[i]
            u = upper_limit_flat[i]
            head = i
        else:
            # Update the overlapping area
            l = l_new
            u = u_new

    # Set the remaining values
    quantized_value = int((u + l) / 2)
    quantized_flat[head:] = quantized_value

    # Clip the values to valid image range and convert to integers
    quantized_flat = np.clip(quantized_flat, 0, 255).astype(int)
    return quantized_flat.reshape(original.shape).flatten().tolist()  # 1D list of integers

def save_quantized_contours(contours_data, filename):
    """
    Save quantized contour data into a compressed binary file using gzip.
    Returns the compression time in seconds.
    """
    start_time = time.time()
    with gzip.open(filename, 'wb') as f:
        pickle.dump(contours_data, f)
    compression_time = time.time() - start_time
    return compression_time

def load_quantized_contours(filename):
    """
    Load quantized contour data from a compressed binary file using gzip.
    Returns the loaded data and the decompression time in seconds.
    """
    start_time = time.time()
    with gzip.open(filename, 'rb') as f:
        contours_data = pickle.load(f)
    decompression_time = time.time() - start_time
    return contours_data, decompression_time

def get_file_size(filename):
    """Get the size of a file in bytes."""
    return os.path.getsize(filename)

def process_image(image_path, mode, value):
    """Process a single image and return quantized contour data."""
    print(f"Processing {image_path} with mode={mode}, value={value}")
    image, x_max = load_image(image_path)

    if image is None:
        return [], 0

    # Process the image
    image_shape = image.shape

    # Apply Multi-Otsu Thresholding
    th = multi_otsu(image, classes=3)

    # Binarization
    bin_img = binary(image, th)

    # Find contours directly and extract contour data
    cont_img, contours_data = find_contours_directly(image, bin_img)

    # Determine whether to apply quantization
    apply_quantization = True
    if mode == "lossless":
        apply_quantization = False
        print(f"Quantization is skipped as mode='lossless' for image {image_path}.")
    elif mode == "abs" and value == 0:
        apply_quantization = False
        print(f"Quantization is skipped as mode='abs' and value=0 for image {image_path}.")

    # Apply error-bound quantization to contour data if needed
    quantized_contours_data = []
    if apply_quantization:
        for data in contours_data:
            # Extract original pixel values
            original_pixels = np.array(data['inside_pixel_values'], dtype=np.int32)

            # Calculate error bounds
            upper_limit, lower_limit = calculate_error_bounds(original_pixels, mode, value)

            # Apply quantization within error bounds
            quantized_pixels = quantize_with_error_bounds(original_pixels, upper_limit, lower_limit)

            # Update the data dictionary with quantized values
            quantized_data = {
                'row_index': data['row_index'],
                'x_start': data['x_start'],
                'x_end': data['x_end'],
                'quantized_values': quantized_pixels  # 1D list of integers
            }
            quantized_contours_data.append(quantized_data)
    else:
        # No quantization; use original pixel values
        for data in contours_data:
            quantized_data = {
                'row_index': data['row_index'],
                'x_start': data['x_start'],
                'x_end': data['x_end'],
                'quantized_values': data['inside_pixel_values']  # 1D list of integers
            }
            quantized_contours_data.append(quantized_data)

    # Return the quantized_contours_data and the size of the original image
    size_original = get_file_size(image_path)
    return quantized_contours_data, size_original

def main():
    # Paths
    image_folder = '/home/amarjitsingh/imgrec/xct/p1/diff1/' 

    # Get list of image files
    image_files = glob.glob(os.path.join(image_folder, '*.tif'))

    # First, process images with quantization
    mode = 'abs'  # Choose error bound mode: "abs", "rel", "absrel", "pwrel", "lossless"
    value = 0.1    # Adjust the value as needed

    # Prepare arguments for multiprocessing
    args_list = [(image_path, mode, value) for image_path in image_files]

    # Use multiprocessing to process images in parallel
    with multiprocessing.Pool() as pool:
        results = pool.starmap(process_image, args_list)

    all_quantized_contours_data = []
    total_original_size = 0

    for quantized_contours_data, size_original in results:
        all_quantized_contours_data.extend(quantized_contours_data)
        total_original_size += size_original

    # Save quantized contour data into a compressed binary file using gzip
    filename_contour_bin = 'contour_quantized.bin.gz'
    compression_time = save_quantized_contours(all_quantized_contours_data, filename_contour_bin)

    # Get the size of the quantized contour data file
    size_contour_bin = get_file_size(filename_contour_bin)

    # Load the data to measure decompression time
    _, decompression_time = load_quantized_contours(filename_contour_bin)

    # Calculate compression and decompression throughputs
    compression_throughput = total_original_size / compression_time if compression_time > 0 else 0
    decompression_throughput = total_original_size / decompression_time if decompression_time > 0 else 0

    # Now process images without quantization
    mode_no_quant = 'lossless'
    value_no_quant = None  # Not needed for 'lossless' mode

    # Prepare arguments for multiprocessing
    args_list_no_quant = [(image_path, mode_no_quant, value_no_quant) for image_path in image_files]

    # Use multiprocessing to process images in parallel
    with multiprocessing.Pool() as pool:
        results_no_quant = pool.starmap(process_image, args_list_no_quant)

    all_quantized_contours_data_no_quant = []
    total_original_size_no_quant = 0

    for quantized_contours_data_no_quant, size_original in results_no_quant:
        all_quantized_contours_data_no_quant.extend(quantized_contours_data_no_quant)
        total_original_size_no_quant += size_original

    # Save non-quantized contour data into a compressed binary file using gzip
    filename_contour_bin_no_quant = 'contour_nonquantized.bin.gz'
    compression_time_no_quant = save_quantized_contours(all_quantized_contours_data_no_quant, filename_contour_bin_no_quant)

    # Get the size of the non-quantized contour data file
    size_contour_bin_no_quant = get_file_size(filename_contour_bin_no_quant)

    # Load the data to measure decompression time
    _, decompression_time_no_quant = load_quantized_contours(filename_contour_bin_no_quant)

    # Calculate compression and decompression throughputs
    compression_throughput_no_quant = total_original_size_no_quant / compression_time_no_quant if compression_time_no_quant > 0 else 0
    decompression_throughput_no_quant = total_original_size_no_quant / decompression_time_no_quant if decompression_time_no_quant > 0 else 0

    # Print out the sizes, times, and compression ratios
    print(f"Total size of original images: {total_original_size} bytes")
    print(f"Total size of quantized contour data: {size_contour_bin} bytes")
    print(f"Compression time (quantized): {compression_time:.2f} seconds")
    print(f"Decompression time (quantized): {decompression_time:.2f} seconds")
    print(f"Compression throughput (quantized): {compression_throughput / (1024 * 1024):.2f} MB/s")
    print(f"Decompression throughput (quantized): {decompression_throughput / (1024 * 1024):.2f} MB/s")
    print(f"Compression ratio (Original / Quantized Contour Data): {total_original_size / size_contour_bin if size_contour_bin != 0 else float('inf'):.2f}")

    print(f"Total size of non-quantized contour data: {size_contour_bin_no_quant} bytes")
    print(f"Compression time (non-quantized): {compression_time_no_quant:.2f} seconds")
    print(f"Decompression time (non-quantized): {decompression_time_no_quant:.2f} seconds")
    print(f"Compression throughput (non-quantized): {compression_throughput_no_quant / (1024 * 1024):.2f} MB/s")
    print(f"Decompression throughput (non-quantized): {decompression_throughput_no_quant / (1024 * 1024):.2f} MB/s")
    print(f"Compression ratio (Original / Non-Quantized Contour Data): {total_original_size_no_quant / size_contour_bin_no_quant if size_contour_bin_no_quant != 0 else float('inf'):.2f}")

if __name__ == "__main__":
    main()
