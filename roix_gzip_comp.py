import cv2
import numpy as np
import time
import os
import gzip  
import pickle
from skimage.filters import threshold_multiotsu
import glob
import multiprocessing
import argparse

#CHECK FOR GPU SUPPORT#
def check_gpu_support():
   ########"" GPU ACC""#### #CUPY is available
    cupy_available = False
    try:
        import cupy
        cupy_available = True
        print("GPU acceleration is available via CuPy")
    except ImportError:
        cupy_available = False
        print("GPU acceleration is not available------>(CuPy not found)")
    
    return cupy_available

def load_image(image_path):
    ###NORMALIZE MAX VALUE#####
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

def multi_otsu(image, classes=5):
    """Applies Multi-Otsu Thresholding to refine the thresholds."""
    try:
        # Check if image has enough unique values
        unique_values = np.unique(image)
        if len(unique_values) < classes:
            print(f"Image has only {len(unique_values)} unique values, using simple threshold")
            # If not enough unique values, use a simple threshold
            if len(unique_values) <= 1:
                return np.array([0])
            else:
                return np.array([np.mean(unique_values)])
        
        # Try with increased number of bins
        th = threshold_multiotsu(image, classes=classes, nbins=256)
        return th
    except Exception as e:
        print(f"Error in multi_otsu: {e}")
        # Fallback to simple threshold
        mean_val = np.mean(image)
        return np.array([mean_val])

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

def save_compressed_data(compression_data, filename):
    """
    Save compressed data structure into a compressed binary file using gzip.
    Returns the compression time in seconds.
    """
    start_time = time.time()
    with gzip.open(filename, 'wb') as f:
        pickle.dump(compression_data, f)
    compression_time = time.time() - start_time
    return compression_time

def get_file_size(filename):
    """Get the size of a file in bytes."""
    return os.path.getsize(filename)

# CPU Processing Function
def process_image_cpu(image_path, mode, value):
    """Process a single image using CPU."""
    print(f"Processing {image_path} with mode={mode}, value={value} (CPU)")
    image, x_max = load_image(image_path)

    if image is None:
        return None, 0

    # Process the image
    image_shape = image.shape

    # Apply Multi-Otsu Thresholding
    th = multi_otsu(image, classes=5)

    # Binarization
    bin_img = binary(image, th)

    # Find contours directly and extract contour data
    cont_img, contours_data = find_contours_directly(image, bin_img)

    # Apply error-bound quantization to contour data
    # Note: if abs=0, this still applies but acts as lossless
    quantized_contours_data = []
    for data in contours_data:
        # Extract original pixel values
        original_pixels = np.array(data['inside_pixel_values'], dtype=np.int32)

        # If value is 0, no quantization occurs but we use the same data structure
        if mode == "abs" and value == 0:
            quantized_pixels = original_pixels.tolist()
        else:
            # Calculate error bounds
            upper_limit, lower_limit = calculate_error_bounds(original_pixels, mode, value)

            # Apply quantization within error bounds
            quantized_pixels = quantize_with_error_bounds(original_pixels, upper_limit, lower_limit)

        # Update the data dictionary with quantized values
        quantized_data = {
            'row_index': data['row_index'],
            'x_start': data['x_start'],
            'x_end': data['x_end'],
            'quantized_values': quantized_pixels,  # 1D list of integers
        }
        quantized_contours_data.append(quantized_data)

    # Create an image data structure with metadata
    image_data = {
        'filename': os.path.basename(image_path),
        'image_shape': image_shape,
        'max_intensity': x_max,
        'contours_data': quantized_contours_data
    }

    # Return the image data structure and the size of the original image
    size_original = get_file_size(image_path)
    return image_data, size_original

# GPU Processing Function - Updated to match the CPU function
def process_image_gpu(image_path, mode, value):
    """Process a single image using GPU (if available)."""
    try:
        import cupy as cp
        print(f"Processing {image_path} with mode={mode}, value={value} (GPU)")
        
        # Load image (still using CPU for loading)
        image, x_max = load_image(image_path)
        
        if image is None:
            return None, 0
            
        # Get image shape for reconstruction
        image_shape = image.shape
            
        # Transfer to GPU
        gpu_image = cp.asarray(image)
        
        # Apply thresholding on CPU (since skimage doesn't work with cupy)
        th = multi_otsu(image, classes=5)
        
        # Binarization on GPU
        bin_img_gpu = cp.zeros_like(gpu_image, dtype=cp.uint8)
        bin_img_gpu[gpu_image > th[0]] = 255
        
        # Transfer back to CPU for contour finding
        bin_img = cp.asnumpy(bin_img_gpu)
        
        # Find contours (using CPU)
        cont_img, contours_data = find_contours_directly(image, bin_img)
        
        # Apply error-bound quantization to contour data
        quantized_contours_data = []
        for data in contours_data:
            # Extract original pixel values and transfer to GPU
            original_pixels_np = np.array(data['inside_pixel_values'], dtype=np.int32)
            original_pixels = cp.asarray(original_pixels_np)
            
            # If value is 0, no quantization occurs but we use the same data structure
            if mode == "abs" and value == 0:
                quantized_pixels = cp.asnumpy(original_pixels).tolist()
            else:
                # Calculate error bounds on GPU
                if mode == "abs":
                    E = cp.full_like(original_pixels, cp.abs(value), dtype=cp.int32)
                elif mode == "rel":
                    E = (cp.abs(original_pixels) * value).astype(cp.int32)
                elif mode == "absrel":
                    if not isinstance(value, (list, tuple)) or len(value) != 2:
                        raise ValueError("For 'absrel' mode, 'value' must be a tuple/list of two elements.")
                    abs_value = cp.abs(value[0])
                    rel_value = (cp.abs(original_pixels) * value[1]).astype(cp.int32)
                    E = cp.minimum(abs_value, rel_value).astype(cp.int32)
                elif mode == "pwrel":
                    E = (cp.abs(original_pixels) * value).astype(cp.int32)
                
                # Upper and lower bounds
                upper_limit = original_pixels + E
                lower_limit = original_pixels - E
                
                # Transfer back to CPU for quantization
                # (since the quantize_with_error_bounds function is more complex)
                upper_limit_np = cp.asnumpy(upper_limit)
                lower_limit_np = cp.asnumpy(lower_limit)
                original_pixels_np = cp.asnumpy(original_pixels)
                
                # Apply quantization on CPU
                quantized_pixels = quantize_with_error_bounds(original_pixels_np, upper_limit_np, lower_limit_np)
            
            # Update the data dictionary with quantized values
            quantized_data = {
                'row_index': data['row_index'],
                'x_start': data['x_start'],
                'x_end': data['x_end'],
                'quantized_values': quantized_pixels,  # 1D list of integers
            }
            quantized_contours_data.append(quantized_data)
        
        # Create an image data structure with metadata
        image_data = {
            'filename': os.path.basename(image_path),
            'image_shape': image_shape,
            'max_intensity': x_max,
            'contours_data': quantized_contours_data
        }
        
        # Return the image data structure and the size of the original image
        size_original = get_file_size(image_path)
        return image_data, size_original
        
    except ImportError:
        print("CuPy not available, falling back to CPU")
        return process_image_cpu(image_path, mode, value)
    except Exception as e:
        print(f"Error in GPU processing: {e}, falling back to CPU")
        return process_image_cpu(image_path, mode, value)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Image contour quantization with error bounds.')
    parser.add_argument('-input', '--input', default='/home/amarjitsingh/imgrec/p1/newdiff1/', 
                        help='Input folder path containing TIF images')
    parser.add_argument('-output', '--output', default='./', 
                        help='Output folder path for the compressed file')
    parser.add_argument('-abs', '--abs', type=int, default=5, 
                        help='Absolute error bound value (0 for lossless mode)')
    parser.add_argument('-gpu', '--use_gpu', action='store_true',
                        help='Use GPU acceleration if available')
    args = parser.parse_args()
    
    # Set parameters from arguments
    image_folder = args.input
    output_folder = args.output
    abs_value = args.abs
    use_gpu = args.use_gpu
    
    # Check for GPU support
    gpu_available = check_gpu_support() if use_gpu else False
    
    # Choose the processing function based on GPU availability
    process_func = process_image_gpu if use_gpu and gpu_available else process_image_cpu
    
    # Print processing mode
    if abs_value == 0:
        print("Processing in lossless mode (abs=0)")
    else:
        print(f"Processing with abs={abs_value}")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get list of image files
    image_files = glob.glob(os.path.join(image_folder, '*.tif'))
    if not image_files:
        print(f"No .tif files found in {image_folder}")
        return
    
    print(f"Found {len(image_files)} .tif files")

    # Prepare arguments for multiprocessing
    mode = 'abs'  # Using abs mode with the specified value
    args_list = [(image_path, mode, abs_value) for image_path in image_files]

    # Use multiprocessing to process images in parallel
    with multiprocessing.Pool() as pool:
        results = pool.starmap(process_func, args_list)

    # Collect image data and calculate total size
    all_image_data = []
    total_original_size = 0
    total_contour_rows = 0

    for image_data, size_original in results:
        if image_data is not None:
            all_image_data.append(image_data)
            total_original_size += size_original
            total_contour_rows += len(image_data['contours_data'])

    print(f"Successfully processed {len(all_image_data)} images")
    print(f"Total rows of contour data: {total_contour_rows}")

    # Create a comprehensive data structure for all images
    compression_data = {
        'images': all_image_data,
        'mode': mode,
        'value': abs_value,
        'timestamp': time.time(),
        'source_folder': os.path.basename(os.path.abspath(image_folder))
    }

    # Save compressed data into a compressed binary file using gzip
    mode_str = "lossless" if abs_value == 0 else f"abs_{abs_value}"
    filename_contour_bin = os.path.join(output_folder, f'contour_quantized_{mode_str}.bin.gz')
    compression_time = save_compressed_data(compression_data, filename_contour_bin)

    # Get the size of the quantized contour data file
    size_contour_bin = get_file_size(filename_contour_bin)

    # Calculate compression throughput
    compression_throughput = total_original_size / compression_time if compression_time > 0 else 0

    # Print out the sizes, time, and compression ratio
    print(f"\nCompression Results:")
    print(f"Total size of original images: {total_original_size} bytes")
    print(f"Total size of compressed data: {size_contour_bin} bytes")
    print(f"Compression time: {compression_time:.2f} seconds")
    print(f"Compression throughput: {compression_throughput / (1024 * 1024):.2f} MB/s")
    print(f"Compression ratio (Original / Compressed): {total_original_size / size_contour_bin if size_contour_bin != 0 else float('inf'):.2f}")
    print(f"Output file: {filename_contour_bin}")

if __name__ == "__main__":
    main()