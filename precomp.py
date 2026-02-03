import numpy as np
import cv2
import os
from skimage import io
import argparse

#CHECK FOR GPU SUPPORT#
def check_gpu_support():
    ########"" GPU ACC""#### #CUPY is available
    cupy_available = False
    try:
        import cupy
        cupy_available = True
    except ImportError:
        cupy_available = False
    
    return cupy_available

#CPU MODE IMAGE PROCESSING#
def process_image_cpu(img1, img2):
    """(numpy) OPERATION"""
    print("Using NumPy For CPU Processing")
    # numpy mode 
    img1_np = np.asarray(img1)
    img2_np = np.asarray(img2)
    
    # UNINT16#
    diff_np = np.abs(img1_np.astype(np.int16) - img2_np.astype(np.int16)).astype(np.uint16)
    
    return diff_np

#GPU MODE IMAGE PROCESSING#
def process_image_gpu_cupy(img1, img2):
    "" #GPU via CuPy#""
    try:
        import cupy as cp
        print("Using CuPy for GPU-based processing")
        
        # Convert numpy arrays to cupy arrays
        gpu_img1 = cp.asarray(img1)
        gpu_img2 = cp.asarray(img2)
        
        # Compute absolute difference on GPU with direct uint16 conversion
        gpu_diff = cp.abs(gpu_img1.astype(cp.int16) - gpu_img2.astype(cp.int16)).astype(cp.uint16)
        
        # Convert back to numpy array
        diff = cp.asnumpy(gpu_diff)
        return diff
    except ImportError:
        print("CuPy not available, falling back to CPU")
        return process_image_cpu(img1, img2)

#MAIN#
def main():
    # PARSE Arguments
    parser = argparse.ArgumentParser(description='Process images with CPU or GPU')
    parser.add_argument('--input', default='raw', help='Input folder path')
    parser.add_argument('--output', default='p1/newdiff1/', help='Output folder path')
    parser.add_argument('--reference', default='q001.tif', help='Reference image path')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--use_cpu', action='store_true', help='Use CPU processing (overrides --use_gpu)')
    parser.add_argument('--method', choices=['auto', 'cpu', 'cupy'], default='auto', 
                       help='Explicitly choose processing method (auto, cpu, cupy)')
    args = parser.parse_args()
    
    #GPU support####
    cupy_available = check_gpu_support()
    
    # check processing method###
    if args.method == 'cpu' or args.use_cpu:
        process_func = process_image_cpu
        print("Using CPU (NumPy) for processing")
    elif args.method == 'cupy':
        if cupy_available:
            process_func = process_image_gpu_cupy
            print("Using GPU (CuPy) for processing")
        else:
            process_func = process_image_cpu
            print("CuPy requested but not available, falling back to CPU (NumPy)")
    else:  # args.method == 'auto'
        use_gpu = args.use_gpu and not args.use_cpu and cupy_available
        
        if use_gpu:
            process_func = process_image_gpu_cupy
            print("Using GPU acceleration via CuPy")
        else:
            process_func = process_image_cpu
            if args.use_gpu and not args.use_cpu and not cupy_available:
                print("GPU acceleration requested but not available, falling back to CPU (NumPy)")
            else:
                print("Using CPU (NumPy) for processing")
    
    #Input folder
    folder_path = args.input
    
    #Base image# 
    img1 = cv2.imread(args.reference, cv2.IMREAD_UNCHANGED)
    
    #Output folder#
    output_folder = args.output
    os.makedirs(output_folder, exist_ok=True)
    
    # Check the image
    if img1 is not None:
        # Get sorted image paths
        image_paths = sorted([os.path.join(folder_path, filename) 
                             for filename in os.listdir(folder_path) 
                             if filename.lower().endswith(('.tif', '.tiff'))])
        
        total_images = len(image_paths)
        processed = 0
        
        for image_path in image_paths:
            processed += 1
            # Load unchanged bit depth#
            img2 = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            
            # Check img2
            if img2 is not None:
                try:
                    # Process the images
                    diff = process_func(img1, img2)
                    
                    # Get image path
                    file_name = os.path.splitext(os.path.basename(image_path))[0]
                    
                    # Save
                    result_file_name = file_name + "_result.tif"
                    result_file_path = os.path.join(output_folder, result_file_name)
                    
                    # Save 16-bit TIFF with original properties
                    io.imsave(result_file_path, diff, check_contrast=False)
                    
                    print(f"Processed {processed}/{total_images}: {os.path.basename(image_path)} saved as {result_file_name}")
                except Exception as e:
                    print(f"Error processing {os.path.basename(image_path)}: {str(e)}")
                    print("Falling back to CPU for this image")
                    diff = process_image_cpu(img1, img2)
                    
                    # Get image path
                    file_name = os.path.splitext(os.path.basename(image_path))[0]
                    
                    # Save
                    result_file_name = file_name + "_result.tif"
                    result_file_path = os.path.join(output_folder, result_file_name)
                    
                    # Save 16-bit TIFF with original properties
                    io.imsave(result_file_path, diff, check_contrast=False)
                    
                    print(f"Processed {processed}/{total_images}: {os.path.basename(image_path)} with CPU (NumPy) saved as {result_file_name}")
            else:
                print(f"Failed to load {os.path.basename(image_path)}")
        
        print(f"Processing complete. Processed {processed}/{total_images} images.")
    else:
        print("Failed to load the reference image (img1).")

if __name__ == "__main__":
    main()