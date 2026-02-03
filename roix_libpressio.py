#!/usr/bin/env python3
import os
import argparse
import json
import time
import numpy as np
import pickle
from pathlib import Path
import libpressio as lp
from skimage.filters import threshold_multiotsu
import cv2


# Check for GPU support
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("CUDA not available. Will use CPU only.")


def load_image(image_path, use_gpu=False):
    """Load an image from a path and normalize it."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to load image {image_path}.")
        return None, None, None, None
        
    x_max = np.max(image)
    if x_max > 0:
        img_nor = image * (255 / x_max)
        img_nor = np.clip(img_nor, 0, 255).astype(np.uint8)
    else:
        img_nor = image
        
    image_shape = image.shape
    
    # Convert to float32 for compression
    data = image.astype(np.float32)
    
    # Move to GPU if available and requested
    if use_gpu and HAS_GPU:
        data = cp.array(data)
        
    return data, img_nor, x_max, image_shape

def multi_otsu_and_binary(img_nor, classes=3):
    """Apply Multi-Otsu Thresholding and binarize."""
    th = threshold_multiotsu(img_nor, classes=classes)
    bin_img = np.zeros_like(img_nor)
    bin_img[img_nor > th[0]] = 255
    return bin_img, th

def extract_contour_data(orig_img, bin_img):
    """Extract contour data using your method - with proper contour mask."""
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contour_data = []
    
    for contour_idx, contour in enumerate(contours):
        # Create mask for this specific contour
        contour_mask = np.zeros_like(bin_img)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        
        # Extract the contour data using your method
        x, y, w, h = cv2.boundingRect(contour)
        contour_data = []
        
        for i in range(y, y + h):
            row_slice = contour_mask[i, x:x + w]
            xs = np.where(row_slice == 255)[0] + x
            if xs.size > 0:
                row_x_start = xs.min()
                row_x_end = xs.max()
                pixel_values = orig_img[i, row_x_start:row_x_end + 1]
                contour_data.append({
                    'row_index': i,
                    'x_start': row_x_start,
                    'x_end': row_x_end,
                    'pixel_length': len(pixel_values),  # Store length for validation
                    'pixel_values': pixel_values.astype(np.float32)  # Convert to float32 for compression
                })
        
        if contour_data:
            all_contour_data.extend(contour_data)
    
    return all_contour_data

def compress_data_with_contours(all_pixel_values, all_contours_metadata, error_bound=1e-6):
    """
    Compress pixel values using SZ3 and store with contour metadata.
    
    Args:
        all_pixel_values: Concatenated pixel values from all contours
        all_contours_metadata: Metadata about contour positions (without pixel values)
        error_bound: Error bound for SZ3 compression
    """
    # Convert to CPU array if needed
    if HAS_GPU and isinstance(all_pixel_values, cp.ndarray):
        combined_data = cp.asnumpy(all_pixel_values)
    else:
        combined_data = all_pixel_values
    
    print(f"Compressing {len(combined_data)} pixel values with error bound: {error_bound:1.0e}")
    
    # Configure SZ3 compressor
    compressor = lp.PressioCompressor.from_config({
        "compressor_id": "sz3",
        "early_config": {
            "pressio:metric": "composite",
            "composite:plugins": ["time", "size", "error_stat"]
        },
        "compressor_config": {
            "pressio:abs": error_bound,
        }
    })
    
    # Compress the pixel values
    comp_data = compressor.encode(combined_data)
    
    # Test decompression to get metrics
    decomp_data = compressor.decode(comp_data, np.zeros_like(combined_data))
    metrics = compressor.get_metrics()
    print(f"Compression metrics: {json.dumps(metrics, indent=2)}")
    
    # Create compression package
    compression_package = {
        'compressed_data': comp_data,         # SZ3 compressed pixel values
        'contours_metadata': all_contours_metadata,  # Position info without pixel values
        'compression_params': {
            'method': 'sz3',
            'error_bound': float(error_bound),
            'mode': 'contour_range_based',
            'original_data_size': len(combined_data)
        }
    }
    
    # Save compressed package
    output_file = f"contour_compressed_eb{error_bound:1.0e}.bin.sz"
    with open(output_file, 'wb') as f:
        pickle.dump(compression_package, f)
    
    print(f"Compressed package saved to {output_file}")
    
    return [{
        'bound': error_bound,
        'output_file': output_file,
        'compression_ratio': len(combined_data) * 4 / len(comp_data)  # Approximate ratio
    }]

def main():
    parser = argparse.ArgumentParser(description='Compress images using contour-based approach with SZ3.')
    parser.add_argument('input_path', help='Path to input directory or single image')
    parser.add_argument('--use_gpu', '-g', action='store_true', help='Use GPU acceleration if available')
    parser.add_argument('--output_dir', '-o', default='./', help='Output directory for compressed files')
    parser.add_argument('--error_bound', '-e', type=float, default=1e-6, 
                        help='Error bound to use for SZ3 compression (e.g., 1e-2, 1e-3, 1e-4)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)
    
    print("Contour-based Image Compression with SZ3")
    print("=" * 40)
    
    # Find image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    image_files = []
    
    if os.path.isdir(args.input_path):
        for ext in image_extensions:
            image_files.extend(list(Path(args.input_path).glob(f'*{ext}')))
    else:
        if any(args.input_path.lower().endswith(ext) for ext in image_extensions):
            image_files = [Path(args.input_path)]
        else:
            print(f"Input path {args.input_path} is not a directory or an image file.")
            return
    
    if not image_files:
        print(f"No image files found in {args.input_path}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Process all images
    all_pixel_values = []
    all_contours_metadata = []
    
    for i, image_path in enumerate(image_files):
        print(f"\nProcessing image {i+1}/{len(image_files)}: {image_path.name}")
        
        # Load image
        data, img_nor, x_max, image_shape = load_image(str(image_path), args.use_gpu)
        
        if data is None:
            print(f"  Skipping: Failed to load")
            continue
        
        # Apply thresholding to get binary image
        bin_img, _ = multi_otsu_and_binary(img_nor)
        
        # Extract contour data using your method
        contour_data = extract_contour_data(img_nor, bin_img)
        
        if not contour_data:
            print(f"  No contours found")
            continue
        
        print(f"  Found {len(contour_data)} contour segments")
        
        # Separate pixel values from metadata
        image_pixel_values = []
        image_contour_metadata = []
        
        for segment in contour_data:
            # Extract pixel values for compression
            pixel_values = segment['pixel_values']
            image_pixel_values.extend(pixel_values)
            
            # Store metadata without pixel values
            metadata = {
                'row_index': segment['row_index'],
                'x_start': segment['x_start'],
                'x_end': segment['x_end'],
                'pixel_length': segment['pixel_length']
            }
            image_contour_metadata.append(metadata)
        
        # Add to global collections
        all_pixel_values.extend(image_pixel_values)
        
        # Store image metadata
        image_metadata = {
            'filename': image_path.name,
            'max_intensity': float(x_max),
            'image_shape': image_shape,
            'contour_segments': image_contour_metadata,
            'total_pixels': len(image_pixel_values)
        }
        all_contours_metadata.append(image_metadata)
        
        print(f"  Extracted {len(image_pixel_values)} pixels from contours")
    
    if not all_pixel_values:
        print("No contour data extracted. Exiting.")
        return
    
    # Convert to numpy array for compression
    combined_data = np.array(all_pixel_values, dtype=np.float32)
    print(f"\nTotal pixels to compress: {len(combined_data)}")
    
    # Save metadata file
    metadata = {
        'num_images': len(image_files),
        'input_path': args.input_path,
        'mode': 'contour_range_based',
        'compression_method': 'libpressio_sz3',
        'error_bound': args.error_bound,
        'total_pixels': len(combined_data),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('compression_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # Compress data
    print(f"\nCompressing with SZ3 (error_bound={args.error_bound:g})...")
    results = compress_data_with_contours(combined_data, all_contours_metadata, args.error_bound)
    
    print(f"\nCompression complete!")
    for result in results:
        print(f"  Output file: {result['output_file']}")
        print(f"  Compression ratio: {result['compression_ratio']:.2f}:1")

if __name__ == "__main__":
    main()
