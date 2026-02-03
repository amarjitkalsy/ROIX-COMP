#!/usr/bin/env python3
import os
import argparse
import pickle
import numpy as np
import cv2
import time
from pathlib import Path

# Try to import libpressio for decompression
try:
    import libpressio as lp
    LIBPRESSIO_AVAILABLE = True
except ImportError:
    LIBPRESSIO_AVAILABLE = False
    print("Warning: libpressio not available. Will use pattern-based placeholders.")

def load_compressed_package(compressed_file):
    """Load compressed package from file."""
    try:
        start_time = time.time()
        with open(compressed_file, 'rb') as f:
            compression_package = pickle.load(f)
        
        load_time = time.time() - start_time
        print(f"Loaded compressed package in {load_time:.2f} seconds")
        
        # Validate package structure
        required_keys = ['compressed_data', 'contours_metadata', 'compression_params']
        for key in required_keys:
            if key not in compression_package:
                print(f"Warning: Missing '{key}' in compressed package")
        
        return compression_package, load_time
    except Exception as e:
        print(f"Error loading compressed file: {e}")
        return None, 0

def decompress_pixel_values(compression_package):
    """Decompress pixel values using LibPressio SZ3 or create placeholders."""
    try:
        comp_data = compression_package['compressed_data']
        compression_params = compression_package['compression_params']
        original_size = compression_params.get('original_data_size', 0)
        error_bound = compression_params.get('error_bound', 1e-6)
        
        if LIBPRESSIO_AVAILABLE:
            print(f"Decompressing {original_size} pixels with LibPressio SZ3 (error_bound={error_bound})")
            
            # Configure SZ3 decompressor
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
            
            # Create buffer for decompressed data
            decomp_data = np.zeros(original_size, dtype=np.float32)
            
            # Decompress
            decomp_data = compressor.decode(comp_data, decomp_data)
            
            print(f"Successfully decompressed {len(decomp_data)} pixel values")
            return decomp_data
        else:
            print(f"LibPressio not available. Creating {original_size} placeholder values")
            
            # Create pattern-based placeholder data
            np.random.seed(42)  # For reproducible patterns
            placeholder_data = np.random.randint(50, 200, size=original_size).astype(np.float32)
            
            print("Using pattern-based placeholders for visualization")
            return placeholder_data
            
    except Exception as e:
        print(f"Error in decompression: {e}")
        return None

def reconstruct_image_from_contours(decompressed_pixels, image_metadata, bit_depth=16):
    """Reconstruct image using your method with decompressed pixel values."""
    try:
        # Extract image information
        image_shape = tuple(image_metadata['image_shape'])
        contour_segments = image_metadata['contour_segments']
        max_intensity = image_metadata.get('max_intensity', 255)
        filename = image_metadata.get('filename', 'unknown')
        
        print(f"  Reconstructing {filename}: {len(contour_segments)} segments, shape {image_shape}")
        
        # Create empty image with proper bit depth
        if bit_depth == 16:
            dtype = np.uint16
            max_bit_value = 65535
        else:
            dtype = np.uint8
            max_bit_value = 255
        
        reconstructed = np.zeros(image_shape, dtype=dtype)
        
        # Track position in decompressed pixel array
        pixel_index = 0
        
        # Reconstruct using your method
        for segment in contour_segments:
            row = segment['row_index']
            start = segment['x_start']
            end = segment['x_end']
            pixel_length = segment['pixel_length']
            
            # Validate segment
            expected_length = end - start + 1
            if pixel_length != expected_length:
                print(f"    Warning: Length mismatch at row {row}: expected {expected_length}, got {pixel_length}")
                pixel_length = expected_length  # Use expected length
            
            # Extract corresponding pixel values from decompressed data
            if pixel_index + pixel_length <= len(decompressed_pixels):
                pixel_values = decompressed_pixels[pixel_index:pixel_index + pixel_length]
                pixel_index += pixel_length
            else:
                print(f"    Warning: Not enough decompressed data for segment at row {row}")
                break
            
            # Place pixels back in image (your reconstruction method)
            if (0 <= row < image_shape[0] and 
                0 <= start < image_shape[1] and 
                end < image_shape[1] and
                len(pixel_values) == (end - start + 1)):
                
                reconstructed[row, start:end+1] = pixel_values.astype(dtype)
        
        # Apply intensity scaling to restore original dynamic range
        if max_intensity != 255 and max_intensity > 0:
            # Scale back to original intensity range
            scaling_factor = max_intensity / 255.0
            reconstructed_float = reconstructed.astype(np.float32) * scaling_factor
            
            # Scale to target bit depth
            if bit_depth == 16:
                reconstructed_float = reconstructed_float * (max_bit_value / 255.0)
            
            reconstructed = np.clip(reconstructed_float, 0, max_bit_value).astype(dtype)
        elif bit_depth == 16 and np.max(reconstructed) <= 255:
            # Scale 8-bit values to 16-bit range
            reconstructed = (reconstructed.astype(np.float32) * (max_bit_value / 255.0)).astype(dtype)
        
        return reconstructed
        
    except Exception as e:
        print(f"    Error reconstructing image: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='Decompress contour-compressed images.')
    parser.add_argument('input_path', help='Input compressed package file (.bin.sz)')
    parser.add_argument('--output_dir', '-o', default='./recon', 
                       help='Output directory for reconstructed images')
    parser.add_argument('--sample', '-s', action='store_true', 
                       help='Process only a sample of images')
    parser.add_argument('--bit_depth', '-b', type=int, choices=[8, 16], default=16,
                       help='Output bit depth (8 or 16)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Contour Decompression Tool")
    print("=" * 30)
    print(f"Output format: {args.bit_depth}-bit TIFF")
    
    # Check input file
    if not os.path.exists(args.input_path):
        print(f"Input file not found: {args.input_path}")
        return
    
    print(f"Processing: {args.input_path}")
    
    # Load compressed package
    compression_package, load_time = load_compressed_package(args.input_path)
    
    if compression_package is None:
        print("Failed to load compressed package.")
        return
    
    # Extract package contents
    contours_metadata = compression_package.get('contours_metadata', [])
    compression_params = compression_package.get('compression_params', {})
    
    print(f"Compression parameters: {compression_params}")
    print(f"Package contains {len(contours_metadata)} images")
    
    # Decompress pixel values
    print("\nDecompressing pixel values...")
    start_time = time.time()
    
    decompressed_pixels = decompress_pixel_values(compression_package)
    
    if decompressed_pixels is None:
        print("Failed to decompress pixel values.")
        return
    
    decomp_time = time.time() - start_time
    print(f"Decompressed {len(decompressed_pixels)} pixels in {decomp_time:.2f} seconds")
    
    # Determine which images to process
    if args.sample and len(contours_metadata) > 10:
        print(f"\nProcessing sample images...")
        # Select first, last, and some middle images
        sample_indices = [0]  # First
        step = max(1, len(contours_metadata) // 8)
        for i in range(step, len(contours_metadata), step):
            sample_indices.append(i)
        sample_indices.append(len(contours_metadata) - 1)  # Last
        sample_indices = sorted(list(set(sample_indices)))
        image_indices = sample_indices
        print(f"Selected {len(image_indices)} sample images")
    else:
        image_indices = range(len(contours_metadata))
    
    # Reconstruct images
    print(f"\nReconstructing images...")
    successful = 0
    reconstruction_start = time.time()
    pixel_offset = 0
    
    for idx in image_indices:
        if idx >= len(contours_metadata):
            continue
            
        image_metadata = contours_metadata[idx]
        
        try:
            # Calculate pixel range for this image
            total_pixels = image_metadata.get('total_pixels', 0)
            
            # Extract pixel values for this specific image
            if pixel_offset + total_pixels <= len(decompressed_pixels):
                image_pixels = decompressed_pixels[pixel_offset:pixel_offset + total_pixels]
            else:
                print(f"  Warning: Not enough pixels for image {idx+1}")
                continue
            
            # Generate output filename
            orig_filename = image_metadata.get('filename', f'image_{idx+1:04d}')
            base_name = os.path.splitext(orig_filename)[0]
            output_filename = f"reconstructed_{base_name}.tif"
            output_path = os.path.join(args.output_dir, output_filename)
            
            print(f"Reconstructing image {idx+1}: {orig_filename}")
            
            # Reconstruct the image
            reconstructed = reconstruct_image_from_contours(
                image_pixels, image_metadata, args.bit_depth
            )
            
            if reconstructed is not None:
                # Save reconstructed image
                success = cv2.imwrite(output_path, reconstructed)
                if success:
                    print(f"  Saved: {output_path}")
                    successful += 1
                else:
                    print(f"  Failed to save: {output_path}")
            
            # Update pixel offset for next image
            pixel_offset += total_pixels
            
        except Exception as e:
            print(f"  Error processing image {idx+1}: {e}")
    
    # Summary
    total_time = time.time() - reconstruction_start
    print(f"\nReconstruction Summary:")
    print(f"  Successfully reconstructed: {successful}/{len(image_indices)} images")
    print(f"  Processing time: {total_time:.2f} seconds")
    print(f"  Output directory: {args.output_dir}")
    
    if not LIBPRESSIO_AVAILABLE:
        print(f"\nNote: LibPressio was not available.")
        print(f"      Images show contour structure with placeholder pixel values.")
    else:
        print(f"\nNote: Images reconstructed using SZ3 decompressed pixel values.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()