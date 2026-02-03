import cv2
import numpy as np
import time
import os
import pickle
import zstandard as zstd  # Using zstandard
import argparse

def load_compressed_data(compressed_file):
    """Load compressed data from a zstd file."""
    try:
        start_time = time.time()
        with open(compressed_file, 'rb') as f:
            compressed_data = f.read()
            
        # Create a ZSTD decompressor
        dctx = zstd.ZstdDecompressor()
        
        # Decompress the data
        decompressed_data = dctx.decompress(compressed_data)
        
        # Unpickle the data
        compression_data = pickle.loads(decompressed_data)
        
        load_time = time.time() - start_time
        print(f"Loaded compressed data in {load_time:.2f} seconds")
        return compression_data, load_time
    except Exception as e:
        print(f"Error loading compressed file: {e}")
        return None, 0

def reconstruct_image(image_data, bit_depth=16):
    """
    Reconstruct an image from its compressed data with correct intensity scaling.
    
    Args:
        image_data: Dictionary containing image metadata and contour data
        bit_depth: Output bit depth (8 or 16)
    
    Returns:
        Reconstructed image with proper bit depth and intensity scaling
    """
    try:
        # Extract image metadata
        image_shape = image_data['image_shape']
        contours_data = image_data['contours_data']
        max_intensity = image_data.get('max_intensity', 255)
        
        # Create empty image with proper bit depth
        if bit_depth == 16:
            # For 16-bit output
            dtype = np.uint16
            max_bit_value = 65535
        else:
            # Default to 8-bit output
            dtype = np.uint8
            max_bit_value = 255
            
        reconstructed = np.zeros(image_shape, dtype=dtype)
        
        # Fill in the image with pixel values
        for contour in contours_data:
            row = contour['row_index']
            start = contour['x_start']
            end = contour['x_end']
            
            # Use quantized_values if available, otherwise use inside_pixel_values
            if 'quantized_values' in contour:
                pixel_values = contour['quantized_values']
            else:
                pixel_values = contour['inside_pixel_values']
            
            # Make sure we're within bounds and have correct number of values
            if row < image_shape[0] and start < image_shape[1] and end < image_shape[1]:
                if len(pixel_values) == (end - start + 1):
                    # Store pixel values in the image
                    reconstructed[row, start:end+1] = pixel_values
        
        # If original max intensity differs from 255, apply the reverse normalization
        if max_intensity != 255 and max_intensity > 0:
            # Convert back to original scale based on max_intensity
            # First convert to float for the calculation
            reconstructed_float = reconstructed.astype(np.float32)
            
            # Reverse the normalization that was applied during compression
            # Original values were multiplied by (255 / x_max)
            # So we multiply by (x_max / 255) to get back
            scaling_factor = max_intensity / 255.0
            
            # For 16-bit output, scale to full 16-bit range
            if bit_depth == 16:
                scaling_factor = max_intensity * (max_bit_value / 255.0) / 255.0
                
            reconstructed_float = reconstructed_float * scaling_factor
            
            # Clip to valid range and convert back to integer
            reconstructed = np.clip(reconstructed_float, 0, max_bit_value).astype(dtype)
        elif bit_depth == 16:
            # If no scaling needed but 16-bit output requested, scale from 8-bit to 16-bit range
            reconstructed = (reconstructed.astype(np.float32) * (max_bit_value / 255.0)).astype(dtype)
        
        return reconstructed
    except Exception as e:
        print(f"Error reconstructing image: {e}")
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Decompress images from contour quantized data.')
    parser.add_argument('-input', '--input', required=True, 
                        help='Input compressed file (.bin.zst)')
    parser.add_argument('-output', '--output', default='./reconstructed', 
                        help='Output folder for reconstructed images')
    parser.add_argument('-bits', '--bits', type=int, choices=[8, 16], default=16,
                        help='Output bit depth (8 or 16)')
    args = parser.parse_args()
    
    # Set parameters from arguments
    compressed_file = args.input
    output_folder = args.output
    bit_depth = args.bits
    
    print(f"Using {bit_depth}-bit output depth")
    
    # Verify input file extension
    if not compressed_file.endswith('.zst') and not compressed_file.endswith('.bin.zst'):
        print(f"Warning: Input file {compressed_file} doesn't have the .zst extension. It may not be a ZSTD-compressed file.")
    
    # Load compressed data
    compression_data, load_time = load_compressed_data(compressed_file)
    
    if compression_data is None:
        print("Failed to load compressed data. Exiting.")
        return
    
    # Extract metadata
    mode = compression_data.get('mode', 'unknown')
    value = compression_data.get('value', 'unknown')
    source_folder = compression_data.get('source_folder', 'unknown')
    images = compression_data['images']
    
    print(f"Decompressing {len(images)} images from {source_folder}")
    print(f"Compression mode: {mode}, value: {value}")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Decompress each image
    start_time = time.time()
    successful = 0
    
    for i, image_data in enumerate(images):
        try:
            # Get image metadata
            filename = image_data['filename']
            output_name = f"reconstructed_{filename}"
            output_path = os.path.join(output_folder, output_name)
            
            # Handle filenames without extensions
            if not os.path.splitext(output_path)[1]:
                output_path = f"{output_path}.tif"
            
            print(f"Reconstructing {filename} ({i+1}/{len(images)})")
            
            # Reconstruct the image with specified bit depth
            reconstructed = reconstruct_image(image_data, bit_depth)
            
            if reconstructed is None:
                print(f"  Failed to reconstruct {filename}")
                continue
            
            # Save the reconstructed image
            # For 16-bit images, using TIFF is better for preserving the bit depth
            if bit_depth == 16:
                # Ensure extension is .tif for 16-bit images
                base, ext = os.path.splitext(output_path)
                if ext.lower() not in ['.tif', '.tiff']:
                    output_path = f"{base}.tif"
                    
            cv2.imwrite(output_path, reconstructed)
            print(f"  Saved to {output_path} ({bit_depth}-bit)")
            successful += 1
            
        except Exception as e:
            print(f"Error processing image {i}: {e}")
    
    # Calculate decompression stats
    decompression_time = time.time() - start_time
    
    # Print results
    print(f"\nDecompression completed in {decompression_time:.2f} seconds")
    print(f"Successfully decompressed {successful}/{len(images)} images to {bit_depth}-bit depth")
    print(f"Total time (including loading): {decompression_time + load_time:.2f} seconds")
    
if __name__ == "__main__":
    main()
