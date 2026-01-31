# ROIX-Comp

<div align="center">

![ROIX Banner](https://github.com/user-attachments/assets/0ae6bd22-2022-4e2d-99e1-430a2f318337)

**Region-of-Interest Extraction and Compression Tool**

</div>

---

## 📖 Overview

**ROIX-Comp** is a specialized software tool designed for **Region of Interest (ROI) extraction and compression** in digital images and video content. It enables users to identify, isolate, and efficiently compress specific areas within visual data while maintaining high quality in critical regions.

### Architecture

ROIX-Comp is divided into **three primary modules**:

1. **Background Elimination for XCT Data** - Process X-ray Computed Tomography data using precomputed image differences
2. **Compression** - Apply variable compression techniques including error-bounded quantization
3. **Decompression** - Restore processed data with proper reconstruction of ROI areas

---

## ✨ Features

### 🎯 1. Background Elimination for XCT Data
Process X-ray Computed Tomography data using precomputed image differences to separate regions of interest from non-essential areas.

- **Reference Image Detection**: Automatically identify precomputed images with no object information
- **Efficient Subtraction**: Remove background noise using reference-based difference computation
- **GPU Acceleration**: Optional CUDA-based processing for faster computation

### 🗜️ 2. Advanced Compression
Apply variable compression techniques including error-bounded quantization to efficiently reduce data size while maintaining quality in important regions.

**Supported Compressors:**
- **Gzip** - General-purpose lossless compression
- **Zstandard (Zstd)** - Fast compression with excellent ratios
- **SZ3** - Error-bounded lossy compressor for scientific data
- **ZFP** - Compressed floating-point arrays

### 🔄 3. Efficient Decompression
Handle the restoration of processed data, ensuring proper reconstruction of ROI areas and efficient rendering of the final output.

- **Bit-depth Preservation**: Support for 8-bit, 16-bit, and 32-bit images
- **Lossless Reconstruction**: Perfect recovery for lossless compression modes
- **Batch Processing**: Decompress multiple images efficiently

---

### Minimum Requirements

## 🛠️ Software Requirements

### Operating System
- Up-to-date mainstream Linux distribution (Fedora, Ubuntu, or newer)

### Compiler and Build Tools
- **C++ Compiler**: GCC 13.2.1+ (C++ compliant)
- **Build System**: CMake 3.27.0-rc2 or newer

### Core Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| **NumPy** | 2.0+ | CPU-based numerical operations |
| **CuPy** | Latest | GPU-accelerated operations |
| **OpenCV** | 4.9.0+ | Image processing |
| **Zstandard** | 0.22+ | Compression backend |
| **Gzip** | 1.12+ | Compression backend |

### Optional Dependencies

| Library | Purpose |
|---------|---------|
| **LibPressio** | Integration with SZ3 and ZFP compressors |
| **CUDA Toolkit** | GPU acceleration support |
| **NVIDIA Driver** | GPU hardware support |

> **Note**: For LibPressio installation, refer to: https://github.com/robertu94/libpressio

> **Important**: Using the alternative Spack and LibPressio installation/deployment method will significantly increase setup time. However, execution and analysis performance remains unchanged.

---

## 📦 Installation

### Step 1: Install NVIDIA Driver (For GPU Support)

Install the appropriate NVIDIA driver for your GPU.
```bash
# Check if NVIDIA driver is installed
nvidia-smi

# If not installed, follow the official guide:
# https://www.nvidia.com/Download/index.aspx
```

### Step 2: Install CUDA Toolkit

Install CUDA toolkit compatible with your system.
```bash
# Download and install CUDA from:
# https://developer.nvidia.com/cuda-downloads

# Verify CUDA installation
nvcc --version
```

### Step 3: Install System Dependencies

**On Fedora/RHEL:**
```bash
sudo dnf install gcc gcc-c++ cmake python3 python3-pip \
                 opencv opencv-devel zstd zstd-devel gzip
```

**On Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install build-essential cmake python3 python3-pip \
                     libopencv-dev zstd libzstd-dev gzip
```

### Step 4: Python Environment Setup

The system utilizes:
- **CuPy** for GPU-accelerated operations
- **NumPy** for CPU-based operations
```bash
# Create virtual environment (recommended)
python3 -m venv roix-env
source roix-env/bin/activate

# Install NumPy
pip install numpy==2.0

# Install OpenCV
pip install opencv-python

# Install CuPy (match with your CUDA version)
# For CUDA 11.x:
pip install cupy-cuda11x

# For CUDA 12.x:
pip install cupy-cuda12x
```

> ⚠️ **Critical**: Ensure that CuPy is compiled against the same CUDA version installed on your system.

**Verify CuPy installation:**
```bash
python3 -c "import cupy as cp; print(cp.__version__); print('CUDA Version:', cp.cuda.runtime.runtimeGetVersion())"
```

### Step 5: Install LibPressio (Optional - For SZ3/ZFP Support)

For integration with SZ3 and ZFP compression libraries, LibPressio installation is required.
```bash
# Clone LibPressio repository
git clone https://github.com/robertu94/libpressio
cd libpressio

# Follow detailed installation instructions at:
# https://github.com/robertu94/libpressio
```

For additional LibPressio parameters and usage, refer to:
https://github.com/szcompressor/fz_tutorial/tree/master/exercises/1_basics

### Step 6: Clone ROIX-Comp
```bash
git clone https://github.com/yourusername/roix-comp.git
cd roix-comp
```

---

## 🚀 Usage

### Supported Image Formats

ROIX-Comp currently supports the following image format:

- **TIFF (.tif)** - Tagged Image File Format
- **Processing Mode**: Grayscale images only

---

## 📋 Module 1: Background Removal

In the background removal process, XCT datasets contain multiple images.

### Understanding the Workflow

**Step 1:** Identify the precomputed image that contains no information about the object.

**Step 2:** After identifying the precomputed image, it is referred to as the **reference image**, which will be used for the background removal process.

**Classification:**
- **Precomputed image** → **Reference image** (no object information)
- **Other TIFF images** → **Input images** (contain object information)

### Command Syntax

Execute the following command:
```bash
python precomp.py <input_directory> <output_directory> <reference_image> [--use_gpu]
```

### Command-Line Arguments

| Argument | Type | Meaning | Example |
|----------|------|---------|---------|
| `<input_directory>` | Required | The input directory path containing the images | `./input` |
| `<output_directory>` | Required | Directory path to output folder | `./output` |
| `<reference_image>` | Required | Specifies the path of the precomputed file for difference of TIFF images | `q001.tif` |
| `--use_gpu` | Optional | Specify the mode is set for GPU processing | `--use_gpu` |

> **Note**: For CPU processing, the default mode is used. Simply omit the `--use_gpu` flag.

### Usage Examples

**CPU Processing (Default):**
```bash
python precomp.py ./input ./output q001.tif
```

**GPU Processing:**
```bash
python precomp.py ./input ./output q001.tif --use_gpu
```

**Full Path Example:**
```bash
python precomp.py /home/user/xct_data/raw /home/user/xct_data/cleaned reference_q001.tif --use_gpu
```

### Output Files

After computing precomputed difference, it contains TIFF images with no background information. These TIFF images are further considered for compression.

---

## 📋 Module 2: Compression Script

To operate the compression script, perform the following steps:

### Input Requirements

**Input TIFF**: The compression scripts require input TIFF images on which further operations are performed for compression.

### Data Compressors

Data compressors used to evaluate the results are listed below:

- **Gzip** - General-purpose lossless compression
- **Zstd** - Zstandard fast compression
- **SZ3** - Error-bounded lossy compressor
- **ZFP** - Floating-point array compression

### Available Compression Scripts

For compression, we have three scripts:

| Script | Compressor | Type |
|--------|-----------|------|
| `roix_gzip_comp.py` | Gzip | Lossless |
| `roix_zstd_comp.py` | Zstandard | Lossless |
| `roix_libpressio_comp.py` | SZ3 / ZFP | Lossy/Lossless |

---

### 2.1 Gzip Compression
```bash
python roix_gzip_comp.py --input <input_folder> --output <output_folder> [-gpu]
```

This command processes all images in the specified input directory using GPU acceleration and saves the results to the output directory.

#### Command-Line Arguments

| Argument | Type | Meaning | Example |
|----------|------|---------|---------|
| `--input` | Required | Specifies the input directory path containing the images to be processed | `--input /home/amarjitsingh/imgrec/p1/newdiff1/` |
| `--output` | Required | Specifies the directory path where processed output files will be saved | `--output ./` |
| `-gpu` | Optional | Flag to enable GPU acceleration for processing. When omitted, processing will default to CPU | `-gpu` |

#### Examples

**CPU Processing:**
```bash
python roix_gzip_comp.py --input ./cleaned_images/ --output ./compressed/
```

**GPU Processing:**
```bash
python roix_gzip_comp.py --input ./cleaned_images/ --output ./compressed/ -gpu
```

---

### 2.2 Zstandard (Zstd) Compression
```bash
python roix_zstd_comp.py --input <input_folder> --output <output_folder> [-gpu]
```

This command processes all images in the specified input directory using GPU acceleration and saves the results to the output directory.

#### Command-Line Arguments

| Argument | Type | Meaning | Example |
|----------|------|---------|---------|
| `--input` | Required | Specifies the input directory path containing the images to be processed | `--input /home/amarjitsingh/imgrec/p1/newdiff1/` |
| `--output` | Required | Specifies the directory path where processed output files will be saved | `--output ./` |
| `-gpu` | Optional | Flag to enable GPU acceleration for processing. When omitted, processing will default to CPU | `-gpu` |

#### Examples

**CPU Processing:**
```bash
python roix_zstd_comp.py --input ./cleaned_images/ --output ./compressed/
```

**GPU Processing:**
```bash
python roix_zstd_comp.py --input ./cleaned_images/ --output ./compressed/ -gpu
```

---

### 2.3 LibPressio Compression (SZ3/ZFP)

**Script:** `roix_libpressio_comp.py` → for SZ3 and ZFP compressor

LibPressio API calls in `roix_libpressio_comp.py` are used to evaluate different HPC-based compressors. To evaluate different compressors like SZ3 and ZFP, you need to change the compressor ID in the LibPressio template.

#### Selecting Compressor

Edit the compressor ID in the script:
```python
"compressor_id": "sz3"  # For SZ3 compressor
```

or if you want to switch to ZFP compressor, use:
```python
"compressor_id": "zfp"  # For ZFP compressor
```

#### Command Syntax
```bash
python roix_libpressio_comp.py --input <input_folder> --output <output_folder> --error_bound <error-bounded> --use_gpu
```

This command processes all images in the specified input directory using GPU acceleration and saves the results to the output directory.

#### Command-Line Arguments

| Argument | Type | Meaning | Example |
|----------|------|---------|---------|
| `--input` | Required | Input directory path containing TIFF images | `--input ./cleaned/` |
| `--output` | Required | Output directory for compressed files | `--output ./compressed/` |
| `--error_bound` | Required | Error bound value for lossy compression | `--error_bound 1e-3` |
| `--use_gpu` | Optional | Enable GPU acceleration | `--use_gpu` |

#### Examples

**SZ3 Compression with GPU:**
```bash
python roix_libpressio_comp.py --input ./cleaned_images/ --output ./compressed/ --error_bound 1e-3 --use_gpu
```

**ZFP Compression (CPU):**
```bash
# First, change compressor_id to "zfp" in the script
python roix_libpressio_comp.py --input ./cleaned_images/ --output ./compressed/ --error_bound 1e-4
```

#### Additional Parameters

For additional parameters use, you can consider the following reference:
https://github.com/szcompressor/fz_tutorial/tree/master/exercises/1_basics

---

## 📋 Module 3: Decompression Script

To operate the decompression script, perform the following steps:

### Reconstruction of Images

The decompression module restores compressed data back to TIFF images with proper bit-depth preservation.

---

### 3.1 Gzip & Zstandard Decompression

**Scripts:**
- `roix_gzip_decomp.py` - For Gzip compressed files
- `roix_zstd_decomp.py` - For Zstandard compressed files

#### Command Syntax
```bash
python roix_zstd_decomp.py -input <compressed_file> -output <reconstructed_folder> -bits 16
```

This command decompresses the file `contour_quantized.bin.zst` and saves the resulting 16-bit grayscale images to the specified directory.

#### Decompression Command-Line Arguments

| Argument | Type | Meaning | Example |
|----------|------|---------|---------|
| `-input` | Required | Specifies the path to the compressed binary file that needs to be decompressed. Must be a file compressed using ROIX-Comp | `-input contour_quantized_lossless.bin.zst` |
| `-output` | Required | Specifies the directory where the decompressed image files will be saved. Directory will be created if it doesn't exist | `-output ./rh` |
| `-bits` | Required | Specifies the bit depth of the original images. Common values are 8, 16, or 32, depending on the source image format | `-bits 16` |

#### Examples

**Decompress Zstandard File (16-bit):**
```bash
python roix_zstd_decomp.py -input contour_quantized_lossless.bin.zst -output ./rh -bits 16
```

**Decompress Gzip File (8-bit):**
```bash
python roix_gzip_decomp.py -input compressed_data.bin.gz -output ./reconstructed/ -bits 8
```

---

### 3.2 LibPressio Decompression (SZ3/ZFP)

**Script:** `roix_libpressio_decomp.py`

To operate the decompression script, perform the following steps:

#### Command Syntax
```bash
python roix_libpressio_decomp.py -input <compressed_file> -output <reconstructed_folder> --bit_depth 8
```

or
```bash
python roix_libpressio_decomp.py -input <compressed_file> -output <reconstructed_folder> --bit_depth 16
```

This command decompresses the file `contour_quantized_.bin.sz` and saves the resulting grayscale images to the specified directory.

#### Decompression Command-Line Arguments

| Argument | Type | Meaning | Example |
|----------|------|---------|---------|
| `-input` | Required | Specifies the path to the compressed binary file that needs to be decompressed. Must be a file compressed using ROIX-Comp | `-input contour_quantized_lossless.bin.sz` |
| `-output` | Required | Specifies the directory where the decompressed image files will be saved. Directory will be created if it doesn't exist | `-output ./rh` |
| `--bit_depth` | Required | Specifies the bit depth of the original images. Common values are 8 or 16, depending on the source image format | `--bit_depth 8` or `--bit_depth 16` |

#### Examples

**Decompress SZ3 File (16-bit):**
```bash
python roix_libpressio_decomp.py -input contour_quantized_lossless.bin.sz -output ./rh --bit_depth 16
```

**Decompress ZFP File (8-bit):**
```bash
python roix_libpressio_decomp.py -input compressed_data.bin.zfp -output ./reconstructed/ --bit_depth 8
```

---

## 🔄 Complete Workflow Examples

### Example 1: Full Pipeline with Zstandard Compression
```bash
# Step 1: Remove background from XCT images (GPU mode)
python precomp.py ./raw_xct_data/ ./cleaned_data/ reference_q001.tif --use_gpu

# Step 2: Compress cleaned images using Zstandard (GPU mode)
python roix_zstd_comp.py --input ./cleaned_data/ --output ./compressed/ -gpu

# Step 3: Decompress for analysis (16-bit images)
python roix_zstd_decomp.py -input ./compressed/contour_quantized.bin.zst \
                           -output ./reconstructed/ -bits 16
```

### Example 2: Scientific Data with SZ3 (Error-Bounded Compression)
```bash
# Step 1: Background removal (GPU accelerated)
python precomp.py /data/xct/raw /data/xct/cleaned q001.tif --use_gpu

# Step 2: Compress with SZ3 and error bound
# (First, ensure compressor_id is set to "sz3" in roix_libpressio_comp.py)
python roix_libpressio_comp.py --input /data/xct/cleaned \
                               --output /data/xct/compressed \
                               --error_bound 1e-4 --use_gpu

# Step 3: Reconstruct images
python roix_libpressio_decomp.py -input /data/xct/compressed/output.bin.sz \
                                 -output /data/xct/reconstructed \
                                 --bit_depth 16
```

### Example 3: CPU-Only Processing with Gzip
```bash
# Step 1: Background removal (CPU mode - default)
python precomp.py ./input ./cleaned ref.tif

# Step 2: Gzip compression (CPU mode)
python roix_gzip_comp.py --input ./cleaned --output ./compressed

# Step 3: Decompress
python roix_gzip_decomp.py -input ./compressed/output.bin.gz \
                           -output ./final -bits 16
```

### Example 4: High-Quality Processing with ZFP
```bash
# Step 1: Background removal
python precomp.py ./xct_scans/ ./cleaned/ reference.tif --use_gpu

# Step 2: ZFP compression
# (Change compressor_id to "zfp" in roix_libpressio_comp.py)
python roix_libpressio_comp.py --input ./cleaned/ \
                               --output ./compressed/ \
                               --error_bound 1e-5 --use_gpu

# Step 3: Decompression
python roix_libpressio_decomp.py -input ./compressed/output.bin.zfp \
                                 -output ./restored/ --bit_depth 16
```

---
