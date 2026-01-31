ROIX-Comp is a specialized software tool designed for Region of Interest (ROI) extraction and compression in digital images and video content. It enables users to identify, isolate, and efficiently compress specific areas within visual data while maintaining high quality in critical regions.
1. Background Elimination for XCT Data
The background elimination module processes X-ray Computed Tomography data using precomputed image differences to separate regions of interest from non-essential areas.

2. Compression
The compression module applies variable compression techniques including error-bounded quantization to efficiently reduce data size while maintaining quality in important regions.

3. Decompression
The decompression module handles the restoration of processed data, ensuring proper reconstruction of ROI areas and efficient rendering of the final output.

Software Requirements
Operating System
Up-to-date mainstream Linux distribution (e.g., Fedora or newer)

Compiler and Build System
C++ compliant host compiler (tested on GCC 13.2.1)

Modern CMake build system (e.g., 3.27.0-rc2 or newer)

Dependencies
The following libraries are required:

NumPy 2.0

OpenCV 4.9.0

Zstandard 0.22

Gzip 1.12

Compression Library Integration
For integration with SZ and ZFP compression libraries:

LibPressio installation is required

For detailed instructions, refer to: https://github.com/robertu94/libpressio


