# CUDA Batch Image Processing Pipeline

## Project Description

This project demonstrates GPU-accelerated batch image processing using custom CUDA kernels. The program generates or reads a batch of grayscale PGM images, processes every image on the GPU, and writes both processed output images and an execution log.

The submitted configuration processes **256 grayscale images** of size **256x256**, satisfying the requirement to process hundreds of small pieces of image data.

## GPU Computation

The project uses two CUDA kernels:

1. **Box Blur Kernel**
   - Applies a 3x3 neighborhood average to smooth each image.
   - Each CUDA thread processes one output pixel.

2. **Sobel Edge Detection Kernel**
   - Computes approximate horizontal and vertical image gradients.
   - Produces an edge map showing sharp transitions in the image.
   - Each CUDA thread processes one output pixel.

The CPU is responsible for generating or loading PGM images and writing outputs. The actual image-processing operations are performed on the GPU.

## Repository Structure

```text
.
├── main.cu              # CUDA/C++ source code
├── Makefile             # Build and run targets
├── run.sh               # Reproducible execution script
├── input/               # Generated input PGM images after running
├── output/              # Processed output images after running
│   ├── blur/            # Box blur outputs
│   └── edge/            # Sobel edge outputs
└── logs/                # Execution log after running
```

## Build

```bash
make clean build
```

## Run

Recommended:

```bash
./run.sh
```

Manual equivalent:

```bash
./cuda_image_pipeline \
  --input_dir input \
  --output_dir output \
  --num_images 256 \
  --width 256 \
  --height 256 \
  --generate 1
```

## CLI Arguments

```text
--input_dir      Directory containing input PGM images
--output_dir     Directory where processed images are written
--num_images     Number of images to process
--width          Width of generated/read images
--height         Height of generated/read images
--generate       1 = generate synthetic PGM images, 0 = read existing PGM images
```

## Outputs

After execution, the project writes:

```text
input/image_0000.pgm
output/blur/image_0000_blur.pgm
output/edge/image_0000_edge.pgm
logs/execution_log.csv
```

The execution log contains one row per processed image:

```text
image_id,width,height,blur_ms,edge_ms,total_ms
```

This log is the main proof that the program processed many images in one execution.

## Observations

The box blur kernel smooths the image by averaging neighboring pixels. The Sobel edge kernel highlights boundaries and sharp transitions after the blur stage. Larger batches demonstrate how the same GPU pipeline can be applied repeatedly across many independent image inputs.

## Lessons Learned

This project demonstrates the basic CUDA image-processing workflow:

1. Prepare image data on the CPU.
2. Allocate GPU memory with `cudaMalloc`.
3. Copy image data from host to device using `cudaMemcpy`.
4. Launch CUDA kernels with one thread per pixel.
5. Copy processed results back to the CPU.
6. Save output artifacts and kernel timing logs.

The main challenge was keeping the implementation simple enough to run in the Coursera lab while still proving meaningful GPU computation over a batch of image data.
