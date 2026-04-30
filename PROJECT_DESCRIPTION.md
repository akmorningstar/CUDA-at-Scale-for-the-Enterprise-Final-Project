# Short Project Description

I developed a CUDA-based batch image processing pipeline that generates and processes 256 grayscale PGM images. The program applies two GPU kernels to each image: a 3x3 box blur kernel and a Sobel edge-detection kernel. The pipeline demonstrates host-to-device memory transfer, GPU kernel execution, device-to-host result transfer, and batch output generation.

Execution evidence is produced through processed output images and a CSV log containing per-image GPU kernel timings. The repository includes source code, a README, Makefile, run script, generated sample inputs after execution, processed outputs after execution, and logs.
