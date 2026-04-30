#!/usr/bin/env bash
set -e

make clean build

./cuda_image_pipeline \
  --input_dir input \
  --output_dir output \
  --num_images 256 \
  --width 256 \
  --height 256 \
  --generate 1
