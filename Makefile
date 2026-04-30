COMPILER=nvcc
TARGET=cuda_image_pipeline
SRC=main.cu
FLAGS=--std c++17 -Wno-deprecated-gpu-targets

.PHONY: clean build run

build:
	$(COMPILER) $(FLAGS) $(SRC) -o $(TARGET)

run:
	./$(TARGET) --input_dir input --output_dir output --num_images 256 --width 256 --height 256 --generate 1

clean:
	rm -f $(TARGET)
	rm -rf input output logs
