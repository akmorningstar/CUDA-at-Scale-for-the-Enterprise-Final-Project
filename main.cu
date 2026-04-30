#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Config {
    std::string inputDir = "input";
    std::string outputDir = "output";
    int numImages = 256;
    int width = 256;
    int height = 256;
    bool generate = true;
};

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t status = (call);                                           \
        if (status != cudaSuccess) {                                           \
            std::cerr << "CUDA error: " << cudaGetErrorString(status)         \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

static void makeDirectory(const std::string &path) {
    std::string command = "mkdir -p \"" + path + "\"";
    int rc = std::system(command.c_str());
    if (rc != 0) {
        throw std::runtime_error("Failed to create directory: " + path);
    }
}

static std::string imagePath(const std::string &dir, int index, const std::string &suffix = "") {
    std::ostringstream oss;
    oss << dir << "/image_" << std::setw(4) << std::setfill('0') << index << suffix << ".pgm";
    return oss.str();
}

static Config parseArguments(int argc, char **argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto requireValue = [&](const std::string &name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for argument: " + name);
            }
            return argv[++i];
        };

        if (arg == "--input_dir") {
            cfg.inputDir = requireValue(arg);
        } else if (arg == "--output_dir") {
            cfg.outputDir = requireValue(arg);
        } else if (arg == "--num_images") {
            cfg.numImages = std::stoi(requireValue(arg));
        } else if (arg == "--width") {
            cfg.width = std::stoi(requireValue(arg));
        } else if (arg == "--height") {
            cfg.height = std::stoi(requireValue(arg));
        } else if (arg == "--generate") {
            cfg.generate = std::stoi(requireValue(arg)) != 0;
        } else if (arg == "--help") {
            std::cout << "Usage: ./cuda_image_pipeline "
                      << "--input_dir input --output_dir output "
                      << "--num_images 256 --width 256 --height 256 --generate 1\n";
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (cfg.numImages <= 0 || cfg.width <= 0 || cfg.height <= 0) {
        throw std::runtime_error("num_images, width, and height must be positive");
    }
    return cfg;
}

static std::vector<unsigned char> generateImage(int width, int height, int imageIndex) {
    std::vector<unsigned char> pixels(width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int gradient = (x * 255) / std::max(1, width - 1);
            int stripe = ((x / 16 + y / 16 + imageIndex) % 2) ? 70 : 0;
            int circle = ((x - width / 2) * (x - width / 2) + (y - height / 2) * (y - height / 2))
                         < ((width / 5 + imageIndex % 17) * (height / 5 + imageIndex % 13))
                         ? 80 : 0;
            int value = (gradient + stripe + circle + imageIndex * 3) % 256;
            pixels[y * width + x] = static_cast<unsigned char>(value);
        }
    }
    return pixels;
}

static void writePgm(const std::string &path, const std::vector<unsigned char> &pixels, int width, int height) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Could not open output image: " + path);
    }
    out << "P5\n" << width << " " << height << "\n255\n";
    out.write(reinterpret_cast<const char *>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
}

static std::vector<unsigned char> readPgm(const std::string &path, int expectedWidth, int expectedHeight) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Could not open input image: " + path);
    }

    std::string magic;
    int width = 0;
    int height = 0;
    int maxValue = 0;
    in >> magic >> width >> height >> maxValue;
    in.get(); // consume one newline after header

    if (magic != "P5" || maxValue != 255) {
        throw std::runtime_error("Only binary P5 PGM images with max value 255 are supported: " + path);
    }
    if (width != expectedWidth || height != expectedHeight) {
        throw std::runtime_error("Input image dimensions do not match requested width/height: " + path);
    }

    std::vector<unsigned char> pixels(width * height);
    in.read(reinterpret_cast<char *>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
    if (!in) {
        throw std::runtime_error("Could not read full image payload: " + path);
    }
    return pixels;
}

__global__ void boxBlurKernel(const unsigned char *input, unsigned char *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int sum = 0;
    int count = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                sum += input[ny * width + nx];
                ++count;
            }
        }
    }
    output[y * width + x] = static_cast<unsigned char>(sum / count);
}

__global__ void sobelEdgeKernel(const unsigned char *input, unsigned char *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        output[y * width + x] = 0;
        return;
    }

    int gx = -input[(y - 1) * width + (x - 1)] - 2 * input[y * width + (x - 1)] - input[(y + 1) * width + (x - 1)]
             + input[(y - 1) * width + (x + 1)] + 2 * input[y * width + (x + 1)] + input[(y + 1) * width + (x + 1)];

    int gy = -input[(y - 1) * width + (x - 1)] - 2 * input[(y - 1) * width + x] - input[(y - 1) * width + (x + 1)]
             + input[(y + 1) * width + (x - 1)] + 2 * input[(y + 1) * width + x] + input[(y + 1) * width + (x + 1)];

    int magnitude = abs(gx) + abs(gy);
    output[y * width + x] = static_cast<unsigned char>(min(255, magnitude));
}

static float elapsedMs(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

static void processImageOnGpu(const std::vector<unsigned char> &input,
                              std::vector<unsigned char> &blurred,
                              std::vector<unsigned char> &edges,
                              int width,
                              int height,
                              float &blurMs,
                              float &edgeMs) {
    size_t bytes = input.size() * sizeof(unsigned char);

    unsigned char *dInput = nullptr;
    unsigned char *dBlur = nullptr;
    unsigned char *dEdge = nullptr;
    CUDA_CHECK(cudaMalloc(&dInput, bytes));
    CUDA_CHECK(cudaMalloc(&dBlur, bytes));
    CUDA_CHECK(cudaMalloc(&dEdge, bytes));
    CUDA_CHECK(cudaMemcpy(dInput, input.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    cudaEvent_t blurStart, blurStop, edgeStart, edgeStop;
    CUDA_CHECK(cudaEventCreate(&blurStart));
    CUDA_CHECK(cudaEventCreate(&blurStop));
    CUDA_CHECK(cudaEventCreate(&edgeStart));
    CUDA_CHECK(cudaEventCreate(&edgeStop));

    CUDA_CHECK(cudaEventRecord(blurStart));
    boxBlurKernel<<<grid, block>>>(dInput, dBlur, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(blurStop));
    CUDA_CHECK(cudaEventSynchronize(blurStop));
    blurMs = elapsedMs(blurStart, blurStop);

    CUDA_CHECK(cudaEventRecord(edgeStart));
    sobelEdgeKernel<<<grid, block>>>(dBlur, dEdge, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(edgeStop));
    CUDA_CHECK(cudaEventSynchronize(edgeStop));
    edgeMs = elapsedMs(edgeStart, edgeStop);

    CUDA_CHECK(cudaMemcpy(blurred.data(), dBlur, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(edges.data(), dEdge, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(blurStart));
    CUDA_CHECK(cudaEventDestroy(blurStop));
    CUDA_CHECK(cudaEventDestroy(edgeStart));
    CUDA_CHECK(cudaEventDestroy(edgeStop));
    CUDA_CHECK(cudaFree(dInput));
    CUDA_CHECK(cudaFree(dBlur));
    CUDA_CHECK(cudaFree(dEdge));
}

int main(int argc, char **argv) {
    try {
        Config cfg = parseArguments(argc, argv);
        makeDirectory(cfg.inputDir);
        makeDirectory(cfg.outputDir);
        makeDirectory(cfg.outputDir + "/blur");
        makeDirectory(cfg.outputDir + "/edge");
        makeDirectory("logs");

        std::ofstream log("logs/execution_log.csv");
        if (!log) {
            throw std::runtime_error("Could not open logs/execution_log.csv");
        }
        log << "image_id,width,height,blur_ms,edge_ms,total_ms\n";

        double totalBlur = 0.0;
        double totalEdge = 0.0;

        for (int i = 0; i < cfg.numImages; ++i) {
            std::string inputPath = imagePath(cfg.inputDir, i);
            std::vector<unsigned char> input;

            if (cfg.generate) {
                input = generateImage(cfg.width, cfg.height, i);
                writePgm(inputPath, input, cfg.width, cfg.height);
            } else {
                input = readPgm(inputPath, cfg.width, cfg.height);
            }

            std::vector<unsigned char> blurred(input.size());
            std::vector<unsigned char> edges(input.size());
            float blurMs = 0.0f;
            float edgeMs = 0.0f;

            processImageOnGpu(input, blurred, edges, cfg.width, cfg.height, blurMs, edgeMs);

            writePgm(imagePath(cfg.outputDir + "/blur", i, "_blur"), blurred, cfg.width, cfg.height);
            writePgm(imagePath(cfg.outputDir + "/edge", i, "_edge"), edges, cfg.width, cfg.height);

            float totalMs = blurMs + edgeMs;
            log << i << "," << cfg.width << "," << cfg.height << ","
                << std::fixed << std::setprecision(4) << blurMs << "," << edgeMs << "," << totalMs << "\n";

            totalBlur += blurMs;
            totalEdge += edgeMs;
        }

        std::cout << "Processed " << cfg.numImages << " images of size "
                  << cfg.width << "x" << cfg.height << " using CUDA kernels.\n";
        std::cout << "Average blur kernel time: " << (totalBlur / cfg.numImages) << " ms\n";
        std::cout << "Average edge kernel time: " << (totalEdge / cfg.numImages) << " ms\n";
        std::cout << "Outputs written to: " << cfg.outputDir << "\n";
        std::cout << "Execution log written to: logs/execution_log.csv\n";

        CUDA_CHECK(cudaDeviceReset());
        return EXIT_SUCCESS;
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
}
