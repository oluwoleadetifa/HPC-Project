# Accelerating Batch Image Processing

A Python benchmarking framework for comparing image-processing performance across **sequential execution**, **multithreading**, **multiprocessing**, and **GPU acceleration**.

The project explores a practical systems question: **when does additional parallelism actually improve throughput, and when does coordination or transfer overhead erase the benefit?**

## What the project measures

The benchmark suite evaluates:

- wall-clock runtime
- throughput in images/second
- speedup relative to a sequential baseline
- scaling behavior across worker counts
- CPU-versus-GPU tradeoffs

The image-processing pipeline includes grayscale conversion, histogram equalization, Gaussian blur, Sobel filtering, and Canny edge detection.

## Execution models

- Sequential Python execution
- ThreadPoolExecutor multithreading
- ProcessPoolExecutor multiprocessing
- CUDA-enabled GPU processing

## Representative results

| Resolution | Sequential | Threads (8) | Processes (8) |
| --- | ---: | ---: | ---: |
| 256×256 | 499.67 img/s | **1930.92 img/s** | 1022.67 img/s |
| 512×512 | 131.56 img/s | **451.55 img/s** | 395.49 img/s |
| 1024×1024 | 35.43 img/s | **110.73 img/s** | 103.86 img/s |

For the tested workloads, multithreading performed particularly well because much of the heavy image-processing work executes in native libraries that can release the Python GIL. GPU execution was not automatically faster: for lightweight, unbatched operations, kernel-launch and host/device-transfer overhead could dominate.

## Repository structure

- `scripts/` — benchmark and dataset-generation code
- `results/` — benchmark outputs and experiment artifacts
- `README.md` — project overview and representative findings

## Setup

Requirements:

- Python 3.10+
- NumPy
- OpenCV
- Matplotlib
- tqdm
- optional CUDA-compatible dependencies for GPU experiments

Clone the repository:

```bash
git clone https://github.com/oluwoleadetifa/HPC-Project.git
cd HPC-Project
```

Install the CPU dependencies:

```bash
pip install numpy opencv-python matplotlib tqdm
```

For GPU experiments, install the CUDA-compatible packages appropriate for your environment.

## Why this project matters

The project is less about "GPU = faster" and more about understanding **execution-model tradeoffs**. It demonstrates why workload size, native-library behavior, serialization, communication overhead, synchronization, and host/device transfer all matter when choosing a parallel architecture.

## Author

Oluwole Adetifa  
[LinkedIn](https://www.linkedin.com/in/oluwole-adetifa-278586113) • [Portfolio](https://oluwoleadetifa.com)
