# Accelerating Batch Image Processing

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

This repository contains a Python-based benchmarking framework for evaluating image-processing pipelines across **sequential execution**, **multithreading**, **multiprocessing**, and **GPU acceleration**. The goal is to compare throughput, runtime, and scalability for CPU-bound workloads and lightweight GPU tasks.

The project systematically tests different resolutions, batch sizes, and parallelization strategies to determine optimal execution models for high-performance image processing.

---

## Features

* CPU benchmarking:

  * Sequential execution
  * `ThreadPoolExecutor` multithreading
  * `ProcessPoolExecutor` multiprocessing
* GPU benchmarking with CUDA-enabled pipelines
* Image-processing pipeline includes:

  1. Grayscale conversion
  2. Histogram equalization
  3. Gaussian blur (7×7 kernel)
  4. Sobel filtering (X & Y)
  5. Canny edge detection
* Performance metrics:

  * Wall-clock time
  * Throughput (images/sec)
  * Speedup relative to sequential baseline
* Synthetic dataset generation for reproducibility

---

## Installation

### Requirements

* Python 3.10+
* Libraries:

  ```bash
  pip install numpy opencv-python matplotlib tqdm
  ```
* Optional for GPU acceleration:

  ```bash
  pip install cupy-cuda11x torch torchvision
  ```

---

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/batch-image-processing.git
   cd batch-image-processing
   ```

2. Generate synthetic images:

   ```bash
   python generate_dataset.py --resolution 256 512 1024 --count 100 500 1000
   ```

3. Run benchmarks:

   ```bash
   python run_benchmarks.py --mode sequential
   python run_benchmarks.py --mode threadpool --workers 8
   python run_benchmarks.py --mode processpool --workers 8
   python run_benchmarks.py --mode gpu
   ```

4. Visualize results:

   ```bash
   python plot_results.py
   ```

---

## Results

* **CPU Performance:** Multithreading (8 threads) provided the highest throughput for small-to-medium images, outperforming multiprocessing in most cases due to low thread overhead and GIL release by native libraries.
* **GPU Performance:** GPU acceleration achieved moderate speedup, but performance was limited by small, lightweight operations and host-device transfer overhead.
* **Combined CPU–GPU Comparison:** Threaded CPU execution often surpassed GPU performance for unbatched, lightweight pipelines.

Example CPU throughput table (images/sec):

| Resolution | Sequential | Threads (8) | Processes (8) |
| ---------- | ---------- | ----------- | ------------- |
| 256×256    | 499.67     | **1930.92** | 1022.67       |
| 512×512    | 131.56     | **451.55**  | 395.49        |
| 1024×1024  | 35.43      | **110.73**  | 103.86        |

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements

* Inspired by research on CPU-GPU hybrid pipelines and Amdahl’s Law.
* References:

  * Qian, D. (2016). *High performance computing: a brief review and prospects.*
  * Hangün, B., & Eyecioğlu, Ö. (2017). *Performance comparison between OpenCV CPU and GPU functions.*
  * Teodoro, G., et al. (2012, 2013). *Accelerating large scale image analyses on hybrid systems.*

---

## Contact

Oluwole Adetifa – [LinkedIn](https://www.linkedin.com/in/oluwoleadetifa) – [oluw.adetifa@example.com](mailto:oluw.adetifa@example.com)