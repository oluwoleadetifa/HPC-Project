# run_experiment.py
import csv
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from processor import process_image_from_path, process_image

DATA_ROOT = Path("../data")
RESULTS_ROOT = Path("../results")
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

SIZES = [256, 512, 1024]
COUNTS = [100, 500, 1000]

WORKERS = [1, 2, 4, 8]   # threadpool & processpool workers


def run_sequential(img_paths, size):
    t0 = time.perf_counter()
    for p in img_paths:
        process_image(p, size=(size, size), return_time=False)
    return time.perf_counter() - t0


def run_threadpool(img_paths, size, workers):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_image_from_path, p, (size, size)) for p in img_paths]
        for f in as_completed(futures):
            f.result()
    return time.perf_counter() - t0


def run_processpool(img_paths, size, workers):
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_image_from_path, p, (size, size)) for p in img_paths]
        for f in as_completed(futures):
            f.result()
    return time.perf_counter() - t0

def run_cuda(img_paths, size):
    t0 = time.perf_counter()
    for p in img_paths:
        process_image(p, size=(size,size), use_cuda=True)
    return time.perf_counter() - t0


def run_experiment():
    csv_path = RESULTS_ROOT / "timings.csv"
    summary_path = RESULTS_ROOT / "summary.txt"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "size",
            "count",
            "method",
            "workers",
            "time_sec",
            "images_per_sec"
        ])

        for size in SIZES:
            for count in COUNTS:

                # Get image paths
                data_dir = DATA_ROOT / str(size) / str(count)
                img_paths = sorted([str(p) for p in data_dir.glob("*.png")])

                print(f"\n=== Running size={size}, count={count}, {len(img_paths)} images ===")

                ## 1. Sequential
                t = run_sequential(img_paths, size)
                writer.writerow([size, count, "sequential", 1, t, count / t])
                print(f"Sequential: {t:.2f}s ({count/t:.2f} img/s)")

                ## 2. ThreadPool
                for w in WORKERS:
                    t = run_threadpool(img_paths, size, w)
                    writer.writerow([size, count, "threadpool", w, t, count / t])
                    print(f"ThreadPool-{w}: {t:.2f}s ({count/t:.2f} img/s)")

                ## 3. ProcessPool
                for w in WORKERS:
                    t = run_processpool(img_paths, size, w)
                    writer.writerow([size, count, "processpool", w, t, count / t])
                    print(f"ProcessPool-{w}: {t:.2f}s ({count/t:.2f} img/s)")

                ## 4. GPU (CUDA)
                # t = run_cuda(img_paths, size)
                # writer.writerow([size, count, "cuda", 1, t, count / t])
                # print(f"CUDA GPU: {t:.2f}s ({count/t:.2f} img/s)")


    # Save summary
    with open(summary_path, "w") as f:
        f.write("Experiment complete. See timings.csv for full results.\n")

    print("\nDONE. Results saved to /results/")
    print("timings.csv + summary.txt generated.")


if __name__ == "__main__":
    run_experiment()
