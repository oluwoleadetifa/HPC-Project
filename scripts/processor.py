# processor.py
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from numba import cuda

@cuda.jit
def invert_kernel(img, out):
    x, y = cuda.grid(2)
    if x < img.shape[0] and y < img.shape[1]:
        out[x, y] = 255 - img[x, y]

ImageLike = Union[str, Path, np.ndarray]


def process_image(
    image: ImageLike,
    size: Tuple[int, int] = (256, 256),
    save_path: Optional[Union[str, Path]] = None,
    return_time: bool = False,
    return_image: bool = False,
    canny_thresh: Tuple[int, int] = (50, 150),
    use_cuda: bool = False
) -> Union[float, np.ndarray, Tuple[float, np.ndarray]]:
    """
    CPU-heavy image pipeline:
      1. load (if given a path)
      2. convert to grayscale
      3. resize to `size` (width, height)
      4. histogram equalization (on grayscale)
      5. Gaussian blur (7x7)
      6. Canny edge detection
      7. Sobel filter (x-direction + y-direction), combine with edges

    Args:
        image: file path (str/Path) or a numpy.ndarray (BGR image).
        size: target resolution as (width, height), default (256,256).
        save_path: optional path to save the final processed image (uint8).
        return_time: if True, return elapsed seconds (float) or (elapsed, result) if return_image True.
        return_image: if True, return processed image (uint8 numpy array).
        canny_thresh: Canny low/high thresholds.

    Returns:
        If return_time and not return_image -> float (seconds)
        If return_image and not return_time -> np.ndarray (uint8 processed)
        If both True -> (seconds, np.ndarray)
        If neither -> True on success (useful when saving only)
    """
    t0 = time.perf_counter()

    # 1) load
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Unable to read image: {image}")
    elif isinstance(image, np.ndarray):
        img = image.copy()
    else:
        raise TypeError("`image` must be a path or numpy.ndarray")

    # 2) convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3) resize
    # OpenCV expects (width, height) in resize target argument order (dsize=(w,h))
    w, h = size
    # use INTER_LINEAR as a good default (bilinear)
    gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_LINEAR)

    # 4) histogram equalization (improves contrast; still uint8)
    eq = cv2.equalizeHist(gray)

    # 5) gaussian blur with 7x7 kernel
    blurred = cv2.GaussianBlur(eq, (7, 7), 0)

    # 6) canny edge detection
    edges = cv2.Canny(blurred, canny_thresh[0], canny_thresh[1])

    # 7) sobel filter (both axes), convert to uint8 scale
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    # Normalize and convert to uint8
    sobel_norm = np.uint8(np.clip((sobel_mag / (sobel_mag.max() + 1e-12)) * 255.0, 0, 255))

    # Combine edges and sobel to increase CPU work and create a single-channel result
    combined = cv2.addWeighted(edges, 0.6, sobel_norm, 0.4, 0)

    # =====================================================
    # OPTIONAL: GPU post-processing step (custom CUDA kernel)
    # =====================================================
    if use_cuda:
        # Convert CPU image to int32 for CUDA (numba doesn't like uint8)
        if not cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU or CUDA driver detected.")
        img32 = combined.astype(np.int32)

        d_in = cuda.to_device(img32)
        d_out = cuda.device_array_like(img32)

        threadsperblock = (16, 16)
        blockspergrid_x = (img32.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid_y = (img32.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        invert_kernel[blockspergrid, threadsperblock](d_in, d_out)
        cuda.synchronize()

        combined = d_out.copy_to_host().astype(np.uint8)

    # Optionally save: convert to 3-channel for nicer viewing if desired
    if save_path is not None:
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        # saved as grayscale single-channel image
        cv2.imwrite(str(sp), combined)

    elapsed = time.perf_counter() - t0

    if return_time and return_image:
        return elapsed, combined
    if return_time:
        return elapsed
    if return_image:
        return combined
    return True
def process_image_from_path(path: str, *args, **kwargs):
    """
    Wrapper intended for ProcessPoolExecutor: accepts a string path and forwards to process_image.
    Returns elapsed time or True/array consistent with process_image.
    """
    return process_image(path, *args, **kwargs)

if __name__ == "__main__":
    import time
    from pathlib import Path

    img_path = "../data/sythentic_imageimg_0000.png"  # make sure you have generated images first
    # Quick correctness check (get processed image)
    out = process_image(img_path, size=(512,512), save_path="samples/out_512.png", return_image=True)
    print("Got processed image shape:", out.shape)

    # Quick timing single image
    elapsed = process_image(img_path, size=(1024,1024), return_time=True)
    print(f"Elapsed (single image, 1024x1024): {elapsed:.4f}s")

    # Tiny loop benchmark (N images sequential)
    imgs = sorted([str(p) for p in Path("images").glob("*.png")])[:50]
    t0 = time.perf_counter()
    for p in imgs:
        process_image(p, size=(512,512), return_time=False)
    t1 = time.perf_counter()
    print(f"Sequential 50 images @512: {t1-t0:.2f}s --> {50/(t1-t0):.2f} imgs/sec")
