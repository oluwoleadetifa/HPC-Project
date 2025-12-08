# synthetic_images_generator.py
from pathlib import Path
import numpy as np
from PIL import Image
import zipfile

def make_noise_image(w, h, seed=0, channels=3):
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 256, size=(h, w, channels), dtype=np.uint8)
    return Image.fromarray(arr)

def make_gradient_image(w, h, seed=0, channels=3):
    rng = np.random.RandomState(seed)
    c1 = rng.randint(0, 256, size=(channels,), dtype=np.uint8)
    c2 = rng.randint(0, 256, size=(channels,), dtype=np.uint8)
    angle = rng.uniform(0, 2*np.pi)
    xs = np.linspace(-1, 1, w)[None, :]
    ys = np.linspace(-1, 1, h)[:, None]
    proj = np.cos(angle)*xs + np.sin(angle)*ys
    t = (proj - proj.min()) / (proj.max() - proj.min())
    arr = np.zeros((h, w, channels), dtype=np.uint8)
    for ch in range(channels):
        arr[..., ch] = ((c1[ch] * (1 - t) + c2[ch] * t)).astype(np.uint8)
    return Image.fromarray(arr)

def generate_images(output_dir, sizes, per_size=3, mode='noise', seed=0, fmt='png'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    global_counter = 0
    for s in sizes:
        for i in range(per_size):
            img_seed = seed + global_counter
            if mode == 'noise':
                img = make_noise_image(s, s, seed=img_seed)
            else:
                img = make_gradient_image(s, s, seed=img_seed)
            fname = output_dir / f"img_{s}x{s}_{i+1:02d}_seed{img_seed}.{fmt}"
            img.save(fname)
            files.append(fname)
            global_counter += 1
    return files

if __name__ == "__main__":
    # example usage
    for i in [100,500,1000]:
      for j in [256, 512, 1024]:
        files = generate_images(f"../data/{j}/{i}", sizes=[j], per_size=i,
                                mode="noise", seed=42, fmt="png")