import cv2
import os
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm


def canny_sketch(img, low=35, high=110):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    # increase sigmaColor + sigmaSpace slightly to suppress hair texture
    filtered = cv2.bilateralFilter(
        equalized, d=7, sigmaColor=65, sigmaSpace=65)
    edges = cv2.Canny(filtered, low, high)
    sketch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return sketch


def make_pair(sketch, photo):
    return np.concatenate([sketch, photo], axis=1)


def generate_pairs(input_dir, output_dir, low, high, size, limit):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    images = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    if limit:
        images = images[:limit]

    print(f"Processing {len(images)} images...")
    skipped = 0

    for img_path in tqdm(images):
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        img = cv2.resize(img, (size, size))
        sketch = canny_sketch(img, low, high)
        pair = make_pair(sketch, img)

        out_path = output_path / img_path.name
        cv2.imwrite(str(out_path), pair)

    print(f"Done. Saved {len(images) - skipped} pairs to {output_dir}")
    if skipped:
        print(f"Skipped {skipped} unreadable files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  required=True,
                        help="Folder of raw face images")
    parser.add_argument("--output_dir", required=True,
                        help="Folder to save sketch-photo pairs")
    parser.add_argument("--low",   type=int, default=50,
                        help="Canny low threshold")
    parser.add_argument("--high",  type=int, default=150,
                        help="Canny high threshold")
    parser.add_argument("--size",  type=int, default=256,
                        help="Output image size")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max images to process")
    args = parser.parse_args()

    generate_pairs(args.input_dir, args.output_dir, args.low,
                   args.high, args.size, args.limit)
