"""
augmentation.py
---------------
Applies augmentations to the training split of a YOLOv8 dataset.

Two modes:
  uniform   — applies the same number of augmentations to every image
  balanced  — automatically computes per-class factors to reach a target count

Augmentations applied (5 transforms + 7 combinations = 12 total):
  flip_h, flip_v, rot90, brightness, blur,
  flip_h+brightness, flip_v+brightness, rot90+brightness,
  flip_h+blur, rot90+blur, flip_h+rot90, flip_v+rot90

Usage:
  python augmentation.py --data Gloves-2 --mode uniform --factor 5
  python augmentation.py --data Gloves-2 --mode balanced --target 2000

To force re-run: delete train/.augmented_done
"""

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import yaml

# ── Configuration ──────────────────────────────────────────────────────────────
RANDOM_SEED      = 42
BRIGHTNESS_LIMIT = 0.30
BLUR_LIMIT       = 5
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DONE_FLAG        = ".augmented_done"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ── Augmentation pool ──────────────────────────────────────────────────────────

def build_pool() -> dict:
    bp = A.BboxParams(format="yolo", label_fields=["class_ids"], min_visibility=0.3, clip=True)

    fh  = A.HorizontalFlip(p=1.0)
    fv  = A.VerticalFlip(p=1.0)
    r90 = A.RandomRotate90(p=1.0)
    br  = A.RandomBrightnessContrast(brightness_limit=BRIGHTNESS_LIMIT, contrast_limit=BRIGHTNESS_LIMIT, p=1.0)
    bl  = A.GaussianBlur(blur_limit=(3, BLUR_LIMIT), p=1.0)

    def p(*t): return A.Compose(list(t), bbox_params=bp)

    return {
        "flip_h":            p(fh),
        "flip_v":            p(fv),
        "rot90":             p(r90),
        "brightness":        p(br),
        "blur":              p(bl),
        "flip_h_brightness": p(fh, br),
        "flip_v_brightness": p(fv, br),
        "rot90_brightness":  p(r90, br),
        "flip_h_blur":       p(fh, bl),
        "rot90_blur":        p(r90, bl),
        "flip_h_rot90":      p(fh, r90),
        "flip_v_rot90":      p(fv, r90),
    }


# ── YOLO label I/O ─────────────────────────────────────────────────────────────

def read_labels(path: Path):
    class_ids, bboxes = [], []
    if not path.exists():
        return class_ids, bboxes
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        class_ids.append(int(parts[0]))
        bboxes.append(list(map(float, parts[1:5])))
    return class_ids, bboxes


def write_labels(path: Path, class_ids, bboxes):
    with open(path, "w") as f:
        for cls, (cx, cy, w, h) in zip(class_ids, bboxes):
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_class_names(dataset_root: Path) -> dict:
    """Returns {class_id: class_name} from data.yaml."""
    yaml_path = dataset_root / "data.yaml"
    if not yaml_path.exists():
        return {}
    data = yaml.safe_load(yaml_path.read_text())
    return {i: name for i, name in enumerate(data.get("names", []))}


def get_original_images(train_images: Path) -> list:
    """Returns all non-augmented images in train/images/."""
    return sorted([
        p for p in train_images.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS and "_aug_" not in p.stem
    ])


def compute_balanced_factors(images: list, train_labels: Path, target: int) -> dict:
    """
    Counts images per class, then computes the augmentation factor needed
    so each class reaches approximately `target` images.
    Returns {class_id: factor}.
    """
    counts = defaultdict(int)
    for img in images:
        class_ids, _ = read_labels(train_labels / (img.stem + ".txt"))
        for cls in set(class_ids):
            counts[cls] += 1
    return {cls: max(0, math.ceil(target / count) - 1) for cls, count in counts.items()}


def select_augmentations(factor: int, pool: dict, rng: random.Random) -> list:
    """
    Picks `factor` augmentations from the pool.
    Samples without replacement up to pool size, then with replacement beyond that.
    Returns [(index, aug_name), ...] for unique file naming.
    """
    names = list(pool.keys())
    if factor <= len(names):
        chosen = rng.sample(names, factor)
    else:
        chosen = names + rng.choices(names, k=factor - len(names))
    return list(enumerate(chosen, start=1))


# ── Core ───────────────────────────────────────────────────────────────────────

def run(dataset_path: str, mode: str, factor: int = 5, target: int = 2000, force: bool = False):
    root         = Path(dataset_path)
    train_images = root / "train" / "images"
    train_labels = root / "train" / "labels"
    done_flag    = root / "train" / DONE_FLAG

    if done_flag.exists() and not force:
        print(f"[augmentation] Already applied. Delete '{done_flag}' or use --force to re-run.")
        return

    if not train_images.exists(): raise FileNotFoundError(f"Not found: {train_images}")
    if not train_labels.exists(): raise FileNotFoundError(f"Not found: {train_labels}")

    rng       = random.Random(RANDOM_SEED)
    pool      = build_pool()
    names     = load_class_names(root)
    originals = get_original_images(train_images)

    if not originals:
        print("[augmentation] No images found in train/images/")
        return

    # ── Compute factor per image ───────────────────────────────────────────────
    if mode == "uniform":
        img_factors = {img: factor for img in originals}
        print(f"\n[augmentation] Mode       : uniform  (factor={factor})")

    else:  # balanced
        class_factors = compute_balanced_factors(originals, train_labels, target)

        # Print summary table
        counts = defaultdict(int)
        for img in originals:
            cls_ids, _ = read_labels(train_labels / (img.stem + ".txt"))
            for c in set(cls_ids): counts[c] += 1

        print(f"\n[augmentation] Mode       : balanced  (target={target})")
        print(f"\n  {'Class':<30} {'Images':>6}  {'Factor':>6}  {'~After':>7}")
        print(f"  {'─'*55}")
        for cls_id in sorted(counts):
            n = counts[cls_id]
            f = class_factors.get(cls_id, 0)
            print(f"  {names.get(cls_id, f'class_{cls_id}'):<30} {n:>6}  {f:>6}  {n*(1+f):>7}")

        # Per image: use the highest factor among all classes in that image
        img_factors = {}
        for img in originals:
            cls_ids, _ = read_labels(train_labels / (img.stem + ".txt"))
            img_factors[img] = max((class_factors.get(c, 0) for c in set(cls_ids)), default=0)

    total = sum(img_factors.values())
    print(f"\n[augmentation] Train images: {len(originals)}")
    print(f"[augmentation] To generate : ~{total}")
    print(f"[augmentation] Starting...\n")

    # ── Apply ──────────────────────────────────────────────────────────────────
    generated = 0
    skipped   = 0

    for idx, img_path in enumerate(originals, 1):
        img_fac = img_factors.get(img_path, 0)
        if img_fac == 0:
            continue

        if idx % 100 == 0 or idx == len(originals):
            print(f"  [{idx}/{len(originals)}] {img_path.name}")

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [WARNING] Cannot read {img_path.name} — skipping")
            skipped += 1
            continue

        image_rgb         = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        class_ids, bboxes = read_labels(train_labels / (img_path.stem + ".txt"))

        for aug_idx, aug_name in select_augmentations(img_fac, pool, rng):
            out_stem = f"{img_path.stem}_aug_{aug_idx:02d}_{aug_name}"
            out_img  = train_images / f"{out_stem}{img_path.suffix}"
            out_lbl  = train_labels / f"{out_stem}.txt"

            if out_img.exists():
                continue

            try:
                result = pool[aug_name](image=image_rgb, bboxes=bboxes or [], class_ids=class_ids or [])
                cv2.imwrite(str(out_img), cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR))
                write_labels(out_lbl, result["class_ids"], result["bboxes"])
                generated += 1
            except Exception as e:
                print(f"  [ERROR] {img_path.name} + {aug_name}: {e}")
                skipped += 1

    # ── Done flag ──────────────────────────────────────────────────────────────
    done_flag.write_text(
        f"mode: {mode}\n"
        f"original_images: {len(originals)}\n"
        f"generated: {generated}\n"
        f"skipped: {skipped}\n"
    )

    print(f"\n[augmentation] Done.")
    print(f"  Original : {len(originals)}")
    print(f"  Generated: {generated}")
    print(f"  Skipped  : {skipped}")
    print(f"  Total    : {len(originals) + generated}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply augmentations to a YOLOv8 training split.")

    parser.add_argument("--data",   required=True,                              help="Path to dataset root (contains train/, valid/, test/)")
    parser.add_argument("--mode",   required=True, choices=["uniform", "balanced"], help="uniform or balanced")
    parser.add_argument("--factor", type=int, default=5,                        help="[uniform] augmentations per image (default: 5)")
    parser.add_argument("--target", type=int, default=2000,                     help="[balanced] target images per class (default: 2000)")
    parser.add_argument("--force",  action="store_true",                        help="Re-run even if already applied")

    args = parser.parse_args()
    run(args.data, args.mode, args.factor, args.target, args.force)
