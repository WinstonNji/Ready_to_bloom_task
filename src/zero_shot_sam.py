"""Method B: zero-shot FastSAM counting. No training and no labels needed.

FastSAM is a model that was pre-trained to outline every object in an image. We run it,
then keep only the outlines that look like a plank end: wide, thin, and plank sized.
The number of outlines we keep is the count.

Two sets of settings are stored in PRESETS:
    default   chosen by reasoning about the problem, not by looking at the true counts
    tuned     the best settings found by src/tune_fastsam.py

For each image the script reports the count error, recall, precision and CPU time.
Recall is the share of true planks that fall inside a box. Precision is the share of
boxes that hold a true plank. They show whether the boxes are in the right places,
which a matching total alone does not.

Usage:
    python src/zero_shot_sam.py              tuned settings
    python src/zero_shot_sam.py --default    default settings
"""
# argparse reads the command line options
import argparse
# csv writes the results table
import csv
# time measures how long each run takes
import time

# OpenCV draws boxes and saves images
import cv2
# numpy handles box arrays
import numpy as np

# Shared helpers for paths, images, annotations and box matching
from common import (OVERLAY_DIR, RESULTS_DIR, list_images, load_bgr, load_points,
                    match_points)

# conf, imgsz and iou are FastSAM settings. The rest are the plank shape filter:
# aspect is width divided by height, area is a fraction of the whole image,
# consist (optional) drops boxes whose height is far from the median height.
PRESETS = {
    "default": dict(conf=0.2, imgsz=1024, iou=0.7, min_aspect=2.0, max_aspect=9.0,
                    min_area=0.0008, max_area=0.03, consist=None),
    "tuned": dict(conf=0.2, imgsz=640, iou=0.7, min_aspect=2.5, max_aspect=9.0,
                  min_area=0.0004, max_area=0.03, consist=None),
}


def box_iou(a, b):
    """Overlap of two boxes: shared area divided by combined area (0 to 1)."""
    # Corners of the shared rectangle
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    # Area of the shared rectangle, zero if the boxes do not touch
    shared = max(0, x2 - x1) * max(0, y2 - y1)
    # Area covered by either box: both areas added, minus the shared part counted twice
    combined = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - shared
    # Guard against dividing by zero
    return shared / combined if combined > 0 else 0


def run_fastsam(model, img, cfg):
    """Run FastSAM on one image and return every detected box as an (N, 4) array."""
    # device="cpu" runs on the processor, retina_masks gives full-size masks,
    # conf is the minimum confidence, iou the duplicate-removal threshold.
    result = model(img, device="cpu", retina_masks=True, imgsz=cfg["imgsz"],
                   conf=cfg["conf"], iou=cfg["iou"], verbose=False)[0]
    # No detections at all gives an empty array
    if result.boxes is None:
        return np.zeros((0, 4))
    # Boxes as (x1, y1, x2, y2) in a numpy array
    return result.boxes.xyxy.cpu().numpy()


def keep_plank_like(boxes, img_area, min_aspect, max_aspect, min_area, max_area, consist=None):
    """Keep only boxes that look like a plank end and remove duplicates."""
    # Shape and size filter
    kept = []
    for x1, y1, x2, y2 in boxes:
        # Width and height of this box
        w, h = x2 - x1, y2 - y1
        # Skip boxes with no height
        if h <= 0:
            continue
        # Keep the box if its shape and its size relative to the image are plank-like
        if min_aspect <= w / h <= max_aspect and min_area <= (w * h) / img_area <= max_area:
            kept.append((x1, y1, x2, y2))

    # Optional filter: drop boxes whose height differs a lot from the typical plank height
    if consist and len(kept) >= 5:
        # Median height of the boxes kept so far
        median_h = float(np.median([b[3] - b[1] for b in kept]))
        # Keep only heights within a factor of consist of the median
        kept = [b for b in kept if median_h / consist <= (b[3] - b[1]) <= median_h * consist]

    # The same plank can be found twice, so keep the larger box of any overlapping pair.
    # Sort largest first so the larger box wins.
    kept.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    unique = []
    for box in kept:
        # Add the box only if it overlaps every box already kept by less than 0.5
        if all(box_iou(box, other) < 0.5 for other in unique):
            unique.append(box)
    return unique


def draw_check(img, boxes, points, count, hit_flags):
    """Draw boxes (green if they hold a true plank, red if not) and the true points."""
    # Draw on a copy so the original stays unchanged
    out = img.copy()
    for box, hit in zip(boxes, hit_flags):
        # Green for a box on a real plank, red for a false alarm (BGR order)
        color = (0, 200, 0) if hit else (0, 0, 255)
        cv2.rectangle(out, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 1)
    # Small blue dot at each clicked plank
    for x, y in points:
        cv2.circle(out, (int(x), int(y)), 2, (255, 120, 0), -1)
    # Yellow label with the count
    cv2.putText(out, f"FastSAM count: {count}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return out


def main():
    # Define the command line option for choosing the preset
    parser = argparse.ArgumentParser()
    parser.add_argument("--default", action="store_true", help="use the untuned settings")
    args = parser.parse_args()
    # Pick the preset name and its settings
    preset = "default" if args.default else "tuned"
    cfg = PRESETS[preset]

    # Imported here so the helper functions above can be used without ultralytics installed
    from ultralytics import FastSAM

    # Load the model. The weights download the first time.
    model = FastSAM("FastSAM-s.pt")
    # Make sure the overlay folder exists
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    paths = list_images()

    # The first run is slower, so do one before timing
    run_fastsam(model, load_bgr(paths[0]), cfg)

    # Collect one result row per image, and totals for overall recall
    rows = []
    total_found = total_true = 0
    for path in paths:
        # Load the image, its size, and the clicked points
        img = load_bgr(path)
        h, w = img.shape[:2]
        points = load_points(path)

        # Run the model and time it
        start = time.perf_counter()
        raw = run_fastsam(model, img, cfg)
        ms = (time.perf_counter() - start) * 1000
        # Keep only the plank-like boxes
        boxes = keep_plank_like(raw, w * h, cfg["min_aspect"], cfg["max_aspect"],
                                cfg["min_area"], cfg["max_area"], cfg["consist"])

        # Compare with the clicked points: how many were found and how many boxes were good
        found, good = match_points(points, boxes)
        total_found += found
        total_true += len(points)
        # For each box, whether it holds at least one clicked point (used for the colors)
        hit_flags = [any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for x, y in points)
                     for b in boxes]
        # Save the check image
        cv2.imwrite(str(OVERLAY_DIR / f"{path.stem}_fastsam_{preset}.jpg"),
                    draw_check(img, boxes, points, len(boxes), hit_flags))

        # Record the numbers for this image
        rows.append({"image": path.name, "true": len(points), "predicted": len(boxes),
                     "abs_error": abs(len(boxes) - len(points)),
                     "recall": round(found / len(points), 2),
                     "precision": round(good / max(1, len(boxes)), 2),
                     "ms_cpu": round(ms)})
        print(rows[-1])

    # Write the table of results to a CSV file named after the preset
    with open(RESULTS_DIR / f"zero_shot_fastsam_{preset}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Summary: average count error and recall over all images
    mae = sum(r["abs_error"] for r in rows) / len(rows)
    print(f"Preset {preset}: mean absolute error {mae:.1f} planks, "
          f"overall recall {total_found / total_true:.0%}")


# Run main only when the file is executed directly
if __name__ == "__main__":
    main()
