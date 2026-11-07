"""Method C: fine-tune a pre-trained YOLO detector and test it leave-one-image-out.

Leave-one-out: we only have 5 images, so for each image we train on the other four and
then count the one the model has never seen. This gives an honest error estimate.

Random crops: four images are too few to train on directly. From each training image we
cut many random crops at different zoom levels. YOLO also adds its own augmentation
(flips, color changes, mosaic, scaling) while it trains.

Usage on a GPU machine such as Colab:
    python src/train_yolo.py --epochs 80           leave-one-out test
    python src/train_yolo.py --final --epochs 80   train on all images for delivery

Run src/annotate.py first, because training needs the clicked points.
"""
# argparse reads the command line options
import argparse
# csv writes the results table
import csv
# random picks the crop positions
import random
# shutil deletes old fold folders
import shutil
# time measures prediction speed
import time

# OpenCV crops, resizes and saves images
import cv2

# Shared helpers for paths, images, annotations, box estimation and scoring
from common import (OVERLAY_DIR, RESULTS_DIR, ROOT, boxes_from_points, list_images,
                    load_bgr, load_points, match_points)

# Generated crops of the confidential images. This folder is ignored by git.
YOLO_DIR = ROOT / "data" / "yolo"


def write_sample(img, boxes, img_path, label_path):
    """Save an image and its YOLO label file.

    Each label line is: class, box center x, center y, width, height, with the last
    four scaled to between 0 and 1. There is one class, 0, for a plank end.
    """
    # Image size, needed to scale the box coordinates
    h, w = img.shape[:2]
    # Save the image file
    cv2.imwrite(str(img_path), img)
    # One text line per box: class 0, then center x, center y, width, height as fractions
    lines = [f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
             f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}" for x1, y1, x2, y2 in boxes]
    # Save the label file next to the image
    label_path.write_text("\n".join(lines))


def random_crops(img, boxes, count, size=640):
    """Yield `count` random crops, each resized to size x size, with their boxes.

    Each crop covers 45 to 100 percent of the image side, which gives different zoom levels.
    """
    # Image height and width
    h, w = img.shape[:2]
    for _ in range(count):
        # Pick a random window inside the image
        fraction = random.uniform(0.45, 1.0)
        # Window size in pixels
        crop_w, crop_h = int(w * fraction), int(h * fraction)
        # Random top-left corner that keeps the window inside the image
        x0, y0 = random.randint(0, w - crop_w), random.randint(0, h - crop_h)
        # Cut the window out of the image
        crop = img[y0:y0 + crop_h, x0:x0 + crop_w]
        # Factors for scaling the crop to the output size
        scale_x, scale_y = size / crop_w, size / crop_h

        # Move each box into the crop. Keep it only if at least 60 percent is still visible.
        kept = []
        for x1, y1, x2, y2 in boxes:
            # Part of the box that lies inside the crop
            cx1, cy1 = max(x1, x0), max(y1, y0)
            cx2, cy2 = min(x2, x0 + crop_w), min(y2, y0 + crop_h)
            # Visible area and full area of the box
            visible = max(0, cx2 - cx1) * max(0, cy2 - cy1)
            full = (x2 - x1) * (y2 - y1)
            if full > 0 and visible / full >= 0.6:
                # Shift to crop coordinates and scale to the output size
                kept.append(((cx1 - x0) * scale_x, (cy1 - y0) * scale_y,
                             (cx2 - x0) * scale_x, (cy2 - y0) * scale_y))
        # Resize the crop to the fixed training size and hand it out with its boxes
        yield cv2.resize(crop, (size, size)), kept


def build_fold(holdout, images, crops_per_image):
    """Build the training folder for one fold and return the path of its data.yaml.

    The held-out image goes into the validation folder untouched. Every other image
    contributes random crops to the training folder. With holdout=None (final mode)
    all images are used for training.
    """
    # Folder name: the held-out image name, or "all" in final mode
    fold = YOLO_DIR / f"fold_{holdout.stem if holdout else 'all'}"
    # Remove any old copy so stale crops do not leak into this run
    if fold.exists():
        shutil.rmtree(fold)
    # Create the train and val folders for images and labels
    for split in ("train", "val"):
        (fold / "images" / split).mkdir(parents=True)
        (fold / "labels" / split).mkdir(parents=True)

    for path in images:
        # Load the image and estimate its boxes from the clicked points
        img = load_bgr(path)
        h, w = img.shape[:2]
        boxes = boxes_from_points(load_points(path), w, h)

        # Final mode has no held-out image, but YOLO still needs a validation folder.
        # The first image fills it and its score is ignored.
        is_holdout = path == holdout
        if is_holdout or (holdout is None and path == images[0]):
            write_sample(img, boxes, fold / "images/val" / f"{path.stem}.jpg",
                         fold / "labels/val" / f"{path.stem}.txt")
        # The held-out image must never be used for training
        if is_holdout:
            continue

        # Write the random training crops of this image
        for i, (crop, crop_boxes) in enumerate(random_crops(img, boxes, crops_per_image)):
            write_sample(crop, crop_boxes, fold / "images/train" / f"{path.stem}_{i}.jpg",
                         fold / "labels/train" / f"{path.stem}_{i}.txt")

    # data.yaml tells YOLO where the images are and what the class is called
    data_yaml = fold / "data.yaml"
    data_yaml.write_text(f"path: {fold.as_posix()}\ntrain: images/train\nval: images/val\n"
                         f"names:\n  0: plank_end\n")
    return data_yaml


def train_fold(YOLO, args, holdout, images):
    """Train one model and return it loaded from its last saved weights."""
    # Build the training data for this fold
    data_yaml = build_fold(holdout, images, args.crops)
    # Name of the output folder under runs/
    name = f"fold_{holdout.stem}" if holdout else "final_all"

    # Start from COCO pre-trained weights and fine-tune on our crops.
    # val=False skips per-epoch validation, so the held-out image cannot influence training.
    model = YOLO(args.model)
    model.train(data=str(data_yaml), epochs=args.epochs, imgsz=640, batch=16,
                project=str(ROOT / "runs"), name=name, exist_ok=True,
                degrees=10, fliplr=0.5, flipud=0.0, scale=0.4, mosaic=1.0,
                hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, verbose=False, val=False)
    # Reload the weights of the final epoch so the model used for counting is explicit
    return YOLO(str(ROOT / "runs" / name / "weights" / "last.pt"))


def main():
    # Define the command line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80, help="passes over the training crops")
    parser.add_argument("--crops", type=int, default=60, help="random crops per training image")
    parser.add_argument("--model", default="yolo11n.pt", help="pre-trained starting weights")
    parser.add_argument("--conf", type=float, default=0.25, help="minimum detection confidence")
    parser.add_argument("--final", action="store_true",
                        help="train on all images with no held-out test")
    args = parser.parse_args()

    # Imported here so the helper functions can be used without ultralytics installed
    from ultralytics import YOLO

    # Same crops on every run
    random.seed(0)
    # Use only images that have clicked points
    images = [p for p in list_images() if load_points(p)]
    # Stop with a clear message if nothing has been annotated
    assert len(images) >= 2, "Annotate the images first with python src/annotate.py"
    # Make sure the overlay folder exists
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    # Final mode: one model trained on everything, used for delivery and export
    if args.final:
        train_fold(YOLO, args, None, images)
        print("final weights: runs/final_all/weights/last.pt")
        return

    # Leave-one-out mode: train a fresh model for each held-out image
    rows = []
    for holdout in images:
        # Train on the other images
        model = train_fold(YOLO, args, holdout, images)
        # Load the held-out image and its true points
        img = load_bgr(holdout)
        points = load_points(holdout)

        # The first prediction is slower, so run one before timing
        model.predict(img, imgsz=640, conf=args.conf, verbose=False)
        # Time the real prediction
        start = time.perf_counter()
        result = model.predict(img, imgsz=640, conf=args.conf, iou=0.5, max_det=1000,
                               verbose=False)[0]
        ms = (time.perf_counter() - start) * 1000

        # Score the count, and check that the boxes land on the true planks
        predicted = len(result.boxes)
        found, good = match_points(points, result.boxes.xyxy.cpu().numpy())
        # Save the image with predicted boxes drawn on it
        cv2.imwrite(str(OVERLAY_DIR / f"{holdout.stem}_yolo_loo.jpg"), result.plot())
        # Record the numbers for this fold
        rows.append({"image": holdout.name, "predicted": predicted, "true": len(points),
                     "abs_error": abs(predicted - len(points)),
                     "recall": round(found / len(points), 2),
                     "precision": round(good / max(1, predicted), 2),
                     "ms": round(ms, 1)})
        print(rows[-1], flush=True)

    # Summary: average count error over all folds
    mae = sum(r["abs_error"] for r in rows) / len(rows)
    print(f"Mean absolute error over {len(rows)} held-out images: {mae:.1f} planks")
    # Save the per-image table
    with open(RESULTS_DIR / "yolo_loo.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# Run main only when the file is executed directly
if __name__ == "__main__":
    main()
