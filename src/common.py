"""Shared helpers used by every other script.

It covers three jobs: finding and loading images, reading and writing the clicked
ground-truth points, and turning those points into boxes for training and scoring.

Folder layout the helpers expect:
    data/images/            the test images (not in git, they are confidential)
    data/annotations/       one JSON file of clicked points per image
    results/                CSV tables
    results/overlays/       images with boxes drawn on them (not in git)
"""
# json reads and writes the annotation files
import json
# Path builds file paths that work on Windows, macOS and Linux
from pathlib import Path

# OpenCV converts color formats
import cv2
# numpy holds images and point coordinates as arrays
import numpy as np
# Pillow opens image files, including formats OpenCV may not read
from PIL import Image

# Lets Pillow open .avif files on versions that do not support them natively
try:
    # The import registers the AVIF reader, the name itself is never used
    import pillow_avif  # noqa: F401
except ImportError:
    # Newer Pillow versions read AVIF directly, so a missing plugin is fine
    pass

# Project root folder, found from this file's location so scripts work from any folder
ROOT = Path(__file__).resolve().parent.parent
# Folder with the test images
IMAGE_DIR = ROOT / "data" / "images"
# Folder with one JSON file of clicked points per image
ANN_DIR = ROOT / "data" / "annotations"
# Folder for CSV result tables
RESULTS_DIR = ROOT / "results"
# Folder for images with boxes drawn on them
OVERLAY_DIR = RESULTS_DIR / "overlays"
# File extensions that count as images
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".avif", ".webp"}


def list_images():
    """Return every image file in data/images, sorted by name."""
    # Keep only files with an image extension, compared in lower case
    return sorted(p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def load_bgr(path):
    """Load an image as a BGR numpy array, the color order OpenCV expects."""
    # Pillow reads the file and converts it to RGB, then numpy makes an array
    rgb = np.array(Image.open(path).convert("RGB"))
    # OpenCV expects blue, green, red order, so swap the channels
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def ann_path(image_path):
    """Path of the annotation file that belongs to an image."""
    # plank1.jpg maps to data/annotations/plank1.json
    return ANN_DIR / (Path(image_path).stem + ".json")


def load_points(image_path):
    """Return the clicked points [(x, y), ...] in original image pixels.

    Returns an empty list if the image has not been annotated yet.
    """
    # Find the annotation file for this image
    path = ann_path(image_path)
    # No file means the image has not been annotated yet
    if not path.exists():
        return []
    # Read the JSON and turn each [x, y] pair into a tuple
    return [tuple(point) for point in json.loads(path.read_text())["points"]]


def save_points(image_path, points):
    """Write the clicked points for an image to its annotation file."""
    # Create the annotations folder on first use
    ANN_DIR.mkdir(parents=True, exist_ok=True)
    # Store the image name, the count, and the points rounded to one decimal
    data = {
        "image": Path(image_path).name,
        "count": len(points),
        "points": [[round(float(x), 1), round(float(y), 1)] for x, y in points],
    }
    # Write the data as readable, indented JSON
    ann_path(image_path).write_text(json.dumps(data, indent=1))


def match_points(points, boxes):
    """Check how well predicted boxes cover the true points.

    A true plank counts as found when its point lies inside any predicted box.
    Returns (planks found, boxes that hold a true plank).
    Recall is the first number divided by the number of true planks.
    Precision is the second number divided by the number of boxes.
    """
    # With no points or no boxes nothing can match
    if not len(points) or not len(boxes):
        return 0, 0
    # inside[i][j] is True when point i lies inside box j
    inside = np.array([[b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in boxes]
                       for x, y in points])
    # Count points covered by any box, and boxes that cover any point
    return int(inside.any(axis=1).sum()), int(inside.any(axis=0).sum())


def boxes_from_points(points, img_w, img_h, width_ratio=5.0):
    """Turn clicked points into rough boxes (x1, y1, x2, y2) for YOLO training.

    We only have a point per plank, not a real box, so the size is estimated.
    Height is a bit under the median distance between neighbouring points, which
    is about the thickness of one plank. Width is width_ratio times the height.
    These boxes are approximate and only as accurate as the clicks.
    """
    # Work with the points as a float array
    pts = np.array(points, dtype=float)
    # At least two points are needed to estimate a spacing
    if len(pts) < 2:
        return []

    # Distance from every point to every other point, as a square table
    dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    # A point is at distance zero from itself, so exclude that entry
    np.fill_diagonal(dists, np.inf)
    # Box height: 80 percent of the typical nearest-neighbor distance
    box_h = float(np.median(dists.min(axis=1))) * 0.8
    # Box width: a plank end is much wider than it is tall
    box_w = box_h * width_ratio

    # Center a box on each point and keep it inside the image
    return [(max(0, x - box_w / 2), max(0, y - box_h / 2),
             min(img_w - 1, x + box_w / 2), min(img_h - 1, y + box_h / 2))
            for x, y in pts]
