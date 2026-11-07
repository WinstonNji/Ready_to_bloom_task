"""Method A: classic OpenCV baseline. No training and no neural network.

Idea: in an end-on photo of a plank stack, each plank is a thin horizontal band, so
brightness changes sharply at the edge between two planks. We measure how strong the
horizontal edges are on every pixel row inside the stack face. That gives a wavy line
with one bump per plank edge, and counting the bumps counts the planks.

The method is deliberately simple. It only looks at a narrow strip in the middle of
the image, so it misses other stacks and fails on tilted views and glare.

Usage:
    python src/baseline_classical.py
    python src/baseline_classical.py --roi 0.3,0.1,0.7,0.9
"""
# argparse reads the command line options
import argparse
# csv writes the results table
import csv

# OpenCV does the image processing
import cv2
# numpy handles the arrays
import numpy as np
# gaussian_filter1d smooths the row profile
from scipy.ndimage import gaussian_filter1d
# find_peaks locates the bumps in the profile
from scipy.signal import find_peaks

# Shared helpers for paths, images and annotations
from common import OVERLAY_DIR, RESULTS_DIR, list_images, load_bgr, load_points


def count_planks(img, roi, smooth=1.5, prominence=0.15):
    """Count plank edges inside the region of interest.

    roi is (x1, y1, x2, y2) as fractions of the image size.
    Returns (count, y position of each edge, region in pixels).
    """
    # Image height and width in pixels
    h, w = img.shape[:2]
    # Convert the region from fractions to pixel coordinates
    x1, y1, x2, y2 = int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h)

    # Crop the region and convert it to grayscale, since color is not needed
    gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    # A light blur so noise does not create false edges
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # The Sobel filter in the y direction responds to horizontal edges
    edges = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))

    # Average the edge strength across each row, then smooth the result
    profile = gaussian_filter1d(edges.mean(axis=1), smooth)
    # Rescale to the range 0 to 1 so the peak threshold works on any image
    profile = (profile - profile.min()) / (profile.max() - profile.min() + 1e-6)

    # Each peak in the profile is one edge between two planks.
    # prominence sets how much a bump must stand out, distance keeps peaks at least 3 rows apart.
    peaks, _ = find_peaks(profile, prominence=prominence, distance=3)
    # Peak positions are relative to the crop, so add y1 to get image rows
    return len(peaks), peaks + y1, (x1, y1, x2, y2)


def draw(img, edge_rows, region, count):
    """Draw the search region, a red line at each detected edge, and the count."""
    # Draw on a copy so the original stays unchanged
    out = img.copy()
    x1, y1, x2, y2 = region
    # Light blue rectangle around the search region
    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 200, 0), 1)
    # Red horizontal line at every detected plank edge
    for row in edge_rows:
        cv2.line(out, (x1, int(row)), (x2, int(row)), (0, 0, 255), 1)
    # Yellow label with the count in the top left corner
    cv2.putText(out, f"classical count: {count}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return out


def main():
    # Define the command line option for the search region
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi", default="0.35,0.05,0.65,0.95",
                        help="search region as fractions x1,y1,x2,y2")
    args = parser.parse_args()
    # Turn the text "a,b,c,d" into four numbers
    roi = tuple(float(v) for v in args.roi.split(","))

    # Make sure the overlay folder exists
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    # One result row per image
    rows = []
    for path in list_images():
        # Load the image and count planks in the region
        img = load_bgr(path)
        count, edge_rows, region = count_planks(img, roi)

        # Compare with the clicked count when the image has been annotated
        true = len(load_points(path)) or None
        # Save the image with the detected edges drawn on it
        cv2.imwrite(str(OVERLAY_DIR / f"{path.stem}_classical.jpg"),
                    draw(img, edge_rows, region, count))
        # Record the result, with the error only if the true count is known
        rows.append({"image": path.name, "predicted": count, "true": true,
                     "abs_error": abs(count - true) if true else ""})
        print(rows[-1])

    # Write the table of results to a CSV file
    with open(RESULTS_DIR / "baseline_classical.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "predicted", "true", "abs_error"])
        writer.writeheader()
        writer.writerows(rows)


# Run main only when the file is executed directly
if __name__ == "__main__":
    main()
