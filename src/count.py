"""Count planks in photos with a trained YOLO model, draw the boxes, and print timing.

Usage:
    python src/count.py --weights runs/final_all/weights/last.pt
    python src/count.py --weights runs/final_all/weights/last.pt --images some_photo.jpg

Without --images it uses every image in data/images.
"""
# argparse reads the command line options
import argparse
# time measures how long each prediction takes
import time
# Path handles the --images argument
from pathlib import Path

# OpenCV saves the result images
import cv2

# Shared helpers for paths and image loading
from common import OVERLAY_DIR, list_images, load_bgr


def main():
    # Define the command line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="trained YOLO weights file")
    parser.add_argument("--images", default=None, help="one image file or a folder of images")
    parser.add_argument("--conf", type=float, default=0.25, help="minimum detection confidence")
    args = parser.parse_args()

    # Imported here so the helper functions can be used without ultralytics installed
    from ultralytics import YOLO

    # Load the trained model from its weights file
    model = YOLO(args.weights)

    # Work out which images to process
    target = Path(args.images) if args.images else None
    if target is None:
        # No path given, so use the project images
        paths = list_images()
    elif target.is_file():
        # A single image file
        paths = [target]
    else:
        # A folder, so use everything inside it
        paths = sorted(target.iterdir())

    # Make sure the overlay folder exists
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    # The first prediction is slower, so run one before timing
    model.predict(load_bgr(paths[0]), verbose=False)

    for path in paths:
        # Load the image and start the timer
        img = load_bgr(path)
        start = time.perf_counter()
        # Run the detector. imgsz is the input size, iou the duplicate-removal threshold,
        # max_det the largest number of boxes allowed.
        result = model.predict(img, imgsz=640, conf=args.conf, iou=0.5, max_det=1000,
                               verbose=False)[0]
        # Elapsed time in milliseconds
        ms = (time.perf_counter() - start) * 1000

        # result.plot() returns the image with all boxes drawn on it
        cv2.imwrite(str(OVERLAY_DIR / f"{path.stem}_count.jpg"), result.plot())
        # The number of boxes is the plank count
        print(f"{path.name}: {len(result.boxes)} planks ({ms:.0f} ms)")


# Run main only when the file is executed directly
if __name__ == "__main__":
    main()
