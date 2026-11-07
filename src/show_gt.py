"""Draw the clicked points and the boxes made from them so they can be checked by eye.

Usage:
    python src/show_gt.py

Output goes to results/overlays/<name>_gt.jpg. These files show the confidential
images, so they stay out of git.
"""
# OpenCV draws the rectangles, dots and text
import cv2

# Shared helpers for paths, images, annotations and box estimation
from common import OVERLAY_DIR, boxes_from_points, list_images, load_bgr, load_points


def main():
    # Make sure the overlay folder exists
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    for path in list_images():
        # Load the clicked points for this image
        points = load_points(path)
        # Skip images that have not been annotated
        if not points:
            continue

        # Load the image and its size
        img = load_bgr(path)
        h, w = img.shape[:2]

        # Green boxes are the estimated plank boxes, red dots are the clicks
        for x1, y1, x2, y2 in boxes_from_points(points, w, h):
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
        for x, y in points:
            cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)
        # Yellow label with the true count
        cv2.putText(img, f"true count: {len(points)}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Save the picture and report what was written
        out = OVERLAY_DIR / f"{path.stem}_gt.jpg"
        cv2.imwrite(str(out), img)
        print("wrote", out.name, len(points))


# Run main only when the file is executed directly
if __name__ == "__main__":
    main()
