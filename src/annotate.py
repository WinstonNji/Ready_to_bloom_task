"""Click-to-count annotation tool. It creates the correct counts we measure against.

Usage:
    python src/annotate.py                       annotate every image in turn
    python src/annotate.py --only plank3.jpg     annotate one image
    python src/annotate.py --size 650            smaller window if the bottom is cut off

Controls:
    left click    add a point on the center of one plank end
    right click   remove the nearest point
    u             undo the last point
    s             save and move to the next image
    q or Esc      quit without saving the current image

Saved points load again the next time, so a second pass can fix misses.
"""
# argparse reads the command line options
import argparse

# OpenCV shows the window and draws on the image
import cv2
# numpy finds the nearest point when deleting
import numpy as np

# Shared helpers for images and annotation files
from common import list_images, load_bgr, load_points, save_points

# Window title, which also reminds the user of the controls
WINDOW = "annotate (s = save and next, u = undo, right click = delete, q = quit)"


def annotate_one(path, target_size):
    """Show one image, collect clicks, and return True if it was saved."""
    # Load the image and read its height and width
    img = load_bgr(path)
    h, w = img.shape[:2]

    # Scale the image so its longest side is target_size, which makes small images easier to click
    scale = target_size / max(h, w)
    # Resize with cubic interpolation so the enlarged image stays smooth
    shown = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # Points are stored in original image pixels, not in the scaled window
    points = load_points(path)

    def redraw():
        """Draw all points and the running count on a copy of the image."""
        # Start from a clean copy so old dots do not pile up
        canvas = shown.copy()
        # Draw a red dot for each point, converted to window pixels
        for x, y in points:
            cv2.circle(canvas, (int(x * scale), int(y * scale)), 4, (0, 0, 255), -1)
        # Black bar behind the text so it is readable on any image
        cv2.rectangle(canvas, (0, 0), (420, 34), (0, 0, 0), -1)
        # Show the file name and how many planks are marked so far
        cv2.putText(canvas, f"{path.name}: {len(points)} planks", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # Display the finished frame
        cv2.imshow(WINDOW, canvas)

    def on_mouse(event, mouse_x, mouse_y, flags, param):
        """Add a point on left click, remove the nearest point on right click."""
        # Convert the click position from window pixels to original image pixels
        x, y = mouse_x / scale, mouse_y / scale
        # Left click adds a point
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            redraw()
        # Right click removes the point closest to the click
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            # Distance from the click to every existing point
            distances = [np.hypot(px - x, py - y) for px, py in points]
            # Remove the closest one
            points.pop(int(np.argmin(distances)))
            redraw()

    # Create the window at its natural size
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    # Send mouse events to on_mouse
    cv2.setMouseCallback(WINDOW, on_mouse)
    # Draw the first frame
    redraw()

    # Wait for a key press until the user saves or quits
    while True:
        # Check for a key every 20 ms and keep only the lowest 8 bits
        key = cv2.waitKey(20) & 0xFF
        # s saves the points and moves on
        if key == ord("s"):
            save_points(path, points)
            print(f"saved {path.name}: {len(points)} planks")
            return True
        # u removes the most recent point
        if key == ord("u") and points:
            points.pop()
            redraw()
        # q or Esc (key code 27) quits without saving this image
        if key in (ord("q"), 27):
            return False


def main():
    # Define the command line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="annotate only this file name, for example plank3.jpg")
    parser.add_argument("--size", type=int, default=700, help="window size in pixels")
    # Read the options the user typed
    args = parser.parse_args()

    # Go through the images in order and stop as soon as the user quits
    for path in list_images():
        # Skip other images when --only is given
        if args.only and path.name != args.only:
            continue
        # annotate_one returns False when the user quits
        if not annotate_one(path, args.size):
            break
    # Close the window when finished
    cv2.destroyAllWindows()


# Run main only when the file is executed directly, not when it is imported
if __name__ == "__main__":
    main()
