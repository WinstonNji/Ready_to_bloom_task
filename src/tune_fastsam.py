"""Tune the FastSAM settings and measure how far the tuning can be trusted.

The search has three steps:
    1. Run FastSAM once for each model setting and keep the raw boxes.
    2. Try many shape filter settings on those saved boxes. This is fast because the
       model does not run again.
    3. Report two numbers:
       tuned on all 5   best settings chosen with all images. This is optimistic,
                        because the settings are picked after seeing the answers.
       leave-one-out    for each image, pick settings using only the other four, then
                        test on that image. This is the fairer estimate for new photos.

Usage:
    python src/tune_fastsam.py        takes a few minutes on a CPU
"""
# csv writes the results table
import csv
# itertools.product builds every combination of settings
import itertools

# numpy handles the error tables
import numpy as np

# Shared helpers for paths, images and annotations
from common import RESULTS_DIR, list_images, load_bgr, load_points
# FastSAM helpers and the default preset, shared with zero_shot_sam.py
from zero_shot_sam import PRESETS, keep_plank_like, run_fastsam

# Settings to try. Model grid is (conf, imgsz, iou). Filter grid is
# (min aspect, max aspect, min area, max area, height consistency).
MODEL_GRID = list(itertools.product([0.1, 0.2, 0.4], [640, 1024], [0.7, 0.9]))
FILTER_GRID = list(itertools.product([1.5, 2.0, 2.5],
                                     [6.0, 9.0, 14.0],
                                     [0.0004, 0.0008, 0.0015],
                                     [0.02, 0.03, 0.04],
                                     [None, 1.4, 1.8]))


def main():
    # Imported here so the helper functions can be used without ultralytics installed
    from ultralytics import FastSAM

    # Load the model, then only the images that have clicked points
    model = FastSAM("FastSAM-s.pt")
    paths = [p for p in list_images() if load_points(p)]
    # Load each image once and reuse it for every setting
    images = [load_bgr(p) for p in paths]
    # True plank count per image, from the clicked points
    true = np.array([len(load_points(p)) for p in paths])
    # Image area in pixels, needed by the size filter
    areas = [img.shape[0] * img.shape[1] for img in images]

    # Step 1: run the model once per model setting and keep the raw boxes
    raw = {}
    for conf, imgsz, iou in MODEL_GRID:
        # Pack the three model settings into one dictionary
        cfg = dict(conf=conf, imgsz=imgsz, iou=iou)
        # One list of raw boxes per image for this model setting
        raw[(conf, imgsz, iou)] = [run_fastsam(model, img, cfg) for img in images]
        # Progress message, flushed so it appears immediately
        print("done", (conf, imgsz, iou), flush=True)

    # Step 2: score every combination. errors[k][i] is the count error of
    # combination k on image i.
    combos, errors = [], []
    for model_setting in MODEL_GRID:
        for filter_setting in FILTER_GRID:
            # Predicted count per image after applying this filter to the saved boxes
            predicted = np.array([
                len(keep_plank_like(raw[model_setting][i], areas[i], *filter_setting[:4],
                                    consist=filter_setting[4]))
                for i in range(len(paths))])
            # Remember the combination and its absolute errors
            combos.append((model_setting, filter_setting))
            errors.append(np.abs(predicted - true))
    # Table with one row per combination and one column per image
    errors = np.array(errors)

    # Step 3a: best combination chosen with all images (optimistic)
    best = int(np.argmin(errors.mean(axis=1)))

    # Step 3b: leave-one-out. Pick the best combination without image i, then score it on i.
    loo = []
    for i in range(len(paths)):
        # Indices of the other images
        others = [j for j in range(len(paths)) if j != i]
        # Combination with the lowest mean error on the other images
        pick = int(np.argmin(errors[:, others].mean(axis=1)))
        # Its error on the held-out image
        loo.append(int(errors[pick, i]))

    # The untuned default, for comparison
    d = PRESETS["default"]
    default_combo = ((d["conf"], d["imgsz"], d["iou"]),
                     (d["min_aspect"], d["max_aspect"], d["min_area"], d["max_area"],
                      d["consist"]))
    # Row of the default combination in the error table
    default = combos.index(default_combo)

    # Print the three results side by side
    print("\nImages          :", [p.name for p in paths])
    print("True counts     :", true.tolist())
    print(f"Tuned on all 5  : mean error {errors[best].mean():.1f}  per image {errors[best].tolist()}")
    print("  settings (conf, imgsz, iou), (min aspect, max aspect, min area, max area, consist):")
    print("  ", combos[best])
    print(f"Leave-one-out   : mean error {np.mean(loo):.1f}  per image {loo}")
    print(f"Default         : mean error {errors[default].mean():.1f}  per image {errors[default].tolist()}")

    # Save the three results to a CSV file
    with open(RESULTS_DIR / "fastsam_tuning.csv", "w", newline="") as f:
        writer = csv.writer(f)
        # Header row: variant name, mean error, then one column per image
        writer.writerow(["variant", "mae"] + [p.name for p in paths])
        writer.writerow(["tuned_on_all_5", round(errors[best].mean(), 2)] + errors[best].tolist())
        writer.writerow(["leave_one_out", round(float(np.mean(loo)), 2)] + loo)
        writer.writerow(["default", round(errors[default].mean(), 2)] + errors[default].tolist())


# Run main only when the file is executed directly
if __name__ == "__main__":
    main()
