"""Export trained YOLO weights to Core ML, the model format used by iOS apps.

Usage:
    python src/export_coreml.py --weights runs/final_all/weights/last.pt

The .mlpackage is written next to the weights. The export runs on Linux, including
Colab, but the exported model can only be run and timed on a Mac or an iPhone.
Use numpy below 2.4, because coremltools 9 fails to convert YOLO with newer numpy.
"""
# argparse reads the command line options
import argparse


def main():
    # Define the command line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="trained YOLO weights file")
    parser.add_argument("--imgsz", type=int, default=640, help="model input size in pixels")
    args = parser.parse_args()

    # Imported here so the file can be read and checked without ultralytics installed
    from ultralytics import YOLO

    # nms=True adds duplicate-box removal inside the model, so the app gets final boxes.
    # half=True stores the weights in 16-bit floats, which makes the model smaller.
    path = YOLO(args.weights).export(format="coreml", imgsz=args.imgsz, nms=True, half=True)
    # Print where the package was written
    print("exported:", path)


# Run main only when the file is executed directly
if __name__ == "__main__":
    main()
