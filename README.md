# Scaffold plank counting (Ready to Bloom test task)

This project counts the steel scaffold planks in a photograph of a plank stack and marks each detection on the image. The full write-up is in [report.md](report.md).

The five test images were supplied by Ready to Bloom for this task only. They are not in this repository. To run the code, place them in `data/images/` (the folder is ignored by git).

## Counting rule

The images show stacks of steel planks seen end-on, not round pipes. The task did not define the correct count, so this project uses one rule: each plank end that is individually visible on the front faces of all stack columns counts once. The long side faces are not counted again, and planks too hidden or blurred to tell apart are not counted.

## Results

Mean absolute error over the five images, in planks:

| Method | Mean absolute error |
| --- | --- |
| OpenCV edge profile | 30.0 |
| FastSAM, default settings | 9.2 |
| YOLO11n fine-tuned, leave-one-out | 11.4 |
| FastSAM, tuned settings, leave-one-out | 1.2 |

These figures come from five images and are rough indications. See the report for the limits.

## Layout

| Path | Contents |
| --- | --- |
| `src/` | annotation tool, the three counting methods, tuning, counting, and Core ML export |
| `notebooks/` | executed Colab notebook for the YOLO experiments |
| `data/annotations/` | clicked plank points for each image, and `data/gt_counts.csv` with the correct counts |
| `results/` | tables of counts, errors, and timings |
| `report.md` | approach, results, error analysis, and the iOS view |

## Running the code

Install the requirements, put the images in `data/images/`, and run the scripts from the project folder.

```
pip install -r requirements.txt
python src/annotate.py
python src/baseline_classical.py
python src/zero_shot_sam.py
python src/tune_fastsam.py
```

YOLO training needs a GPU. Use the notebook in `notebooks/` on Google Colab, or run `python src/train_yolo.py` on a GPU machine.

## Use of AI assistance

An AI assistant (Claude, by Anthropic) was used heavily in this project. It wrote most of the code and suggested the methods that were compared. The author annotated all 224 plank ends by hand, chose the counting rule, ran the GPU training and the Core ML export on Google Colab, and reviewed the results and prediction overlays at each step.
