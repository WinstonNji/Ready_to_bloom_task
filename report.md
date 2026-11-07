# Scaffold plank counting from photographs: methods, results, and deployment on iOS

Winston Nji, test task for Ready to Bloom, November 2026

## 1. Summary

The task was to build a model that counts scaffolding items in photographs, with the detected items marked on the image, for later use in an iOS application. Five test photographs were supplied. They show stacks of steel scaffold planks seen end-on, not round pipes, so the counted object is the plank end face (Section 2).

We compared three methods against counts that we annotated by hand: a classical OpenCV edge-profile method, a zero-shot FastSAM segmentation model with a shape filter, and a YOLO11n detector fine-tuned on crops of the photographs. Table 1 gives the mean absolute error (MAE) in planks per photograph.

Table 1. Mean absolute error over the five photographs.

| Method | Training data used | MAE (planks) |
| --- | --- | --- |
| OpenCV edge profile | none | 30.0 |
| FastSAM, default settings | none | 9.2 |
| YOLO11n fine-tuned, leave-one-out | 4 photographs per fold | 11.4 |
| FastSAM, settings tuned, leave-one-out | settings chosen on 4 photographs per fold | 1.2 |
| FastSAM, settings tuned on all 5 (optimistic) | settings chosen on all 5 photographs | 0.4 |

The most accurate method on these photographs is FastSAM with tuned filter settings (MAE 1.2 under leave-one-out). It is also the heaviest (about 0.9 s per photograph on a laptop CPU). YOLO11n is small and fast (about 11 ms on a GPU) and exports to Core ML, but with four training photographs per fold its error was 11.4. With only five photographs, all of these figures are indications of behavior and not reliable estimates of accuracy on new photographs.

## 2. Task definition and counting convention

The brief asks for the number of scaffolding pipes (tubes) in a photograph. The supplied photographs (file names containing the Japanese words for scaffold part and steel plank) show flat steel planks stacked and photographed from the end. We therefore count plank ends.

The brief did not define the correct count. We adopted this rule: each plank end face that is individually visible on the front face of any stack in the photograph counts once. The long side faces of the stacks are not counted again, and planks too hidden or blurred to distinguish are not counted. Other rules (for example, counting only the largest stack) would give different reference counts. The correct counts used here are 54, 24, 73, 13, and 60 for photographs 1 to 5, a total of 224 planks.

## 3. Data and annotation strategy

With five photographs, the main constraint is data. Three choices follow from it.

First, we annotate with a single click on the center of each plank end, using a small tool written for this task (`src/annotate.py`). A point is much faster to place than a box, and it is sufficient for the correct count and for checking whether a predicted box lies on a real plank. We estimated the training boxes from the points: the height is 0.8 times the median distance between neighboring points and the width is five times the height. These boxes are approximate and inherit the imprecision of the clicks. We found that this matters (Section 7).

Second, we use pre-trained models throughout. FastSAM was trained on a large segmentation dataset and needs no training here. YOLO11n starts from weights pre-trained on COCO, and we fine-tune it for one class.

Third, we use augmentation to enlarge the training set. For every training photograph, we cut 60 random crops covering 45 to 100 percent of the image side and resize them to 640 pixels, which varies the apparent plank size. Ultralytics adds rotation up to 10 degrees, horizontal flips, scale variation, mosaic, and color jitter.

## 4. Methods

Method A, classical edge profile. We convert a narrow vertical strip in the middle of the photograph to grayscale and blur it. The vertical Sobel gradient responds to the horizontal boundaries between planks. We average the gradient magnitude along each pixel row, smooth the result, and count the peaks of the resulting profile. The method uses no training and no labels.

Method B, zero-shot FastSAM. The FastSAM-s model outlines every object in the photograph. We keep boxes only if they look like a plank end (width to height ratio within a range, and area within a range as a fraction of the photograph), and we remove overlapping duplicates. We used two sets of settings. We chose the default settings by reasoning about the problem before looking at the correct counts (input size 1024, aspect ratio 2.0 to 9.0, area 0.08 to 3 percent of the photograph). The tuned settings came from a search of about 2,900 combinations of model and filter settings, run by `src/tune_fastsam.py`. The selected combination uses input size 640, aspect ratio 2.5 to 9.0, and area 0.04 to 3 percent.

Method C, fine-tuned YOLO11n. We fine-tune a YOLO11n detector (2.6 million parameters) for 80 epochs on the random crops, with batch size 16 and 640-pixel input. The count is the number of detections with confidence of at least 0.25 after non-maximum suppression at an overlap threshold of 0.5. We disable per-epoch validation and use the final-epoch weights, so the held-out photograph cannot influence training.

## 5. Evaluation protocol

Methods A and B with default settings use no labels, so we evaluate them directly on all five photographs. For methods that use the correct counts, we evaluate by leave-one-out. For YOLO, we train a separate model on four photographs and test it on the fifth, five times. For the tuned FastSAM settings, we select the best combination using the correct counts of four photographs and test it on the fifth. We report the figure for settings tuned on all five photographs only to show how optimistic it is to select settings after seeing the answers.

We report three measures. The absolute count error is the absolute difference between predicted and correct counts. Recall is the fraction of correct planks whose clicked point lies inside at least one predicted box. Precision is the fraction of predicted boxes that contain at least one clicked point. Recall and precision matter because a total can be right by coincidence when false detections offset misses.

## 6. Results

### 6.1 Counts and errors

Table 2 gives the correct and predicted counts. The predicted counts of the tuned FastSAM are those of the settings chosen on all five photographs. We did not store the per-photograph predictions of the leave-one-out selection, only their errors, which Table 3 gives.

Table 2. Correct and predicted counts.

| Photograph | Correct | OpenCV | FastSAM default | FastSAM tuned (all 5) | YOLO11n (leave-one-out) |
| --- | --- | --- | --- | --- | --- |
| 1 | 54 | 12 | 25 | 54 | 62 |
| 2 | 24 | 12 | 19 | 23 | 22 |
| 3 | 73 | 19 | 75 | 73 | 89 |
| 4 | 13 | 16 | 10 | 13 | 19 |
| 5 | 60 | 21 | 67 | 59 | 35 |

Table 3. Absolute count error in planks.

| Photograph | OpenCV | FastSAM default | FastSAM tuned (leave-one-out) | FastSAM tuned (all 5) | YOLO11n (leave-one-out) |
| --- | --- | --- | --- | --- | --- |
| 1 | 42 | 29 | 0 | 0 | 8 |
| 2 | 12 | 5 | 4 | 1 | 2 |
| 3 | 54 | 2 | 0 | 0 | 16 |
| 4 | 3 | 3 | 1 | 0 | 6 |
| 5 | 39 | 7 | 1 | 1 | 25 |
| Mean | 30.0 | 9.2 | 1.2 | 0.4 | 11.4 |

Counts of the YOLO model trained on all five photographs (59, 25, 73, 12, 61) are not a valid test and are omitted from the tables, because every photograph was also a training photograph.

> [ Figure 1: Predicted boxes on the five photographs for each method ]
> Withheld because the photographs are confidential. Insert from: results/overlays/ (files named `*_fastsam_tuned.jpg`, `*_yolo_loo.jpg`, `*_classical.jpg`).

### 6.2 Localization

Table 4 gives recall and precision, which test whether the boxes lie on real planks.

Table 4. Recall and precision per photograph (recall / precision).

| Photograph | FastSAM default | FastSAM tuned (all 5) | YOLO11n (leave-one-out) |
| --- | --- | --- | --- |
| 1 | 0.37 / 0.76 | 0.87 / 0.85 | 1.00 / 0.90 |
| 2 | 0.33 / 0.42 | 0.54 / 0.52 | 1.00 / 1.00 |
| 3 | 0.99 / 0.92 | 0.88 / 0.84 | 0.92 / 0.79 |
| 4 | 0.31 / 0.40 | 0.77 / 0.69 | 0.69 / 0.42 |
| 5 | 0.97 / 0.93 | 0.88 / 0.93 | 0.52 / 0.89 |

Overall recall is 72 percent for default FastSAM (162 of 224 planks) and 83 percent for tuned FastSAM (187 of 224). The tuned counts are therefore closer to the truth partly because more planks are found, but the count error of 0.4 is smaller than the localization quality alone would suggest, since false detections offset misses. Recall of 1.00 for YOLO on photographs 1 and 2 should be read with caution, because the point-in-box criterion is permissive when boxes are large or overlap.

### 6.3 Inference time

Table 5. Inference time per photograph after a warm-up call.

| Method | Hardware | Time |
| --- | --- | --- |
| OpenCV edge profile | laptop CPU | 0.6 to 1.7 ms (18.8 ms for the first photograph) |
| FastSAM tuned, input 640 | laptop CPU, integrated graphics only, 8 GB RAM | 551 to 1181 ms, mean 860 ms |
| FastSAM default, input 1024 | same laptop CPU | about 1.8 to 3.3 s |
| YOLO11n, leave-one-out models | Colab Tesla T4 GPU | 10.9 to 12.1 ms |
| YOLO11n, final model | Colab Tesla T4 GPU | 11 to 18 ms |

On-device iOS timings were not measured, because the Core ML model can only be run on macOS or iOS (Section 8).

## 7. Error analysis

The OpenCV method reads only a strip in the middle of the photograph, so it counts one column of one stack and misses all others. It is close on photograph 4 only by coincidence: that photograph has few plank ends and many horizontal lines on the top surfaces.

Default FastSAM undercounts photograph 1 (25 against 54). It boxes background shelves and misses many plank ends in the crowded lower area. On photograph 3 it boxes plank ends of a partly visible stack on the right that we did not annotate, so its near-exact total of 75 against 73 is partly a coincidence of extra and missing boxes.

Tuned FastSAM improves photograph 1 to 87 percent recall. Its weakest photographs are 2 and 4, with recall of 54 and 77 percent. Both photographs mix plank ends with the long perforated top surfaces, which are similar in shape to the target.

YOLO11n undercounts photograph 5 (35 against 60, recall 0.52). It detects most of the left column but misses much of the middle and right columns, and it also places boxes on the side face. It overcounts photograph 3 (89 against 73), where predicted boxes pile on top of each other. On photograph 4 its precision is 0.42 because it fires on top surfaces. Likely causes are the small training set, the approximate training boxes, and the fact that no hyperparameters were tuned.

Annotation also contributes error. A few clicks fall between planks or are doubled on thin planks, which affects both the correct counts by a small amount and the derived training boxes. The counting rule is our own convention, and planks that are partly hidden are a judgment call.

## 8. Running on iOS

Export. We exported the final YOLO11n weights, trained on all five photographs, to Core ML with the Ultralytics exporter. The first attempt failed because of a numpy 2.4 incompatibility in the converter. A repeat with numpy restricted to versions below 2.4 succeeded and produced a package of about 5.4 MB with 16-bit weights. The package is a pipeline of the detector and a non-maximum suppression stage, takes a 640 by 640 image, and exposes the overlap and confidence thresholds as inputs, so the application can set them. Setting the overlap threshold to 0.5 reproduces the counting rule used in this study. We inspected the package but did not run it, since running it requires macOS or iOS. Its predictions and on-device latency are unverified. A 2.6 million parameter detector would be expected to be well within real-time limits on recent iPhones, but this needs to be measured.

We did not export or test FastSAM for iOS. It is a larger model (about 23 MB) and its speed on a laptop CPU suggests it is considerably heavier than the 5.4 MB YOLO package.

What would be needed for real conditions.

1. More and more varied data. Five photographs do not cover lighting, distance, viewing angle, stack type, backgrounds, or plank dimensions. A realistic target is a few hundred photographs from the actual use sites, annotated with boxes or points under a written counting rule.
2. Capture guidance in the app. A face-on view, a minimum distance, and a visible overlay of the working area would remove many of the failure cases seen here (steep top views, heavy perspective, partly visible neighboring stacks).
3. Resolution. In photograph 3 the distance from one plank end to the next is only about 13 pixels at the supplied resolution. For large stacks, the app should photograph at higher resolution and either tile the image or run the detector at a larger input size, merging detections across tile borders.
4. A way to correct the count. The user should be able to add or remove detections with a tap, with the result stored. Corrections are the cheapest source of new labeled data.
5. Training labels. Using the tuned FastSAM boxes, matched to clicked points, as training labels for YOLO is a promising way to get cleaner labels from few clicks. We have not tested it, so it is a proposal and not a result.
6. An evaluation protocol. Accuracy should be measured on a held-out set from real sites with a stated tolerance, for example the share of photographs counted within one plank or within two percent.
7. Alternatives for dense or occluded stacks. Density or point regression, which predicts a count map and not boxes, is an option that does not need accurate boxes. We did not evaluate it here.

## 9. Limitations

The evaluation uses five photographs, so every figure in this report is a rough indication. We chose the FastSAM settings from about 2,900 combinations with five reference counts, and the leave-one-out figure of 1.2 reduces but does not remove the risk that the settings fit these photographs. We trained the YOLO model once with default hyperparameters, so its error of 11.4 is not an upper bound on what the approach can reach. The YOLO run used an unseeded random crop generator and we have not repeated it, and a repeat would differ slightly. The correct counts are our own annotations under our own counting rule. On-device performance of the Core ML model is unmeasured.

## 10. Reproducibility

The code is in this repository. We did not include the photographs, because they are confidential and were supplied for this task only. Placing them in `data/images/` and the annotation files in `data/annotations/` allows the scripts to be run.

| Script | Purpose |
| --- | --- |
| `src/annotate.py` | click annotation tool that creates the correct counts |
| `src/baseline_classical.py` | method A, OpenCV edge profile |
| `src/zero_shot_sam.py` | method B, FastSAM with default or tuned settings (`--default` for untuned) |
| `src/tune_fastsam.py` | settings search with leave-one-out check |
| `src/train_yolo.py` | method C, leave-one-out training and evaluation (`--final` for the delivery model) |
| `src/count.py` | counting with trained YOLO weights, with timing |
| `src/export_coreml.py` | Core ML export |
| `notebooks/scaffold_count_colab.ipynb` | executed Colab notebook for the YOLO experiments |

Software versions: Python 3.12 locally and 3.13 on Colab, Ultralytics 8.4.203, PyTorch 2.x, coremltools 9.0.
