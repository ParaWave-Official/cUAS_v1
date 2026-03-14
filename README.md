# cUAS_v1 Drone Detection Benchmark
c-UAS small performance benchmark for Shepherd.AI

This document describes the evaluation methodology, dataset construction, file structure, and benchmark results for an open-vocabulary drone detection system tested across two publicly available datasets: BirDrone and Anti-UAV.

---

## 1. Dataset Construction

### 1.1 Image Selection

500 images were randomly sampled from each source dataset for concision in inference time during evaluation, stratified to ensure an even distribution across two features:

- **Image modality:** thermal (infrared) vs. visible (RGB/EO)
- **Drone presence:** images containing at least one drone vs. drone-free background images, many containing birds

This stratification ensures the benchmark reflects realistic operating conditions, where the detector must handle both modality types and must not over-trigger on negative scenes.

### 1.2 Manual Review and Ground Truth Filtering

Following random selection, all images were manually reviewed by a human annotator. Ground truth bounding boxes deemed unreasonable from a static-image perspective were removed. Criteria for removal included:

- Boxes covering implausibly small regions relative to the image
- Boxes placed on objects not clarly identifiable to the human eye for categorization

The final retained images after this review process form the evaluated dataset.

### 1.3 Source Datasets

- **BirDrone** ([dataset link](https://universe.roboflow.com/fatin-zamri/birdrone-dataset)) - A dataset of drone and bird imagery captured in the visible spectrum, designed for distinguishing small UAVs from bird clutter in outdoor environments. 244 images were selected from this publicly available dataset.
birdrone dataset Computer Vision Dataset © 2023 by Fatin Zamri is licensed under CC BY 4.0

- **Anti-UAV** ([dataset link](https://github.com/ucas-vg/Anti-UAV)) - A paired thermal and electro-optical drone detection dataset containing diverse UAV types across a wide range of altitudes, backgrounds, and lighting conditions. Images were selected scross multiple video recordings, totaling 383 images.

Neither dataset was used to train or finetune Shepherd.AI, and images were selected solely from validation and test splits.

---

## 2. Data Access

The full dataset (images and prediction JSON files) is available via Google Drive:

**[Google Drive Folder](https://drive.google.com/drive/folders/1CKr533TGapLA6FmdUiXsh1LgHP-peJvy?usp=sharing)**

### Folder Structure

```
drive/
├── antiuav/
│   ├── images/
│   └── predictions.json
└── birdrone/
    ├── images/
    └── predictions.json
```

---

## 3. JSON Format

Each `predictions.json` file is a dictionary keyed by image filename. Every entry contains three fields: `raw_data` (ground truth annotations), `predictions` (model outputs), and `tags` (image metadata).

### Example Entry

```json
{
  "frame0044.jpg": {
    "tags": ["thermal"],
    "raw_data": [
      {
        "class": "drone",
        "boxes": [
          {
            "x_min": 0.4469,
            "y_min": 0.5449,
            "x_max": 0.4938,
            "y_max": 0.5918
          }
        ]
      }
    ],
    "predictions": [
      {
        "class": "drone",
        "boxes": [
          {
            "x_min": 0.4255,
            "y_min": 0.5312,
            "x_max": 0.5120,
            "y_max": 0.6055,
            "level": "medium"
          }
        ]
      }
    ]
  }
}
```

### Field Descriptions

| Field | Description |
|---|---|
| `tags` | List of image metadata labels. Example values: `"thermal"` or `"visible"`. |
| `raw_data` | Ground truth annotations. Each entry has a `class` name and a list of `boxes`. |
| `raw_data[].boxes` | Normalized bounding boxes (`x_min`, `y_min`, `x_max`, `y_max` in range 0.0 for top left to 1.0 for bottom right of image). This also contains a "level" field denoting the confidence level, one of: `"highest"`, `"high"`, `"medium"`, or `"low"`.  |
| `predictions` | Model output. One entry per predicted class (`"drone"`). |
| `predictions[].boxes` | Predicted bounding boxes with normalized coordinates plu `level` (confidence tier). |

---

## 4. Metric Definitions

All metrics are computed by running `benchmark_analysis.py`, which combines both datasets into a single pool of images and evaluates predictions against ground truth.

### 4.1 Confidence Levels

Predictions are filtered cumulatively from highest to lowest confidence. Evaluating at level `"medium"` includes all boxes labelled `"highest"`, `"high"`, and `"medium"`. This allows analysis of the precision/recall tradeoff across operating thresholds.

### 4.2 Box Matching

Ground truth boxes are matched to prediction boxes using greedy IoU matching:

1. All (GT box, prediction box) pairs are scored by IoU.
2. Pairs are sorted by IoU (highest first).
3. Each GT box and each prediction box may be matched at most once.
4. A pair is only considered a valid match if IoU >= the configured threshold (default: `0.1`). This is done to account for resizing of bounding boxes done in postprocessing for predictions, and to account for dataset box inaccuracies.

### 4.3 False Alert Rate (FAR)

The proportion of images where the model produced at least one prediction but no ground truth drone was present.

```
FAR = (images with predictions AND no GT) / total images * 100
```

### 4.4 Detection Accuracy

The proportion of drone detections that are correct, evaluated only on images where a ground truth box is present. False positives from images with no ground truth do not affect this metric.

```
false_positives = predicted boxes in GT images that did not match any GT box
Detection Accuracy = matched boxes / (total GT boxes + false_positives) * 100
```

### 4.5 Counting Accuracy

A global measure of how closely the total predicted box count matches the total ground truth box count across all images.

```
Counting Accuracy = max(0, 1 - |total predictions - total GT| / max(total GT, 1)) * 100
```

---

## 5. Combined Dataset Evaluation

Both datasets are loaded and merged into a single list of images before any metrics are computed. This means all reported numbers reflect aggregate performance across BirDrone and Anti-UAV together, covering both thermal and visible modalities simultaneously.

The script iterates over every (GT class, prediction class) combination and every confidence level, computing the three metrics above for each combination.

---

## 6. Benchmark Results

Evaluated with IoU threshold `0.1`, GT class `drone`, prediction classes `drone`.

### drone (GT) vs. drone (predictions)

| Level | FAR % | Detection % | Counting % |
|---|---|---|---|
| highest | 0.00 | 0.46 | 7.43 |
| high | 0.16 | 4.09 | 13.86 |
| medium | 2.07 | 76.92 | 86.63 |
| low | 2.23 | 80.63 | 80.69 |

### Notes on Results

- At `"medium"` confidence and above, the system achieves **76.92% detection accuracy** and **86.63% counting accuracy** with only a **2.07% false alert rate**, representing the best overall operating point.
- Raising the threshold to `"high"` or `"highest"` sharply reduces false alerts but causes a large drop in detection coverage, as many true drone boxes are assigned lower confidence tiers.
- The `"low"` level marginally improves detection accuracy over `"medium"` (+3.71%) but slightly reduces counting accuracy (-5.94%) due to additional spurious boxes being included.

---

## 7. Running the Analysis

```bash
python benchmark_analysis.py
```

Edit the parameters at the top of the script to change input files, IoU threshold, ground truth classes, or prediction classes.

**Dependencies:** Python 3.7+, standard library only (no external packages required).
