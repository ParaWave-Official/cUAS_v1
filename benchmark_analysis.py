"""
Detection Metrics Calculator for Open Vocabulary Detection System

Calculates metrics for drone/bird detection predictions against ground truth.
Metrics are calculated per confidence level (cumulative from highest to lowest).
"""

import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


# ============================================================================
# CONFIGURATION
# ============================================================================

# List of JSON file paths to analyze
JSON_FILES = [
    "birdrone_final.json",
    "antiuav_final.json",
]

# Classes to extract from raw_data (ground truth)
GROUND_TRUTH_CLASSES = ["drone"]

# Classes to compare against in predictions
PREDICTION_CLASSES = ["drone"]

# Minimum IoU threshold for box matching (0.0 to 1.0)
IOU_THRESHOLD = 0.1

# Confidence levels in order (highest to lowest)
CONFIDENCE_LEVELS = ["highest", "high", "medium", "low"]


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class BBox:
    """Bounding box with normalized coordinates (0-1)"""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    level: str = None

    def iou(self, other: 'BBox') -> float:
        """Calculate Intersection over Union with another box"""
        x_left = max(self.x_min, other.x_min)
        y_top = max(self.y_min, other.y_min)
        x_right = min(self.x_max, other.x_max)
        y_bottom = min(self.y_max, other.y_max)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        box1_area = (self.x_max - self.x_min) * (self.y_max - self.y_min)
        box2_area = (other.x_max - other.x_min) * (other.y_max - other.y_min)
        union_area = box1_area + box2_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0


@dataclass
class ImageData:
    """Data for a single image"""
    filename: str
    gt_boxes: Dict[str, List[BBox]]   # class -> list of boxes
    pred_boxes: Dict[str, List[BBox]]  # class -> list of boxes


@dataclass
class MetricResult:
    """Results for a specific GT class + Pred class combination"""
    gt_class: str
    pred_class: str
    level: str

    # Counts
    total_images: int
    images_with_gt: int
    images_with_pred: int
    images_with_false_alerts: int  # images with pred but no GT

    total_gt_boxes: int
    total_pred_boxes: int
    matched_boxes: int

    # Metrics
    false_alert_rate: float   # percentage
    detection_accuracy: float  # percentage
    counting_accuracy: float  # percentage


# ============================================================================
# DATA LOADING
# ============================================================================

def load_json_data(filepath: str) -> List[ImageData]:
    """Load and parse JSON file into ImageData objects"""
    with open(filepath, 'r') as f:
        data = json.load(f)

    images = []
    for filename, content in data.items():
        # Parse ground truth boxes
        gt_boxes = defaultdict(list)
        for gt_item in content.get('raw_data', []):
            class_name = gt_item['class']
            for box_data in gt_item.get('boxes', []):
                bbox = BBox(
                    x_min=box_data['x_min'],
                    y_min=box_data['y_min'],
                    x_max=box_data['x_max'],
                    y_max=box_data['y_max']
                )
                gt_boxes[class_name].append(bbox)

        # Parse prediction boxes
        pred_boxes = defaultdict(list)
        for pred_item in content.get('predictions', []):
            class_name = pred_item['class']
            for box_data in pred_item.get('boxes', []):
                bbox = BBox(
                    x_min=box_data['x_min'],
                    y_min=box_data['y_min'],
                    x_max=box_data['x_max'],
                    y_max=box_data['y_max'],
                    level=box_data.get('level')
                )
                pred_boxes[class_name].append(bbox)

        images.append(ImageData(
            filename=filename,
            gt_boxes=dict(gt_boxes),
            pred_boxes=dict(pred_boxes)
        ))

    return images


# ============================================================================
# BOX MATCHING
# ============================================================================

def match_boxes(gt_boxes: List[BBox], pred_boxes: List[BBox],
                iou_threshold: float) -> Tuple[int, int, int]:
    """
    Match prediction boxes to ground truth boxes using greedy matching.

    Returns:
        (matched_count, unmatched_gt_count, unmatched_pred_count)
    """
    if not gt_boxes and not pred_boxes:
        return 0, 0, 0

    if not gt_boxes:
        return 0, 0, len(pred_boxes)

    if not pred_boxes:
        return 0, len(gt_boxes), 0

    matched_gt = set()
    matched_pred = set()
    matched_count = 0

    # Build IoU matrix and keep only pairs that meet the threshold
    iou_matrix = []
    for i, gt_box in enumerate(gt_boxes):
        for j, pred_box in enumerate(pred_boxes):
            iou = gt_box.iou(pred_box)
            if iou >= iou_threshold:
                iou_matrix.append((iou, i, j))

    # Sort by IoU (highest first) and greedily match
    iou_matrix.sort(reverse=True)

    for iou, gt_idx, pred_idx in iou_matrix:
        if gt_idx not in matched_gt and pred_idx not in matched_pred:
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)
            matched_count += 1

    unmatched_gt = len(gt_boxes) - len(matched_gt)
    unmatched_pred = len(pred_boxes) - len(matched_pred)

    return matched_count, unmatched_gt, unmatched_pred


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def filter_boxes_by_level(boxes: List[BBox], level: str) -> List[BBox]:
    """
    Filter boxes to include only those at or above the given confidence level.
    Levels are cumulative: 'medium' includes 'highest', 'high', and 'medium'.
    """
    level_order = {lv: i for i, lv in enumerate(CONFIDENCE_LEVELS)}
    target_level_idx = level_order.get(level, len(CONFIDENCE_LEVELS))

    filtered = []
    for box in boxes:
        if box.level is None:
            continue
        box_level_idx = level_order.get(box.level, len(CONFIDENCE_LEVELS))
        if box_level_idx <= target_level_idx:
            filtered.append(box)

    return filtered


def calculate_metrics(images: List[ImageData], gt_class: str, pred_class: str,
                      level: str, iou_threshold: float) -> MetricResult:
    """Calculate all metrics for a specific class combination and confidence level"""

    total_images = len(images)
    images_with_gt = 0
    images_with_pred = 0
    images_with_false_alerts = 0

    total_gt_boxes = 0
    total_pred_boxes = 0
    total_matched = 0

    # Tracked separately: only from images that have GT present
    gt_images_pred_boxes = 0
    gt_images_matched = 0

    for image in images:
        gt_boxes = image.gt_boxes.get(gt_class, [])

        all_pred_boxes = image.pred_boxes.get(pred_class, [])
        pred_boxes = filter_boxes_by_level(all_pred_boxes, level)

        has_gt = len(gt_boxes) > 0
        has_pred = len(pred_boxes) > 0

        if has_gt:
            images_with_gt += 1
        if has_pred:
            images_with_pred += 1
        if has_pred and not has_gt:
            images_with_false_alerts += 1

        # Global counts (used for counting accuracy and display)
        total_gt_boxes += len(gt_boxes)
        total_pred_boxes += len(pred_boxes)

        matched, _, _ = match_boxes(gt_boxes, pred_boxes, iou_threshold)
        total_matched += matched

        # Scoped counts for detection accuracy (GT images only)
        if has_gt:
            gt_images_pred_boxes += len(pred_boxes)
            gt_images_matched += matched

    # 1. False Alert Rate: images with pred but no GT / total images
    false_alert_rate = (images_with_false_alerts / total_images * 100) if total_images > 0 else 0.0

    # 2. Detection Accuracy: scoped to images with GT only.
    #    False positives = pred boxes in GT images that did not match any GT box.
    #    Denominator = GT boxes + false positives (both from GT images only).
    false_positives = gt_images_pred_boxes - gt_images_matched
    denominator = total_gt_boxes + false_positives
    detection_accuracy = (gt_images_matched / denominator * 100) if denominator > 0 else 0.0

    # 3. Counting Accuracy: based on absolute count error (global)
    #    100% * (1 - |pred_count - gt_count| / max(gt_count, 1))
    count_error = abs(total_pred_boxes - total_gt_boxes)
    max_count = max(total_gt_boxes, 1)
    counting_accuracy = max(0.0, 100.0 * (1 - count_error / max_count))

    return MetricResult(
        gt_class=gt_class,
        pred_class=pred_class,
        level=level,
        total_images=total_images,
        images_with_gt=images_with_gt,
        images_with_pred=images_with_pred,
        images_with_false_alerts=images_with_false_alerts,
        total_gt_boxes=total_gt_boxes,
        total_pred_boxes=total_pred_boxes,
        matched_boxes=total_matched,
        false_alert_rate=false_alert_rate,
        detection_accuracy=detection_accuracy,
        counting_accuracy=counting_accuracy
    )


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("=" * 80)
    print("DETECTION METRICS CALCULATOR")
    print("=" * 80)
    print()

    print(f"Loading {len(JSON_FILES)} JSON file(s)...")
    all_images = []
    for filepath in JSON_FILES:
        images = load_json_data(filepath)
        all_images.extend(images)
        print(f"  Loaded {len(images)} images from {filepath}")
    print(f"Total images: {len(all_images)}")
    print()

    print(f"Ground Truth Classes: {GROUND_TRUTH_CLASSES}")
    print(f"Prediction Classes:   {PREDICTION_CLASSES}")
    print(f"IoU Threshold:        {IOU_THRESHOLD}")
    print(f"Confidence Levels:    {CONFIDENCE_LEVELS}")
    print()
    print("=" * 80)
    print()

    results = []
    for gt_class in GROUND_TRUTH_CLASSES:
        for pred_class in PREDICTION_CLASSES:
            print(f"CLASS COMBINATION: GT='{gt_class}' vs PRED='{pred_class}'")
            print("-" * 80)

            for level in CONFIDENCE_LEVELS:
                result = calculate_metrics(
                    all_images, gt_class, pred_class, level, IOU_THRESHOLD
                )
                results.append(result)

                print(f"\n  Confidence Level: {level.upper()}")
                print(f"  {'─' * 76}")
                print(f"    Images:")
                print(f"      Total:            {result.total_images}")
                print(f"      With GT:          {result.images_with_gt}")
                print(f"      With Predictions: {result.images_with_pred}")
                print(f"      False Alerts:     {result.images_with_false_alerts}")
                print()
                print(f"    Boxes:")
                print(f"      Ground Truth:              {result.total_gt_boxes}")
                print(f"      Predictions:               {result.total_pred_boxes}")
                print(f"      Matched (IoU >= {IOU_THRESHOLD}):  {result.matched_boxes}")
                print()
                print(f"    METRICS:")
                print(f"      False Alert Rate:    {result.false_alert_rate:6.2f}%")
                print(f"      Detection Accuracy:  {result.detection_accuracy:6.2f}%")
                print(f"      Counting Accuracy:   {result.counting_accuracy:6.2f}%")

            print()
            print("=" * 80)
            print()

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print()
    print(f"{'GT Class':<12} {'Pred Class':<12} {'Level':<10} {'FAR %':<8} {'Det %':<8} {'Count %':<8}")
    print("-" * 80)
    for result in results:
        print(f"{result.gt_class:<12} {result.pred_class:<12} {result.level:<10} "
              f"{result.false_alert_rate:6.2f}  {result.detection_accuracy:6.2f}  "
              f"{result.counting_accuracy:6.2f}")
    print()
    print("=" * 80)
    print()
    print("Legend:")
    print("  FAR %   = False Alert Rate (images with pred but no GT / total images)")
    print("  Det %   = Detection Accuracy (matched boxes / (GT boxes + false positives))")
    print("  Count % = Counting Accuracy (100% x (1 - |pred - GT| / max(GT, 1)))")
    print()
    print("Note: Each confidence level includes all boxes at that level and above.")
    print("      E.g., 'medium' includes 'highest', 'high', and 'medium' boxes.")
    print("      Detection Accuracy is scoped to images with GT present only;")
    print("      false positives from GT-free images do not affect this metric.")
    print()


if __name__ == "__main__":
    main()