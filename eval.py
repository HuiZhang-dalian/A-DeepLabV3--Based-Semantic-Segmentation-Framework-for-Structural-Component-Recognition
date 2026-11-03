#!/usr/bin/env python3
"""Evaluate point-level S3DIS predictions stored as CSV files.

Required CSV columns: point_id,label
Optional CSV column: source_views (ignored by this evaluator)
Raw ADE20K predictions are converted to six fixed S3DIS IDs; all other
prediction labels become 0. Ground-truth labels retain their 13-class encoding.
"""

import argparse
import csv
import json
from pathlib import Path


ADE20K_TO_S3DIS = {
    1: 9,    # wall
    4: 7,    # floor
    6: 3,    # ceiling
    9: 10,   # window
    15: 6,   # door
    43: 11,  # column
}
DEFAULT_SIX_CLASS_IDS = [9, 7, 3, 10, 6, 11]


def convert_prediction_label(label, label_space="ade20k"):
    """Convert a prediction once; reserve 0 for all non-target predictions."""
    if label_space == "ade20k":
        return ADE20K_TO_S3DIS.get(label, 0)
    if label_space == "s3dis":
        return label if label in DEFAULT_SIX_CLASS_IDS else 0
    raise ValueError(f"Unsupported prediction label space: {label_space}")


def parse_id_list(value):
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("IDs must be comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("At least one ID is required")
    if len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("IDs must not contain duplicates")
    return result


def read_label_csv(path, source_name):
    labels = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"point_id", "label"}.issubset(reader.fieldnames):
            raise ValueError(
                f"{source_name} CSV must contain 'point_id' and 'label' columns: {path}"
            )
        for line_number, row in enumerate(reader, start=2):
            point_id = (row.get("point_id") or "").strip()
            label_text = (row.get("label") or "").strip()
            if not point_id:
                raise ValueError(f"Empty point_id at {path}:{line_number}")
            if not label_text:
                raise ValueError(f"Empty label at {path}:{line_number}")
            try:
                label = int(label_text)
            except ValueError as exc:
                raise ValueError(
                    f"Non-integer label {label_text!r} at {path}:{line_number}"
                ) from exc
            if point_id in labels:
                raise ValueError(
                    f"Duplicate point_id {point_id!r} in {source_name} CSV: {path}"
                )
            labels[point_id] = label
    return labels


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def mean_defined(values):
    defined = [value for value in values if value is not None]
    return sum(defined) / len(defined) if defined else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-csv", required=True, type=Path, help="Fused prediction CSV")
    parser.add_argument("--gt-csv", required=True, type=Path, help="Ground-truth CSV")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--class-ids", type=parse_id_list, required=True,
        help="13 nonzero ground-truth IDs, including fixed IDs 9,7,3,10,6,11",
    )
    parser.add_argument(
        "--class-names", required=True,
        help="Comma-separated names in the same order as --class-ids",
    )
    parser.add_argument(
        "--six-class-ids", type=parse_id_list, default=DEFAULT_SIX_CLASS_IDS,
        help="Fixed six target IDs (default: 9,7,3,10,6,11)",
    )
    parser.add_argument(
        "--pred-label-space", choices=("ade20k", "s3dis"), default="ade20k",
        help="Use ade20k for raw predictions, or s3dis for predictions already converted",
    )
    args = parser.parse_args()

    class_ids = args.class_ids
    class_names = [name.strip() for name in args.class_names.split(",")]
    if len(class_ids) != 13 or 0 in class_ids:
        parser.error("--class-ids must contain exactly 13 unique nonzero IDs; 0 is reserved")
    if len(class_names) != len(class_ids) or any(not name for name in class_names):
        parser.error("--class-names must provide one non-empty name per --class-ids entry")
    if not set(args.six_class_ids).issubset(class_ids):
        parser.error("Every --six-class-ids value must also appear in --class-ids")
    if set(args.six_class_ids) != set(DEFAULT_SIX_CLASS_IDS):
        parser.error("--six-class-ids must contain the fixed IDs 9,7,3,10,6,11")

    pred_labels = read_label_csv(args.pred_csv, "prediction")
    gt_labels = read_label_csv(args.gt_csv, "ground-truth")
    converted_predictions = {
        point_id: convert_prediction_label(label, args.pred_label_space)
        for point_id, label in pred_labels.items()
    }

    id_to_index = {class_id: index for index, class_id in enumerate(class_ids)}
    class_name_by_id = dict(zip(class_ids, class_names))
    confusion = [[0 for _ in class_ids] for _ in class_ids]
    support = [0 for _ in class_ids]
    missing_prediction_points = 0
    other_prediction_points = 0
    invalid_ground_truth_points = 0

    for point_id, gt_label in gt_labels.items():
        if gt_label not in id_to_index:
            invalid_ground_truth_points += 1
            continue

        gt_index = id_to_index[gt_label]
        support[gt_index] += 1
        pred_label = converted_predictions.get(point_id)
        if pred_label is None:
            missing_prediction_points += 1
            continue

        if pred_label == 0:
            other_prediction_points += 1
            continue
        confusion[gt_index][id_to_index[pred_label]] += 1

    per_class = []
    iou_by_id = {}
    for index, class_id in enumerate(class_ids):
        tp = confusion[index][index]
        fp = sum(confusion[row][index] for row in range(len(class_ids))) - tp
        fn = support[index] - tp
        union = tp + fp + fn
        iou = tp / union if union else None
        iou_by_id[class_id] = iou
        per_class.append({
            "class_id": class_id,
            "class_name": class_names[index],
            "support": support[index],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "iou": iou,
        })

    six_class_names = [class_name_by_id[class_id] for class_id in args.six_class_ids]
    miou_all = mean_defined([iou_by_id[class_id] for class_id in class_ids])
    miou_six = mean_defined([iou_by_id[class_id] for class_id in args.six_class_ids])
    valid_gt_points = sum(support)
    matched_gt_ids = set(gt_labels).intersection(pred_labels)
    extra_prediction_points = len(set(pred_labels).difference(gt_labels))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "converted_prediction_labels.csv",
        ["point_id", "label"],
        converted_predictions.items(),
    )
    write_csv(
        args.output_dir / "confusion_matrix.csv",
        ["ground_truth\\prediction", *class_names],
        [[class_names[row], *confusion[row]] for row in range(len(class_ids))],
    )
    write_csv(
        args.output_dir / "per_class_iou.csv",
        ["class_id", "class_name", "support", "tp", "fp", "fn", "iou"],
        [[
            item["class_id"], item["class_name"], item["support"], item["tp"],
            item["fp"], item["fn"], "" if item["iou"] is None else f'{item["iou"]:.12f}',
        ] for item in per_class],
    )

    metrics = {
        "prediction_csv": str(args.pred_csv),
        "ground_truth_csv": str(args.gt_csv),
        "prediction_label_space": args.pred_label_space,
        "ade20k_to_s3dis_mapping": ADE20K_TO_S3DIS,
        "other_prediction_label": 0,
        "class_ids": class_ids,
        "class_names": class_names,
        "six_class_ids": args.six_class_ids,
        "six_class_names": six_class_names,
        "valid_ground_truth_points": valid_gt_points,
        "matched_ground_truth_ids": len(matched_gt_ids),
        "missing_prediction_points": missing_prediction_points,
        "other_prediction_points": other_prediction_points,
        "extra_prediction_points_not_in_ground_truth": extra_prediction_points,
        "invalid_ground_truth_points_excluded": invalid_ground_truth_points,
        "miou_13_class": miou_all,
        "miou_6_class": miou_six,
        "per_class": per_class,
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print(f"Evaluated valid ground-truth points: {valid_gt_points}")
    print(f"Missing prediction points: {missing_prediction_points}")
    print(f"Other/unmapped predictions converted to 0: {other_prediction_points}")
    print(f"13-class mIoU: {miou_all:.6f}" if miou_all is not None else "13-class mIoU: undefined")
    print(f"6-class mIoU: {miou_six:.6f}" if miou_six is not None else "6-class mIoU: undefined")
    print(f"Results written to: {args.output_dir}")


if __name__ == "__main__":
    main()
