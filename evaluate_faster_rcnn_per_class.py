import json
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# -----------------------------------
# Paths
# -----------------------------------

annotation_file = (
    "coco_dataset/val/annotations.json"
)

prediction_file = (
    "fasterrcnn_predictions.json"
)


# -----------------------------------
# Classes
# -----------------------------------

classes = {
    1: "helmet",
    2: "gloves",
    3: "vest",
    4: "boots",
    5: "goggles",
    6: "none",
    7: "person",
    8: "no_helmet",
    9: "no_goggle",
    10: "no_gloves",
    11: "no_boots"
}


# -----------------------------------
# Load COCO data
# -----------------------------------

coco_gt = COCO(annotation_file)

with open(prediction_file, "r") as file:
    predictions = json.load(file)

coco_dt = coco_gt.loadRes(predictions)


# -----------------------------------
# Evaluate each class
# -----------------------------------

print("\n")
print("=" * 55)
print("           FASTER R-CNN PER-CLASS AP")
print("=" * 55)

print(
    f"{'Class':<15}"
    f"{'AP@50:95':>12}"
    f"{'AP@50':>12}"
)

print("-" * 55)


for category_id, class_name in classes.items():

    evaluator = COCOeval(
        coco_gt,
        coco_dt,
        "bbox"
    )

    evaluator.params.catIds = [category_id]

    evaluator.evaluate()
    evaluator.accumulate()

    precision = evaluator.eval["precision"]

    # Dimensions:
    #
    # [IoU, Recall, Category, Area, MaxDets]
    #
    # We only have one category here.

    # -----------------------------------
    # mAP @ 0.50:0.95
    # -----------------------------------

    precision_all = precision[
        :, :, 0, 0, 2
    ]

    valid_precision = (
        precision_all[
            precision_all > -1
        ]
    )

    if len(valid_precision) > 0:

        ap_5095 = np.mean(
            valid_precision
        )

    else:

        ap_5095 = float("nan")


    # -----------------------------------
    # AP @ 0.50
    # -----------------------------------

    precision_50 = precision[
        0, :, 0, 0, 2
    ]

    valid_precision_50 = (
        precision_50[
            precision_50 > -1
        ]
    )

    if len(valid_precision_50) > 0:

        ap_50 = np.mean(
            valid_precision_50
        )

    else:

        ap_50 = float("nan")


    # -----------------------------------
    # Print
    # -----------------------------------

    print(
        f"{class_name:<15}"
        f"{ap_5095:>12.3f}"
        f"{ap_50:>12.3f}"
    )


print("-" * 55)
print("Evaluation complete!")
print("=" * 55)