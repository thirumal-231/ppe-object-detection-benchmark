import torch
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# -----------------------------------
# Device
# -----------------------------------

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", device)


# -----------------------------------
# Paths
# -----------------------------------

annotation_file = (
    "coco_dataset/val/annotations.json"
)

image_root = (
    "coco_dataset/val/images"
)

model_path = (
    "fasterrcnn_epoch10.pth"
)


# -----------------------------------
# Dataset
# -----------------------------------

dataset = CocoDetection(
    root=image_root,
    annFile=annotation_file,
    transform=ToTensor()
)

print(
    "Validation images:",
    len(dataset)
)


# -----------------------------------
# Model
# -----------------------------------

num_classes = 12

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

model = fasterrcnn_resnet50_fpn(
    weights=weights
)

in_features = (
    model.roi_heads
    .box_predictor
    .cls_score
    .in_features
)

model.roi_heads.box_predictor = (
    FastRCNNPredictor(
        in_features,
        num_classes
    )
)


# -----------------------------------
# Load trained model
# -----------------------------------

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

model.to(device)
model.eval()

print("Model loaded successfully!")


# -----------------------------------
# COCO ground truth
# -----------------------------------

coco_gt = COCO(annotation_file)


# -----------------------------------
# Generate predictions
# -----------------------------------

results = []

print("\nRunning inference...")

with torch.no_grad():

    for index in range(len(dataset)):

        image_tensor, annotations = dataset[index]

        # Get actual COCO image ID
        image_id = dataset.ids[index]

        image_input = image_tensor.to(device)

        prediction = model(
            [image_input]
        )[0]

        boxes = prediction["boxes"].cpu()
        labels = prediction["labels"].cpu()
        scores = prediction["scores"].cpu()

        for box, label, score in zip(
            boxes,
            labels,
            scores
        ):

            x1, y1, x2, y2 = box.tolist()

            width = x2 - x1
            height = y2 - y1

            results.append({
                "image_id": int(image_id),

                "category_id": int(
                    label.item()
                ),

                "bbox": [
                    x1,
                    y1,
                    width,
                    height
                ],

                "score": float(
                    score.item()
                )
            })

        if (index + 1) % 10 == 0:

            print(
                f"Processed "
                f"{index + 1}/{len(dataset)}"
            )


# -----------------------------------
# Save predictions
# -----------------------------------

prediction_file = (
    "fasterrcnn_predictions.json"
)

import json

with open(
    prediction_file,
    "w"
) as file:

    json.dump(
        results,
        file
    )

print(
    f"\nPredictions saved to "
    f"{prediction_file}"
)


# -----------------------------------
# COCO Evaluation
# -----------------------------------

print("\nRunning COCO evaluation...")


coco_dt = coco_gt.loadRes(
    prediction_file
)

coco_eval = COCOeval(
    coco_gt,
    coco_dt,
    "bbox"
)

coco_eval.evaluate()

coco_eval.accumulate()

coco_eval.summarize()


# -----------------------------------
# Print important metrics
# -----------------------------------

print("\n========== RESULTS ==========")

print(
    f"mAP @ 0.50:0.95 : "
    f"{coco_eval.stats[0]:.4f}"
)

print(
    f"mAP @ 0.50      : "
    f"{coco_eval.stats[1]:.4f}"
)

print(
    f"mAP @ 0.75      : "
    f"{coco_eval.stats[2]:.4f}"
)

print(
    "============================")