import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor
from torchvision.models.detection import (
    retinanet_resnet50_fpn,
    RetinaNet_ResNet50_FPN_Weights,
)
from torchvision.models.detection.retinanet import RetinaNetHead
from pycocotools.cocoeval import COCOeval
import json


# -----------------------------
# Configuration
# -----------------------------

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

NUM_CLASSES = 11

MODEL_PATH = "retinanet_epoch5.pth"

CLASS_NAMES = [
    "helmet",
    "gloves",
    "vest",
    "boots",
    "goggles",
    "none",
    "person",
    "no_helmet",
    "no_goggle",
    "no_gloves",
    "no_boots",
]


# -----------------------------
# Dataset
# -----------------------------

dataset = CocoDetection(
    root="../coco_dataset/val/images",
    annFile="../coco_dataset/val/annotations.json",
    transform=ToTensor()
)

print(f"Using device: {DEVICE}")
print(f"Validation images: {len(dataset)}")


# -----------------------------
# Collate
# -----------------------------

def collate_fn(batch):

    images = []
    targets = []

    for image, annotations in batch:

        images.append(image)

        boxes = []
        labels = []

        for ann in annotations:

            x, y, w, h = ann["bbox"]

            boxes.append([
                x,
                y,
                x + w,
                y + h
            ])

            # COCO IDs: 1-11
            # RetinaNet labels: 0-10
            labels.append(ann["category_id"] - 1)

        targets.append({
            "boxes": torch.tensor(
                boxes,
                dtype=torch.float32
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.int64
            )
        })

    return images, targets


data_loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)


# -----------------------------
# Create model
# -----------------------------

print("Creating RetinaNet...")

weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT

model = retinanet_resnet50_fpn(
    weights=weights
)

num_anchors = model.head.classification_head.num_anchors

model.head = RetinaNetHead(
    model.backbone.out_channels,
    num_anchors,
    NUM_CLASSES
)


# -----------------------------
# Load trained model
# -----------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint)

model.to(DEVICE)

# Keep threshold low so COCOeval
# receives enough predictions.
model.score_thresh = 0.001
model.topk_candidates = 1000

model.eval()

print(f"Loaded model: {MODEL_PATH}")


# -----------------------------
# Inference
# -----------------------------

predictions = []

print("\nRunning inference...")

with torch.no_grad():

    for batch_idx, (images, targets) in enumerate(data_loader):

        images_device = [
            image.to(DEVICE)
            for image in images
        ]

        outputs = model(images_device)

        for image_idx, output in enumerate(outputs):

            image_id = dataset.ids[
                batch_idx * 2 + image_idx
            ]

            boxes = output["boxes"].cpu()
            scores = output["scores"].cpu()
            labels = output["labels"].cpu()

            for box, score, label in zip(
                boxes,
                scores,
                labels
            ):

                x1, y1, x2, y2 = box.tolist()

                width = x2 - x1
                height = y2 - y1

                # RetinaNet labels 0-10
                # COCO labels 1-11
                category_id = label.item() + 1

                predictions.append({
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [
                        x1,
                        y1,
                        width,
                        height
                    ],
                    "score": score.item()
                })

        if (batch_idx + 1) % 10 == 0:
            print(
                f"Processed "
                f"{batch_idx + 1}/{len(data_loader)} batches"
            )


# -----------------------------
# Save predictions
# -----------------------------

prediction_file = "retinanet_epoch5_predictions.json"

with open(prediction_file, "w") as f:
    json.dump(predictions, f)

print()
print(f"Total predictions: {len(predictions)}")
print(f"Saved to: {prediction_file}")


# -----------------------------
# COCO Evaluation
# -----------------------------

coco_gt = dataset.coco

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


# -----------------------------
# Overall metrics
# -----------------------------

print("\nOverall Results")

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


# -----------------------------
# Per-class AP
# -----------------------------

print("\nPer-Class AP")
print("-" * 50)

precision = coco_eval.eval["precision"]

# precision dimensions:
#
# [IoU, Recall, Category, Area, MaxDets]

for class_index, class_name in enumerate(CLASS_NAMES):

    class_precision = precision[
        :, :, class_index, 0, 2
    ]

    class_precision = class_precision[
        class_precision > -1
    ]

    if len(class_precision) == 0:
        ap = 0.0
    else:
        ap = class_precision.mean()

    # AP@50
    ap50_values = precision[
        0, :, class_index, 0, 2
    ]

    ap50_values = ap50_values[
        ap50_values > -1
    ]

    if len(ap50_values) == 0:
        ap50 = 0.0
    else:
        ap50 = ap50_values.mean()

    print(
        f"{class_name:<12} "
        f"AP@50:95 = {ap:.3f}   "
        f"AP@50 = {ap50:.3f}"
    )