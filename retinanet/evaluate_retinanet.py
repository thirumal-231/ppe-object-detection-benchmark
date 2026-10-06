import json
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


# -----------------------------
# Configuration
# -----------------------------

NUM_CLASSES = 11
MODEL_PATH = "retinanet_epoch10.pth"

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Using device: {device}")


# -----------------------------
# Dataset
# -----------------------------

dataset = CocoDetection(
    root="../coco_dataset/val/images",
    annFile="../coco_dataset/val/annotations.json",
    transform=ToTensor(),
)

print(f"Validation images: {len(dataset)}")


# -----------------------------
# Collate function
# -----------------------------

def collate_fn(batch):

    images = []
    targets = []

    for image, annotations in batch:

        boxes = []
        labels = []

        for ann in annotations:

            x, y, w, h = ann["bbox"]

            boxes.append([
                x,
                y,
                x + w,
                y + h,
            ])

            # COCO category IDs:
            # 1-11
            #
            # RetinaNet labels:
            # 0-10
            labels.append(
                ann["category_id"] - 1
            )

        target = {
            "boxes": torch.tensor(
                boxes,
                dtype=torch.float32
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.int64
            ),
        }

        images.append(image)
        targets.append(target)

    return images, targets


# -----------------------------
# DataLoader
# -----------------------------

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn,
)


# -----------------------------
# Create RetinaNet
# -----------------------------

weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT

model = retinanet_resnet50_fpn(
    weights=weights
)

num_anchors = (
    model.head
    .classification_head
    .num_anchors
)

model.head = RetinaNetHead(
    model.backbone.out_channels,
    num_anchors,
    NUM_CLASSES,
)


# -----------------------------
# Load trained checkpoint
# -----------------------------

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model.to(device)

# Lower threshold for early checkpoint.
model.score_thresh = 0.001

# Allow more candidate predictions.
model.topk_candidates = 1000

model.eval()

print(
    f"Loaded model: {MODEL_PATH}"
)


# -----------------------------
# COCO ground truth
# -----------------------------

coco_gt = dataset.coco

results = []

print("\nRunning inference...\n")


# -----------------------------
# Inference
# -----------------------------

with torch.no_grad():

    for batch_idx, (images, targets) in enumerate(loader):

        images_device = [
            image.to(device)
            for image in images
        ]

        outputs = model(images_device)


        # -------------------------
        # Debug first batch
        # -------------------------

        if batch_idx == 0:

            print(
                "First batch detections:",
                [
                    len(output["boxes"])
                    for output in outputs
                ]
            )

            if (
                len(outputs) > 0
                and len(outputs[0]["scores"]) > 0
            ):

                print(
                    "Top scores:",
                    outputs[0]["scores"][:10]
                    .detach()
                    .cpu()
                )


        # -------------------------
        # Process predictions
        # -------------------------

        start_idx = batch_idx * 2

        for i, output in enumerate(outputs):

            dataset_idx = start_idx + i

            if dataset_idx >= len(dataset):
                continue

            image_id = dataset.ids[dataset_idx]

            boxes = (
                output["boxes"]
                .detach()
                .cpu()
            )

            scores = (
                output["scores"]
                .detach()
                .cpu()
            )

            labels = (
                output["labels"]
                .detach()
                .cpu()
            )


            # -------------------------
            # Convert predictions
            # -------------------------

            for box, score, label in zip(
                boxes,
                scores,
                labels
            ):

                # RetinaNet labels:
                # 0-10
                #
                # COCO category IDs:
                # 1-11

                category_id = (
                    int(label.item()) + 1
                )

                x1, y1, x2, y2 = (
                    box.tolist()
                )

                results.append({
                    "image_id": int(image_id),

                    "category_id": category_id,

                    "bbox": [
                        x1,
                        y1,
                        x2 - x1,
                        y2 - y1,
                    ],

                    "score": float(
                        score.item()
                    ),
                })


        # -------------------------
        # Progress
        # -------------------------

        if (batch_idx + 1) % 10 == 0:

            print(
                f"Processed "
                f"{batch_idx + 1}/"
                f"{len(loader)} batches"
            )


# -----------------------------
# Check predictions
# -----------------------------

print(
    f"\nTotal predictions: "
    f"{len(results)}"
)


if len(results) == 0:

    raise RuntimeError(
        "No predictions were generated. "
        "Check the model checkpoint and "
        "score threshold."
    )


# -----------------------------
# Save predictions
# -----------------------------

prediction_file = (
    "retinanet_predictions.json"
)

with open(
    prediction_file,
    "w"
) as f:

    json.dump(
        results,
        f
    )

print(
    f"Predictions saved to "
    f"{prediction_file}"
)


# -----------------------------
# COCO evaluation
# -----------------------------

coco_dt = coco_gt.loadRes(
    prediction_file
)

evaluator = COCOeval(
    coco_gt,
    coco_dt,
    "bbox"
)

evaluator.evaluate()

evaluator.accumulate()

evaluator.summarize()