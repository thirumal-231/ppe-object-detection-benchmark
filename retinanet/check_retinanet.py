import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor
from torchvision.models.detection import (
    retinanet_resnet50_fpn,
    RetinaNet_ResNet50_FPN_Weights,
)
from torchvision.models.detection.retinanet import RetinaNetHead


# -----------------------------
# Configuration
# -----------------------------

NUM_CLASSES = 11

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Using device: {device}")


# -----------------------------
# Dataset
# -----------------------------

dataset = CocoDetection(
    root="../coco_dataset/train/images",
    annFile="../coco_dataset/train/annotations.json",
    transform=ToTensor(),
)

print(f"Training images: {len(dataset)}")


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
                y + h
            ])

            labels.append(ann["category_id"]-1)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
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
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn,
)

print("DataLoader created successfully!")


# -----------------------------
# Create RetinaNet
# -----------------------------

weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT

model = retinanet_resnet50_fpn(
    weights=weights
)

# RetinaNet has NO background class.
num_anchors = model.head.classification_head.num_anchors

model.head = RetinaNetHead(
    model.backbone.out_channels,
    num_anchors,
    NUM_CLASSES
)

model.to(device)

print("RetinaNet created successfully!")


# -----------------------------
# Get one batch
# -----------------------------

images, targets = next(iter(loader))

images = [image.to(device) for image in images]

targets = [
    {
        "boxes": target["boxes"].to(device),
        "labels": target["labels"].to(device),
    }
    for target in targets
]

print("Batch loaded successfully!")
print("First image shape:", images[0].shape)
print("First image boxes:", targets[0]["boxes"].shape)
print("First image labels:", targets[0]["labels"])


# -----------------------------
# Forward pass
# -----------------------------

model.train()

losses = model(images, targets)

print("\nForward pass successful!")

print("\nLosses:")

total_loss = 0

for name, loss in losses.items():
    print(f"{name}: {loss.item():.4f}")
    total_loss += loss

print(f"Total loss: {total_loss.item():.4f}")