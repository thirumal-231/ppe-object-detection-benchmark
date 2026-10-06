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

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

NUM_CLASSES = 11

BATCH_SIZE = 2
START_EPOCH = 2
END_EPOCH = 5

LR = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005


# -----------------------------
# Dataset
# -----------------------------

def get_transform():
    return ToTensor()


dataset = CocoDetection(
    root="../coco_dataset/train/images",
    annFile="../coco_dataset/train/annotations.json",
    transform=get_transform()
)

print(f"Using device: {DEVICE}")
print(f"Training images: {len(dataset)}")


# -----------------------------
# Collate function
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

            # COCO category IDs are 1-11
            # RetinaNet expects labels 0-10
            labels.append(ann["category_id"] - 1)

        target = {
            "boxes": torch.tensor(
                boxes,
                dtype=torch.float32
            ),

            "labels": torch.tensor(
                labels,
                dtype=torch.int64
            )
        }

        targets.append(target)

    return images, targets


# -----------------------------
# DataLoader
# -----------------------------

data_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)

print("DataLoader created successfully!")


# -----------------------------
# Create RetinaNet
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
# Load Epoch 1 checkpoint
# -----------------------------

print("Loading Epoch 1 checkpoint...")

checkpoint = torch.load(
    "retinanet_epoch1.pth",
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint)

model.to(DEVICE)

print("Epoch 1 checkpoint loaded!")


# -----------------------------
# Optimizer
# -----------------------------

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LR,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY
)


# -----------------------------
# Training
# -----------------------------

for epoch in range(START_EPOCH, END_EPOCH + 1):

    model.train()

    total_loss = 0.0

    for batch_idx, (images, targets) in enumerate(data_loader):

        images = [
            image.to(DEVICE)
            for image in images
        ]

        targets = [
            {
                key: value.to(DEVICE)
                for key, value in target.items()
            }
            for target in targets
        ]

        # Forward pass
        loss_dict = model(
            images,
            targets
        )

        loss = sum(
            loss_value
            for loss_value in loss_dict.values()
        )

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 50 == 0:
            print(
                f"Epoch {epoch} | "
                f"Batch {batch_idx + 1}/{len(data_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    average_loss = total_loss / len(data_loader)

    print()
    print(f"Epoch {epoch} complete")
    print(f"Average loss: {average_loss:.4f}")

    # Save checkpoint
    checkpoint_path = f"retinanet_epoch{epoch}.pth"

    torch.save(
        model.state_dict(),
        checkpoint_path
    )

    print(f"Model saved to {checkpoint_path}")
    print("-" * 50)


print("Training complete!")