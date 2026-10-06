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
BATCH_SIZE = 2

START_EPOCH = 6
END_EPOCH = 10

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
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn,
)

print("DataLoader created successfully!")


# -----------------------------
# Model
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

model.to(device)

print("RetinaNet created successfully!")


# -----------------------------
# Load Epoch 5 checkpoint
# -----------------------------

checkpoint_path = "retinanet_epoch5.pth"

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint)

print(
    f"Loaded checkpoint: {checkpoint_path}"
)


# -----------------------------
# Optimizer
# -----------------------------

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005,
)


# -----------------------------
# Training
# -----------------------------

for epoch in range(
    START_EPOCH,
    END_EPOCH + 1
):

    model.train()

    epoch_loss = 0.0

    print(
        f"\nStarting Epoch {epoch}/{END_EPOCH}..."
    )

    for batch_idx, (images, targets) in enumerate(loader):

        # Move images to device

        images = [
            image.to(device)
            for image in images
        ]

        # Move targets to device

        targets = [
            {
                "boxes": target["boxes"].to(device),
                "labels": target["labels"].to(device),
            }

            for target in targets
        ]

        # Clear gradients

        optimizer.zero_grad()

        # Forward pass

        loss_dict = model(
            images,
            targets
        )

        # Total loss

        losses = sum(
            loss
            for loss in loss_dict.values()
        )

        # Backpropagation

        losses.backward()

        # Update weights

        optimizer.step()

        # Get loss value

        loss_value = losses.item()

        epoch_loss += loss_value

        # Print progress

        if (batch_idx + 1) % 10 == 0:

            print(
                f"Epoch {epoch} "
                f"Batch {batch_idx + 1}/{len(loader)} "
                f"| Loss: {loss_value:.4f}"
            )

    # Average epoch loss

    average_loss = (
        epoch_loss / len(loader)
    )

    print(
        f"\nEpoch {epoch} complete"
    )

    print(
        f"Average loss: {average_loss:.4f}"
    )


    # -----------------------------
    # Save checkpoint
    # -----------------------------

    checkpoint_path = (
        f"retinanet_epoch{epoch}.pth"
    )

    torch.save(
        model.state_dict(),
        checkpoint_path
    )

    print(
        f"Model saved to {checkpoint_path}"
    )


# -----------------------------
# Training complete
# -----------------------------

print(
    "\nRetinaNet training complete!"
)

print(
    "Checkpoints saved from "
    "epoch 6 through epoch 10."
)