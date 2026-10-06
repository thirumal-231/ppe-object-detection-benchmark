import torch

from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)

from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor
)


# -----------------------------------
# Device
# -----------------------------------

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", device)


# -----------------------------------
# Dataset
# -----------------------------------

dataset = CocoDetection(
    root="coco_dataset/train/images",
    annFile="coco_dataset/train/annotations.json",
    transform=ToTensor()
)

print("Training images:", len(dataset))


# -----------------------------------
# Collate function
# -----------------------------------

def collate_fn(batch):

    images = []
    targets = []

    for image, annotations in batch:

        boxes = []
        labels = []

        for annotation in annotations:

            x, y, width, height = annotation["bbox"]

            x1 = x
            y1 = y
            x2 = x + width
            y2 = y + height

            boxes.append([
                x1,
                y1,
                x2,
                y2
            ])

            labels.append(
                annotation["category_id"]
            )

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

        images.append(image)
        targets.append(target)

    return images, targets


# -----------------------------------
# DataLoader
# -----------------------------------

train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0
)

print("DataLoader created successfully!")


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

model.to(device)

print("Faster R-CNN created successfully!")


# -----------------------------------
# Load Epoch 5 checkpoint
# -----------------------------------

checkpoint_path = "fasterrcnn_epoch5.pth"

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint)

print(
    f"Loaded checkpoint: {checkpoint_path}"
)


# -----------------------------------
# Optimizer
# -----------------------------------

params = [
    parameter
    for parameter in model.parameters()
    if parameter.requires_grad
]

optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)


# -----------------------------------
# Train
# -----------------------------------

model.train()

start_epoch = 6
end_epoch = 10

for epoch in range(start_epoch, end_epoch + 1):

    total_loss = 0.0

    print(
        f"\nStarting epoch {epoch}/{end_epoch}"
    )

    for batch_index, (images, targets) in enumerate(train_loader):

        images = [
            image.to(device)
            for image in images
        ]

        targets = [
            {
                key: value.to(device)
                for key, value in target.items()
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
        loss = sum(
            loss_value
            for loss_value in loss_dict.values()
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()

        # Print progress
        if (batch_index + 1) % 10 == 0:

            print(
                f"Batch "
                f"{batch_index + 1}/{len(train_loader)} "
                f"| Loss: {loss.item():.4f}"
            )

    average_loss = (
        total_loss / len(train_loader)
    )

    print(
        f"\nEpoch {epoch} complete"
    )

    print(
        f"Average loss: {average_loss:.4f}"
    )


    # -----------------------------------
    # Save checkpoint
    # -----------------------------------

    checkpoint_name = (
        f"fasterrcnn_epoch{epoch}.pth"
    )

    torch.save(
        model.state_dict(),
        checkpoint_name
    )

    print(
        f"Model saved to {checkpoint_name}"
    )


print("\nTraining complete!")