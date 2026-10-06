import torch

from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


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

            boxes.append([
                x,
                y,
                x + width,
                y + height
            ])

            labels.append(
                annotation["category_id"]
            )

        boxes = torch.tensor(
            boxes,
            dtype=torch.float32
        ).reshape(-1, 4)

        labels = torch.tensor(
            labels,
            dtype=torch.int64
        )

        targets.append({
            "boxes": boxes,
            "labels": labels
        })

        images.append(image)

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

model.roi_heads.box_predictor = FastRCNNPredictor(
    in_features,
    num_classes
)


# -----------------------------------
# Load Epoch 1
# -----------------------------------

model.load_state_dict(
    torch.load(
        "fasterrcnn_epoch1.pth",
        map_location=device
    )
)

model.to(device)

print("Loaded Epoch 1 checkpoint!")


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
# Continue training
# -----------------------------------

start_epoch = 2
end_epoch = 5

for epoch in range(
    start_epoch,
    end_epoch + 1
):

    model.train()

    total_loss = 0.0

    print(
        f"\n========== Epoch {epoch}/5 =========="
    )

    for batch_index, (images, targets) in enumerate(
        train_loader
    ):

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

        optimizer.zero_grad()

        loss_dict = model(
            images,
            targets
        )

        loss = sum(
            loss_value
            for loss_value in loss_dict.values()
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if (batch_index + 1) % 50 == 0:

            print(
                f"Batch "
                f"{batch_index + 1}/"
                f"{len(train_loader)} "
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

    checkpoint_path = (
        f"fasterrcnn_epoch{epoch}.pth"
    )

    torch.save(
        model.state_dict(),
        checkpoint_path
    )

    print(
        f"Saved: {checkpoint_path}"
    )


print("\nTraining complete!")