import os
import torch
from PIL import Image, ImageDraw, ImageFont

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
# Classes
# -----------------------------------

classes = [
    "background",
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
    "no_boots"
]


# -----------------------------------
# Validation dataset
# -----------------------------------

dataset = CocoDetection(
    root="coco_dataset/val/images",
    annFile="coco_dataset/val/annotations.json",
    transform=ToTensor()
)

print("Validation images:", len(dataset))


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
# Load trained weights
# -----------------------------------

model.load_state_dict(
    torch.load(
        "fasterrcnn_epoch10.pth",
        map_location=device
    )
)

model.to(device)

model.eval()

print("Model loaded successfully!")


# -----------------------------------
# Output directory
# -----------------------------------

os.makedirs(
    "fasterrcnn_predictions",
    exist_ok=True
)


# -----------------------------------
# Run inference
# -----------------------------------

num_images = min(10, len(dataset))

print(
    f"Running inference on {num_images} validation images..."
)


with torch.no_grad():

    for index in range(num_images):

        image_tensor, annotations = dataset[index]

        image_input = image_tensor.to(device)

        predictions = model(
            [image_input]
        )[0]

        boxes = predictions["boxes"].cpu()
        labels = predictions["labels"].cpu()
        scores = predictions["scores"].cpu()

        # Original image
        image = (
            image_tensor
            .permute(1, 2, 0)
            .mul(255)
            .byte()
            .numpy()
        )

        image = Image.fromarray(image)

        draw = ImageDraw.Draw(image)

        # -----------------------------------
        # Draw predictions
        # -----------------------------------

        detection_count = 0

        for box, label, score in zip(
            boxes,
            labels,
            scores
        ):

            # Only show confident predictions
            if score < 0.5:
                continue

            x1, y1, x2, y2 = box.tolist()

            class_name = classes[label.item()]

            text = (
                f"{class_name} "
                f"{score:.2f}"
            )

            draw.rectangle(
                [x1, y1, x2, y2],
                outline="red",
                width=3
            )

            draw.text(
                [x1, max(0, y1 - 15)],
                text,
                fill="red"
            )

            detection_count += 1

        # -----------------------------------
        # Save image
        # -----------------------------------

        output_path = (
            f"fasterrcnn_predictions/"
            f"image_{index:03d}.jpg"
        )

        image.save(output_path)

        print(
            f"[{index + 1}/{num_images}] "
            f"{detection_count} detections -> "
            f"{output_path}"
        )


print("\nValidation inference complete!")