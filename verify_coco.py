import json
from pathlib import Path
from PIL import Image, ImageDraw


SPLIT = "train"

IMAGE_DIR = Path("coco_dataset") / SPLIT / "images"
ANNOTATION_FILE = Path("coco_dataset") / SPLIT / "annotations.json"

OUTPUT_FILE = Path("coco_verification.jpg")


# Load COCO annotations
with open(ANNOTATION_FILE, "r") as f:
    coco = json.load(f)


# Create lookup tables
images = {
    image["id"]: image
    for image in coco["images"]
}

categories = {
    category["id"]: category["name"]
    for category in coco["categories"]
}

annotations_by_image = {}

for annotation in coco["annotations"]:
    image_id = annotation["image_id"]

    annotations_by_image.setdefault(image_id, []).append(annotation)


# Pick the first image
image_info = coco["images"][0]

image_path = IMAGE_DIR / image_info["file_name"]

image = Image.open(image_path).convert("RGB")

draw = ImageDraw.Draw(image)


# Draw annotations
for annotation in annotations_by_image.get(image_info["id"], []):

    x, y, width, height = annotation["bbox"]

    x2 = x + width
    y2 = y + height

    category_name = categories[annotation["category_id"]]

    draw.rectangle(
        [x, y, x2, y2],
        outline="red",
        width=3
    )

    draw.text(
        (x, max(0, y - 15)),
        category_name,
        fill="red"
    )


# Save result
image.save(OUTPUT_FILE)

print(f"Image: {image_info['file_name']}")
print(f"Saved verification image to: {OUTPUT_FILE}")