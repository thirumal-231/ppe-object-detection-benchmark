import json
import shutil
from pathlib import Path
from PIL import Image


# -----------------------------
# Configuration
# -----------------------------

DATASET_DIR = Path("dataset")
OUTPUT_DIR = Path("coco_dataset")

CLASSES = {
    0: "helmet",
    1: "gloves",
    2: "vest",
    3: "boots",
    4: "goggles",
    5: "none",
    6: "person",
    7: "no_helmet",
    8: "no_goggle",
    9: "no_gloves",
    10: "no_boots",
}


# -----------------------------
# Convert one split
# -----------------------------

def convert_split(split):
    image_dir = DATASET_DIR / "images" / split
    label_dir = DATASET_DIR / "labels" / split

    output_image_dir = OUTPUT_DIR / split / "images"
    output_image_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []

    image_id = 1
    annotation_id = 1

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    image_files = [
        file for file in image_dir.iterdir()
        if file.suffix.lower() in image_extensions
    ]

    for image_file in image_files:

        # -----------------------------
        # Read image dimensions
        # -----------------------------

        with Image.open(image_file) as image:
            width, height = image.size

        images.append({
            "id": image_id,
            "file_name": image_file.name,
            "width": width,
            "height": height
        })

        # -----------------------------
        # Copy image
        # -----------------------------

        shutil.copy2(
            image_file,
            output_image_dir / image_file.name
        )

        # -----------------------------
        # Read YOLO annotation
        # -----------------------------

        label_file = label_dir / f"{image_file.stem}.txt"

        if label_file.exists():

            with open(label_file, "r") as file:
                lines = file.readlines()

            for line in lines:

                line = line.strip()

                if not line:
                    continue

                class_id, x_center, y_center, box_width, box_height = map(
                    float,
                    line.split()
                )

                class_id = int(class_id)

                # -----------------------------
                # YOLO normalized coordinates
                # → pixel coordinates
                # -----------------------------

                x_center *= width
                y_center *= height
                box_width *= width
                box_height *= height

                # -----------------------------
                # Convert center coordinates
                # → top-left coordinates
                # -----------------------------

                x_min = x_center - (box_width / 2)
                y_min = y_center - (box_height / 2)

                # Keep boxes inside image boundaries
                x_min = max(0, x_min)
                y_min = max(0, y_min)

                box_width = min(
                    box_width,
                    width - x_min
                )

                box_height = min(
                    box_height,
                    height - y_min
                )

                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "bbox": [
                        x_min,
                        y_min,
                        box_width,
                        box_height
                    ],
                    "area": box_width * box_height,
                    "iscrowd": 0
                })

                annotation_id += 1

        image_id += 1

    # -----------------------------
    # COCO categories
    # -----------------------------

    categories = [
        {
            "id": class_id + 1,
            "name": class_name,
            "supercategory": "object"
        }
        for class_id, class_name in CLASSES.items()
    ]

    # -----------------------------
    # Final COCO structure
    # -----------------------------

    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    # -----------------------------
    # Save JSON
    # -----------------------------

    output_json = OUTPUT_DIR / split / "annotations.json"

    with open(output_json, "w") as file:
        json.dump(coco_data, file, indent=2)

    print(
        f"{split}: "
        f"{len(images)} images, "
        f"{len(annotations)} annotations"
    )


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":

    for split in ["train", "val", "test"]:
        convert_split(split)

    print("\nConversion complete!")