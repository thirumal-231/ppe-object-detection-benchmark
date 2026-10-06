from ultralytics import YOLO
import time


MODEL_PATH = "runs/detect/train/weights/best.pt"
DATA_YAML = "data.yaml"


# -----------------------------
# Load model
# -----------------------------

model = YOLO(MODEL_PATH)

print("Model loaded:", MODEL_PATH)


# -----------------------------
# COCO-style validation metrics
# -----------------------------

print("\nRunning validation...\n")

metrics = model.val(
    data=DATA_YAML,
    split="val",
    imgsz=640,
    batch=2,
    device="mps",
    plots=True
)


# -----------------------------
# Overall metrics
# -----------------------------

print("\n" + "=" * 50)
print("YOLO11 VALIDATION RESULTS")
print("=" * 50)

print(
    f"mAP @ 0.50:0.95 : "
    f"{metrics.box.map:.4f}"
)

print(
    f"mAP @ 0.50      : "
    f"{metrics.box.map50:.4f}"
)

print(
    f"mAP @ 0.75      : "
    f"{metrics.box.map75:.4f}"
)


# -----------------------------
# Per-class AP
# -----------------------------

print("\nPer-Class AP")
print("-" * 50)

class_names = model.names

for class_id, class_name in class_names.items():

    ap50_95 = metrics.box.ap[class_id]

    print(
        f"{class_name:<12} "
        f"AP@50:95 = {ap50_95:.3f}"
    )


# -----------------------------
# Speed
# -----------------------------

print("\nSpeed")
print("-" * 50)

print(
    f"Preprocess : "
    f"{metrics.speed['preprocess']:.2f} ms/image"
)

print(
    f"Inference  : "
    f"{metrics.speed['inference']:.2f} ms/image"
)

print(
    f"Postprocess: "
    f"{metrics.speed['postprocess']:.2f} ms/image"
)

total_ms = (
    metrics.speed["preprocess"]
    + metrics.speed["inference"]
    + metrics.speed["postprocess"]
)

print(
    f"Total      : "
    f"{total_ms:.2f} ms/image"
)

print(
    f"Approx FPS : "
    f"{1000 / total_ms:.2f}"
)