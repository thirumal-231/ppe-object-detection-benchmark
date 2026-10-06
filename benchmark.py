import os
import time
import statistics

import torch
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
    retinanet_resnet50_fpn,
    RetinaNet_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetHead

from ultralytics import YOLO
from PIL import Image
from pycocotools.coco import COCO


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

VAL_IMAGES = "coco_dataset/val/images"
ANNOTATIONS = "coco_dataset/val/annotations.json"

FASTER_CHECKPOINT = "fasterrcnn_epoch5.pth"
RETINA_CHECKPOINT = "retinanet/retinanet_epoch10.pth"
YOLO_CHECKPOINT = "runs/detect/train/weights/best.pt"

NUM_CLASSES = 12       # Faster R-CNN: background + 11 classes
RETINA_CLASSES = 11   # RetinaNet: 11 foreground classes

WARMUP = 10
ITERATIONS = 50

IMAGE_SIZE = 640


# ============================================================
# HELPERS
# ============================================================

def synchronize():
    if DEVICE.type == "mps":
        torch.mps.synchronize()


def load_validation_images():
    coco = COCO(ANNOTATIONS)

    image_ids = coco.getImgIds()
    images = coco.loadImgs(image_ids)

    paths = []

    for img in images:
        path = os.path.join(VAL_IMAGES, img["file_name"])

        if os.path.exists(path):
            paths.append(path)

    return paths


def load_image(path):
    image = Image.open(path).convert("RGB")

    # Resize everything to 640x640 for a controlled benchmark
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    tensor = torch.from_numpy(
        __import__("numpy").array(image)
    ).permute(2, 0, 1).float() / 255.0

    return tensor.to(DEVICE)


def get_model_size(path):
    size_bytes = os.path.getsize(path)
    return size_bytes / (1024 * 1024)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# ============================================================
# FASTER R-CNN
# ============================================================

def load_fasterrcnn():

    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        NUM_CLASSES
    )

    checkpoint = torch.load(
        FASTER_CHECKPOINT,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)

    model.roi_heads.score_thresh = 0.001
    model.roi_heads.nms_thresh = 0.5
    model.roi_heads.detections_per_img = 100

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# RETINANET
# ============================================================

def load_retinanet():

    model = retinanet_resnet50_fpn(
        weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT
    )

    num_anchors = model.head.classification_head.num_anchors

    model.head = RetinaNetHead(
        model.backbone.out_channels,
        num_anchors,
        RETINA_CLASSES
    )

    checkpoint = torch.load(
        RETINA_CHECKPOINT,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)

    model.score_thresh = 0.001
    model.nms_thresh = 0.5
    model.detections_per_img = 100
    model.topk_candidates = 1000

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# BENCHMARK TORCHVISION MODEL
# ============================================================

@torch.inference_mode()
def benchmark_torchvision(model, images):

    # ----------------------------
    # Warmup
    # ----------------------------

    print(f"Warmup: {WARMUP} iterations")

    dummy = images[0]

    for _ in range(WARMUP):
        model([dummy])

    synchronize()

    # ----------------------------
    # Benchmark
    # ----------------------------

    latencies = []

    print(f"Benchmarking: {ITERATIONS} iterations")

    for i in range(ITERATIONS):

        image = images[i % len(images)]

        synchronize()

        start = time.perf_counter()

        model([image])

        synchronize()

        end = time.perf_counter()

        latency_ms = (end - start) * 1000

        latencies.append(latency_ms)

    return latencies


# ============================================================
# YOLO
# ============================================================

def load_yolo():

    model = YOLO(YOLO_CHECKPOINT)

    return model


def benchmark_yolo(model, image_paths):

    # ----------------------------
    # Warmup
    # ----------------------------

    print(f"Warmup: {WARMUP} iterations")

    for i in range(WARMUP):

        model.predict(
            image_paths[i % len(image_paths)],
            imgsz=IMAGE_SIZE,
            conf=0.001,
            iou=0.5,
            max_det=100,
            device="mps",
            verbose=False
        )

    synchronize()

    # ----------------------------
    # Benchmark
    # ----------------------------

    latencies = []

    print(f"Benchmarking: {ITERATIONS} iterations")

    for i in range(ITERATIONS):

        image_path = image_paths[i % len(image_paths)]

        synchronize()

        start = time.perf_counter()

        model.predict(
            image_path,
            imgsz=IMAGE_SIZE,
            conf=0.001,
            iou=0.5,
            max_det=100,
            device="mps",
            verbose=False
        )

        synchronize()

        end = time.perf_counter()

        latency_ms = (end - start) * 1000

        latencies.append(latency_ms)

    return latencies


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(name, latencies, parameters, size_mb):

    mean_latency = statistics.mean(latencies)
    median_latency = statistics.median(latencies)

    sorted_latencies = sorted(latencies)

    p95_index = int(len(sorted_latencies) * 0.95) - 1
    p95 = sorted_latencies[p95_index]

    fps = 1000 / mean_latency

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    print(f"Mean latency   : {mean_latency:.2f} ms")
    print(f"Median latency : {median_latency:.2f} ms")
    print(f"P95 latency    : {p95:.2f} ms")
    print(f"FPS            : {fps:.2f}")
    print(f"Parameters     : {parameters:,}")

    if size_mb is not None:
        print(f"Model size     : {size_mb:.2f} MB")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("PPE OBJECT DETECTION SPEED BENCHMARK")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Warmup: {WARMUP}")
    print(f"Iterations: {ITERATIONS}")

    image_paths = load_validation_images()

    print(f"Validation images available: {len(image_paths)}")

    # Load images once for TorchVision models
    print("\nLoading validation images...")

    tensors = [
        load_image(path)
        for path in image_paths
    ]

    print("Images loaded.")

    # ========================================================
    # Faster R-CNN
    # ========================================================

    print("\nLoading Faster R-CNN...")

    faster = load_fasterrcnn()

    faster_latencies = benchmark_torchvision(
        faster,
        tensors
    )

    faster_params = count_parameters(faster)
    faster_size = get_model_size(FASTER_CHECKPOINT)

    print_results(
        "Faster R-CNN",
        faster_latencies,
        faster_params,
        faster_size
    )

    del faster

    if DEVICE.type == "mps":
        torch.mps.empty_cache()

    # ========================================================
    # RetinaNet
    # ========================================================

    print("\nLoading RetinaNet...")

    retina = load_retinanet()

    retina_latencies = benchmark_torchvision(
        retina,
        tensors
    )

    retina_params = count_parameters(retina)
    retina_size = get_model_size(RETINA_CHECKPOINT)

    print_results(
        "RetinaNet",
        retina_latencies,
        retina_params,
        retina_size
    )

    del retina

    if DEVICE.type == "mps":
        torch.mps.empty_cache()

    # ========================================================
    # YOLO
    # ========================================================

    print("\nLoading YOLO11n...")

    yolo = load_yolo()

    yolo_latencies = benchmark_yolo(
        yolo,
        image_paths
    )

    yolo_params = sum(
        p.numel()
        for p in yolo.model.parameters()
    )

    yolo_size = get_model_size(YOLO_CHECKPOINT)

    print_results(
        "YOLO11n",
        yolo_latencies,
        yolo_params,
        yolo_size
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)

    print("\nUse these results together with your validation mAP:")
    print()
    print("YOLO11n       → mAP50:95 = 0.2764")
    print("Faster R-CNN  → mAP50:95 = 0.2473")
    print("RetinaNet     → mAP50:95 = 0.2010")


if __name__ == "__main__":
    main()