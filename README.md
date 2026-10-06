# PPE Object Detection: YOLO11 vs Faster R-CNN vs RetinaNet

A comparative object detection project for detecting Personal Protective
Equipment (PPE) using three different deep learning architectures:

- YOLO11n
- Faster R-CNN
- RetinaNet

The project evaluates the models not only on detection accuracy, but also
on inference latency, FPS, parameter count, and model size.

The final system is intended to serve as the computer vision backend for
a Unity-based PPE detection application.

---

## Project Overview

Personal Protective Equipment detection is an important computer vision
task for workplace safety applications.

This project investigates how different object detection architectures
perform on the same PPE dataset under the same experimental environment.

The comparison focuses on:

- Detection accuracy
- Class-level performance
- Inference speed
- Model complexity
- Model size
- Deployment suitability

The models are evaluated on the same validation dataset using COCO-style
object detection metrics.

---

## Models

### YOLO11n

A lightweight one-stage object detector optimized for real-time inference.

### Faster R-CNN

A two-stage object detector using a Region Proposal Network followed by
a region-based detection head.

### RetinaNet

A one-stage object detector using focal loss to address foreground/background
class imbalance.

---

## Dataset

The dataset contains 11 object classes:

|  ID | Class     |
| --: | --------- |
|   0 | helmet    |
|   1 | gloves    |
|   2 | vest      |
|   3 | boots     |
|   4 | goggles   |
|   5 | none      |
|   6 | person    |
|   7 | no_helmet |
|   8 | no_goggle |
|   9 | no_gloves |
|  10 | no_boots  |

Dataset split:

| Split      | Images | Annotations |
| ---------- | -----: | ----------: |
| Train      |  1,132 |       9,098 |
| Validation |    143 |       1,172 |
| Test       |    141 |       1,251 |

The original dataset uses YOLO-format annotations.

For Faster R-CNN and RetinaNet, the annotations were converted to
COCO format.

The dataset itself is not included in this repository.

---

## Experimental Setup

### Hardware

- Apple M4 Pro
- Apple Metal Performance Shaders (MPS)

### Software

- Python 3.13
- PyTorch
- TorchVision
- Ultralytics
- pycocotools

### Training

The models were trained using the same PPE dataset.

Faster R-CNN and RetinaNet used pretrained ResNet-50 FPN backbones.

The final deployment checkpoint for each model was selected using
validation performance rather than automatically selecting the final
training epoch.

---

# Results

## Detection Accuracy

Current best validation results:

| Model            | Best Checkpoint |  mAP@50:95 |     mAP@50 |     mAP@75 |
| ---------------- | --------------- | ---------: | ---------: | ---------: |
| **YOLO11n**      | `best.pt`       | **27.64%** | **54.71%** | **24.92%** |
| **Faster R-CNN** | Epoch 5         |     24.73% |     52.83% |     20.20% |
| **RetinaNet**    | Epoch 10        |     20.10% |     45.40% |     15.50% |

All metrics were calculated on the same 143-image validation set.

---

## Inference Performance

Initial controlled benchmark on the Apple M4 Pro:

| Model        | Mean Latency |     FPS | Parameters | Model Size |
| ------------ | -----------: | ------: | ---------: | ---------: |
| **YOLO11n**  |   24.13 ms\* | 41.43\* |      2.59M |    5.22 MB |
| Faster R-CNN |    115.30 ms |    8.67 |     41.35M |  158.24 MB |
| RetinaNet    |     93.69 ms |   10.67 |     32.36M |  123.93 MB |

\* YOLO11n timing is being re-checked because its latency distribution
contained significant outliers. The final benchmark will use a more
controlled timing procedure.

---

# Key Findings

### 1. YOLO11n currently provides the best overall trade-off

YOLO11n achieved the highest validation mAP@50:95 at 27.64%.

It also has substantially fewer parameters and a much smaller model size
than the two ResNet-50-based detectors.

### 2. Faster R-CNN achieved competitive accuracy

Faster R-CNN reached 24.73% mAP@50:95 and 52.83% mAP@50.

Although it was slower and considerably larger than YOLO11n, its detection
accuracy was relatively close to YOLO11n on this dataset.

### 3. RetinaNet improved substantially with additional training

RetinaNet improved from:

- Epoch 5: 7.16% mAP@50:95
- Epoch 10: 20.10% mAP@50:95

This demonstrates that training loss alone should not be used to determine
model quality.

### 4. Training loss does not directly represent validation performance

Faster R-CNN provides an example of this behavior.

Its training loss decreased from 0.5894 at epoch 5 to 0.4260 at epoch 10,
while validation mAP@50:95 decreased from 24.73% to 23.33%.

Therefore, the best validation checkpoint was selected instead of
automatically using the final training epoch.

### 5. The `no_*` classes were significantly harder

Across the models, classes such as:

- `no_helmet`
- `no_goggle`
- `no_gloves`
- `no_boots`

performed substantially worse than positive PPE classes.

These classes represent the absence of PPE and are therefore more
challenging for conventional bounding-box object detection.

---

# Per-Class Performance

YOLO11n per-class mAP@50:95:

| Class     |    AP |
| --------- | ----: |
| helmet    | 44.9% |
| gloves    | 37.0% |
| vest      | 51.3% |
| boots     | 45.1% |
| goggles   | 35.8% |
| none      | 22.7% |
| person    | 53.1% |
| no_helmet |  8.6% |
| no_goggle |  2.1% |
| no_gloves |  3.4% |
| no_boots  |  0.1% |

The results indicate that positive PPE categories are considerably easier
to detect than the absence-based categories.

---

# Architecture Comparison

The experiment highlights different trade-offs between the architectures.

| Property              | YOLO11n   | Faster R-CNN | RetinaNet    |
| --------------------- | --------- | ------------ | ------------ |
| Type                  | One-stage | Two-stage    | One-stage    |
| Accuracy              | Highest   | Competitive  | Lower        |
| Inference speed       | Fastest\* | Slowest      | Intermediate |
| Parameters            | Lowest    | Highest      | High         |
| Model size            | Smallest  | Largest      | Large        |
| Real-time suitability | High      | Lower        | Moderate     |

\* Final YOLO11n timing is pending benchmark verification.

---

# Project Structure

```text
ppe-detection/
│
├── dataset/                 # Local dataset - not committed
├── coco_dataset/            # Generated COCO dataset - not committed
│
├── retinanet/
│   ├── train_retinanet.py
│   └── evaluate_retinanet.py
│
├── train_fasterrcnn.py
├── evaluate_fasterrcnn.py
├── convert_yolo_to_coco.py
├── benchmark.py
│
├── data.yaml
├── requirements.txt
├── .gitignore
└── README.md
```
