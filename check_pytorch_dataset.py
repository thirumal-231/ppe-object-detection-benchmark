import torch
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor


DATASET_ROOT = "coco_dataset/train"
ANNOTATION_FILE = f"{DATASET_ROOT}/annotations.json"


dataset = CocoDetection(
    root=f"{DATASET_ROOT}/images",
    annFile=ANNOTATION_FILE,
    transform=ToTensor()
)


print("Number of images:", len(dataset))


image, annotations = dataset[0]


print("Image shape:", image.shape)
print("Number of annotations:", len(annotations))


if annotations:
    print("\nFirst annotation:")
    print(annotations[0])