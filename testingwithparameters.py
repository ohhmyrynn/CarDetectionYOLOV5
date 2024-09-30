import cv2
from ultralytics import YOLO
import random
import numpy as np

# Load class list
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# Load a pretrained YOLOv8n model
model = YOLO("best.pt", "v8")

# Load the image
image_path = "test1.png"  # Replace with your image path and ground truth path
image = cv2.imread(image_path)
resized_image = cv2.resize(image, (640, 480))  # kalau misal ingin diubah ukkurannya

# Load ground truth (sesuaikan dengan format anotasi ground truth Anda)
ground_truth = ...  # Load ground truth data

# Predict on image
detect_params = model.predict(source=[image], conf=0.3, save=False)

# Convert tensor array to numpy
DP = detect_params[0].numpy()

# Inisialisasi variabel untuk TP, FP, FN
TP = 0
FP = 0
FN = 0

if len(DP) != 0:
    for i in range(len(detect_params[0])):
        boxes = detect_params[0].boxes
        box = boxes[i]  # returns one box
        clsID = box.cls.numpy()[0]
        conf = box.conf.numpy()[0]
        bb = box.xyxy.numpy()[0]

        # Pencocokan prediksi dengan ground truth (sesuaikan dengan format ground truth Anda)
        matched = False
        for gt in ground_truth:
            # Implementasi logika pencocokan bounding box prediksi dengan ground truth
            if (iou(bb, gt.bbox) > 0.5):  # Ganti threshold IoU dengan nilai yang sesuai
                matched = True
                break

        if matched:
            TP += 1  # True Positive
        else:
            FP += 1  # False Positive

        cv2.rectangle(
            image,
            (int(bb[0]), int(bb[1])),
            (int(bb[2]), int(bb[3])),
            detection_colors[int(clsID)],
            2,
        )

        # Display class name and
