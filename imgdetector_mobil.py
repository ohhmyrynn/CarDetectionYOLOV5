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
image_path = "Screenshot 2024-05-26 141859.png"  # Replace with your image path
image = cv2.imread(image_path)
resized_image = cv2.resize(image, (640, 480)) #kalau misal ingin diubah ukkurannya

# Predict on image
detect_params = model.predict(source=[image], conf=0.3, save=False)

# Convert tensor array to numpy
DP = detect_params[0].numpy()
print(DP)

if len(DP) != 0:
    for i in range(len(detect_params[0])):
        boxes = detect_params[0].boxes
        box = boxes[i]  # returns one box
        clsID = box.cls.numpy()[0]
        conf = box.conf.numpy()[0]
        bb = box.xyxy.numpy()[0]

        cv2.rectangle(
            image,
            (int(bb[0]), int(bb[1])),
            (int(bb[2]), int(bb[3])),
            detection_colors[int(clsID)],
            2,
        )

        # Display class name and confidence
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(
            image,
            class_list[int(clsID)]
            + " "
            + str(round(conf, 3))
            + "%",
            (int(bb[0]), int(bb[1]) - 10),
            font,
            1,
            (255, 255, 255),
            2,
        )

# Display the resulting image
cv2.imshow("Deteksi Mobil", image)
cv2.waitKey(0)  # Wait for a key press to close the windows

# Close windows
cv2.destroyAllWindows()
