from ultralytics import YOLO
import os

model = YOLO("yolov8n.pt")

input_folder = "frames"
output_folder = "detections"

os.makedirs(output_folder, exist_ok=True)

for image in os.listdir(input_folder):

    if image.endswith(".jpg"):

        img_path = os.path.join(input_folder, image)

        results = model(img_path)

        results[0].save(
            filename=os.path.join(output_folder, image)
        )

print("Detection Completed")