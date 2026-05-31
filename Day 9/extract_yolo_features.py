from ultralytics import YOLO
import os
import pandas as pd

model = YOLO("yolov8n.pt")

rows = []

for image in os.listdir("frames"):

    if image.endswith(".jpg"):

        path = os.path.join("frames", image)

        results = model(path, verbose=False)

        for box in results[0].boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            rows.append([
                image,
                model.names[cls],
                conf
            ])

df = pd.DataFrame(
    rows,
    columns=[
        "image",
        "object",
        "confidence"
    ]
)

df.to_csv("yolo_features.csv", index=False)

print("CSV Saved")