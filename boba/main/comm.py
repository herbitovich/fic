import os
import sys
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

model = YOLO("main/yolov8_final.pt")
def predict(image_path, model=model):
    
    model.eval()
    boxes = []
    image = np.array(Image.open(image_path))
    image = Image.fromarray(image).resize((640, 640))
    image = np.array(image) / 255.0
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        detections = model(image_tensor)
        
    for box in detections[0].boxes:
        boxes.append([box.cls]+(box.xywh/640).squeeze().tolist())
    return boxes