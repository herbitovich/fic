from ultralytics import YOLO
import torch

# Step 1: Train YOLOv8 model
model = YOLO("yolov8n.pt")  # Load pre-trained YOLOv8 model

# Define the paths to your dataset
dataset_yaml = "dataset.yaml"  # You need to create this YAML file as before

# Set the number of workers for data loading
num_workers = 7  # Adjust this based on your CPU cores

# Train the model on your dataset
model.train(
    data=dataset_yaml,
    epochs=20,  # Adjust the number of epochs depending on your dataset size
    batch=64,  # Batch size
    imgsz=640,  # Image size for training
    device='cpu',  # Use CPU
    workers=num_workers  # Number of workers for data loading
)

# Save the trained model
model.save("yolov8_trained.pt")

