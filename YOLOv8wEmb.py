import os
import torch
import cv2
from skimage.feature import hog
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class PTL_Dataset(Dataset):
    def __init__(self, image_paths, label_paths, target_size=(640, 640)):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.target_size = target_size
    
    def __getitem__(self, idx):
        # Load image
        image = np.array(Image.open(self.image_paths[idx]))
        
        # Read annotation (bounding boxes and class ids)
        label_file = self.label_paths[idx]
        with open(label_file, 'r') as f:
            bboxes = []
            for line in f:
                components = line.strip().split()
                if len(components) == 5:
                    class_id = int(components[0])
                    x_center = float(components[1])
                    y_center = float(components[2])
                    width = float(components[3])
                    height = float(components[4])
                    bboxes.append([class_id, x_center, y_center, width, height])
        
        # Resize image
        image = Image.fromarray(image)
        image = image.resize(self.target_size)  # Resize to (640, 640)
        image = np.array(image) / 255.0  # Normalize to [0, 1]
        
        # Convert to tensor (CHW format)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        
        return image, bboxes
    
    def __len__(self):
        return len(self.image_paths)


from torch.utils.data import DataLoader
import os

# Get paths to images and labels
image_paths = [os.path.join('./datasets/images/train', fname) for fname in os.listdir('./datasets/images/train')]
label_paths = [os.path.join('./datasets/labels/train', fname.replace('.jpg', '.txt')) for fname in os.listdir('./datasets/images/train')]

# Create dataset and data loader
dataset = PTL_Dataset(image_paths, label_paths)
train_loader = DataLoader(dataset)#, shuffle=True)
print("Loaded the data.")

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

class YOLOWithMetricLearning(nn.Module):
    def __init__(self, base_model, embedding_dim=128, embedding_in = 2500):
        super(YOLOWithMetricLearning, self).__init__()
        self.base_model = base_model
        self.embedding_dim = embedding_dim
        self.embedding_in = embedding_in
        # Create embedding head
        self.embedding_head = nn.Sequential(
            nn.Linear(embedding_in, 512),  # Adjust according to YOLO's output shape
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        detections = self.base_model(x)  # Get detections from YOLO
        
        # Extract feature embeddings from detected apples
        embeddings = []
        for detection in detections:
            for box in detection.boxes:
                try:
                    x_center, y_center, width, height = box.xywhn.unsqueeze(0)
                except:
                    print(box.xywhn)
                    exit()
                conf = box.conf
                cls = box.cls

                x_min = int(x_center - (width / 2))
                y_min = int(y_center - (height / 2))
                x_max = int(x_center + (width / 2))
                y_max = int(y_center + (height / 2))

                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(x.shape[1], x_max)
                y_max = min(x.shape[0], y_max)

                cropped_image = x[y_min:y_max, x_min:x_max]

                gray_cropped = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

                hog_features = hog(gray_cropped, 
                                orientations=9, 
                                pixels_per_cell=(8, 8), 
                                cells_per_block=(2, 2), 
                                visualize=False,  # Set to False to only get features
                                multichannel=False)
                if len(hog_features) < self.embedding_in:
                    # Pad with zeros if the feature vector is too short
                    hog_features = np.pad(hog_features, (0, self.embedding_in - len(hog_features)), 'constant')
                elif len(hog_features) > self.embedding_in:
                    # Truncate if the feature vector is too long
                    hog_features = hog_features[:self.embedding_in]
                embedding = self.embedding_head(hog_features)    
                embeddings.append(embedding)
        
        return detections, embeddings

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Calculate distances between anchor-positive and anchor-negative
        pos_distance = F.pairwise_distance(anchor, positive, p=2)
        neg_distance = F.pairwise_distance(anchor, negative, p=2)
        
        # Triplet loss: max(0, distance(anchor, positive) - distance(anchor, negative) + margin)
        loss = F.relu(pos_distance - neg_distance + self.margin)
        return loss.mean()

import random

def select_triplets(embeddings, labels):
    triplets = []
    if len(embeddings) < 3: return []
    # Create a mapping from class labels to the indices of the embeddings that belong to each class
    class_to_embeddings = {}
    for label, *i in enumerate(labels):
        if label not in class_to_embeddings:
            class_to_embeddings[label] = []
        class_to_embeddings[label].append(i)
    
    # Now, let's generate the triplets
    for i in range(len(embeddings)):
        anchor_embedding = embeddings[i]
        anchor_label = labels[i]
        
        # Find a positive (same class as anchor)
        positive_idx = random.choice([j for j in class_to_embeddings[anchor_label] if j != i])
        positive_embedding = embeddings[positive_idx]
        
        # Find a negative (different class from anchor)
        negative_label = random.choice([label for label in class_to_embeddings if label != anchor_label])
        negative_idx = random.choice(class_to_embeddings[negative_label])
        negative_embedding = embeddings[negative_idx]
        
        # Add the triplet (anchor, positive, negative)
        triplets.append((anchor_embedding, positive_embedding, negative_embedding))
    print("TRIPLETS:", triplets)
    return triplets

from torch.optim import Adam

# Load the YOLO model (pre-trained)
model = YOLO('yolov8n.pt')  # Use the YOLOv8n model (smaller and faster)
print("Initialized the model.")

model = YOLOWithMetricLearning(base_model=model, embedding_dim=128)

model_with_metric_learning = nn.DataParallel(model)
optimizer = Adam(model_with_metric_learning.parameters(), lr=0.0001)
triplet_loss = TripletLoss()

print("Starting the training process.")
for epoch in range(20):  # Number of epochs
    for images, labels in train_loader:
        optimizer.zero_grad()

        # Forward pass through the model
        detections, embeddings = model_with_metric_learning(images)
        

        triplets = select_triplets(embeddings, labels)
        if not triplets:
            print(f"No valid triplets found for epoch {epoch+1}. Skipping this iteration.")
            continue
        
        # Unpack the triplets
        anchor, positive, negative = zip(*triplets)  # Unzipping the triplets
        
        # Calculate triplet loss
        loss = triplet_loss(anchor, positive, negative)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
torch.save(model_with_metric_learning.state_dict(), 'yolo_apple_model.pth')

"""
# Inference on a new image
model_with_metric_learning.eval()  # Switch to evaluation mode
image_path = './test_images/apple_test_image.jpg'
image = np.array(Image.open(image_path))
image = Image.fromarray(image).resize((640, 640))  # Resize to match training size
image = np.array(image) / 255.0
image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # Add batch dimension

# Get predictions
with torch.no_grad():
    detections, embeddings = model_with_metric_learning(image_tensor)
    
# Visualize results
results = detections[0]  # First image in batch (we have only one image)
results.show()  # Visualize detections (bounding boxes and classes)


# Save the model
"""

