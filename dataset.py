import os
import random
import shutil

images_dir = './images/'
annotations_dir = './annotations/'
train_images_dir = './images/train/'
val_images_dir = './images/val/'
train_annotations_dir = './annotations/train/'
val_annotations_dir = './annotations/val/'

os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_annotations_dir, exist_ok=True)
os.makedirs(val_annotations_dir, exist_ok=True)

image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

val_images = random.sample(image_files, 11)
train_images = [f for f in image_files if f not in val_images]

for image in train_images:
    shutil.move(os.path.join(images_dir, image), os.path.join(train_images_dir, image))
    
    annotation_file = image.replace('.jpg', '.txt')
    shutil.move(os.path.join(annotations_dir, annotation_file), os.path.join(train_annotations_dir, annotation_file))

for image in val_images:
    shutil.move(os.path.join(images_dir, image), os.path.join(val_images_dir, image))
    
    annotation_file = image.replace('.jpg', '.txt')
    shutil.move(os.path.join(annotations_dir, annotation_file), os.path.join(val_annotations_dir, annotation_file))

print(f"Training images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")
