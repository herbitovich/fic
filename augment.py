import os
import cv2
import albumentations as A
import numpy as np

input_dir = "datasets/images/train/"
output_dir = input_dir  # Directory to save augmented images
annotations_dir = "datasets/labels/train/"

os.makedirs(output_dir, exist_ok=True)

augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.Blur(blur_limit=3, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

for filename in os.listdir(input_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg', 'JPG')):
        img_path = os.path.join(input_dir, filename)
        image = cv2.imread(img_path)

        annotation_file = os.path.join(annotations_dir, f"{os.path.splitext(filename)[0]}.txt")
        bboxes = []
        class_labels = []
        length = 0
        if os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f:
                for line in f.readlines():
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(int(class_id))
                    length += 1

        for _ in range(11):
            augmented = augmentation(image=image, bboxes=bboxes, class_labels=class_labels)
            augmented_image = augmented['image']
            augmented_bboxes = augmented['bboxes']
            augmented_filename = f"{os.path.splitext(filename)[0]}_aug_{_}.jpg"
            augmented_path = os.path.join(output_dir, augmented_filename)
            cv2.imwrite(augmented_path, img=augmented_image)

            # Save in YOLO format
            annotation_filename = f"{os.path.splitext(filename)[0]}_aug_{_}.txt"
            annotation_path = os.path.join(annotations_dir, annotation_filename)

            with open(annotation_path, 'w') as ann_file:
                if length < len(augmented_bboxes): 
                    start = len(augmented_bboxes) - length
                else: start = 0
                for i, bbox in enumerate(augmented_bboxes[start:]):
                    if i > length: break
                    class_id = class_labels[i]
                    ann_file.write(f"{class_id} {' '.join(map(str, bbox))}\n")

print("Augmentation complete! Augmented images and annotations saved to:", output_dir)
