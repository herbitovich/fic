import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path
import os
import matplotlib.cm as cm
import argparse
from .comm import predict


    
def is_intersecting_with_corner(bbox, corner, img_width, img_height):
    x_min, y_min, width, height = bbox
    if corner == 'top-left':
        return x_min < img_width * 0.25 and y_min < img_height * 0.25
    elif corner == 'top-right':
        return x_min + width > img_width * 0.75 and y_min < img_height * 0.25
    elif corner == 'bottom-left':
        return x_min < img_width * 0.25 and y_min + height > img_height * 0.75
    elif corner == 'bottom-right':
        return x_min + width > img_width * 0.75 and y_min + height > img_height * 0.75
    return False

def run(image_path, bboxes, output_dir, save_image_path,show_plot = False):
    dict_of_classes = {0: "СвободностоящаяТипаРюмка",
  1: "Одноцепная Башенного Типа",
  2: "Портальная На Оттяжках",
  3: "Двуцепная Башенного Типа",
  4: "Одноцепная На Оттяжках",
  5: "Типа Набла",
  6: "Многогранная",
  7: "Необычная"
        
    }
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    save_image_path = Path(save_image_path)
    img = Image.open(image_path)
    p = Path(image_path)
    
    img_width, img_height = img.size

    for bbox in bboxes:
        cls, _, _, _, _ = bbox
        cls = int(cls) 
        class_dir = Path(output_dir) / f'{dict_of_classes[cls].replace(' ','')}'
        if not class_dir.exists():
            os.makedirs(class_dir)

    fig, ax = plt.subplots(1, figsize=(12, 8))
    fig.patch.set_facecolor('#101010')
    ax.tick_params(axis='x', labelcolor = 'white')
    ax.tick_params(axis='y', labelcolor = 'white')
    ax.imshow(img)

    color_map = cm.get_cmap('tab20')
    num_classes = len(set(int(b[0]) for b in bboxes)) 
    colors = [color_map(i % 20) for i in range(num_classes)]

    bbox_patches = []
    labels = []
    occupied_corners = {corner: False for corner in ['top-left', 'top-right', 'bottom-left', 'bottom-right']}

    for idx, bbox in enumerate(bboxes):
        cls, x_center_norm, y_center_norm, width_norm, height_norm = bbox
        cls = int(cls)
        x_center_norm, y_center_norm = float(x_center_norm), float(y_center_norm)
        width_norm, height_norm = float(width_norm), float(height_norm)

        x_center = x_center_norm * img_width
        y_center = y_center_norm * img_height
        width = width_norm * img_width
        height = height_norm * img_height

        x_min = x_center - width / 2
        y_min = y_center - height / 2


        for corner in ['top-left', 'top-right', 'bottom-left', 'bottom-right']:
            if is_intersecting_with_corner((x_min, y_min, width, height), corner, img_width, img_height):
                occupied_corners[corner] = True

        color = colors[cls % num_classes]

        bbox_patch = patches.Rectangle(
            (x_min, y_min), width, height, linewidth=2, edgecolor=color, facecolor='none')
        
        ax.add_patch(bbox_patch)

        if f'{dict_of_classes[cls]}' not in labels:
            labels.append(f'{dict_of_classes[cls]}')
            bbox_patches.append(patches.Patch(color=color, label=f'{dict_of_classes[cls]}'))

        cropped_img = img.crop((x_min, y_min, x_min + width, y_min + height))
        cropped_img_name = f'{p.stem}_bbox_{idx}.png'
        cropped_img_path = Path(output_dir) / f'{dict_of_classes[cls].replace(' ','')}' / cropped_img_name
        cropped_img.save(cropped_img_path)

    legend_loc = 'upper left' 

    ax.legend(handles=bbox_patches, loc=legend_loc, fontsize=12,frameon=True, facecolor="#101010", labelcolor="white")

    if save_image_path:
        save_dir = Path(save_image_path).parent
        if not save_dir.exists():
            print(f"Directory {save_dir} does not exist, creating it.")
            os.makedirs(save_dir)
        
        plt.savefig(save_image_path, bbox_inches='tight') 

    if show_plot:
        plt.show()

    plt.close(fig)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process images and their annotations.')
    parser.add_argument('--image-path', type=str, help='Directory containing image')
    parser.add_argument('--output-dir', type=str, default='./output', help='Directory to save output')
    parser.add_argument('--show', type=bool, default=False, help='Flag to show images with bounding boxes')
    parser.add_argument('--save-image', type=str, default='./', help='Path to save the image with bounding boxes')
    
    args = parser.parse_args()
    if not args.image_path:
        print("Error: --image-path argument is required.")
        exit(1)

    return args

if __name__ == '__main__':

    args = parse_arguments()
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)
    output_dir = str(output_dir)
    image_path = Path(args.image_path)
    
    list_of_annots_from_model = predict(image_path)

    if not image_path.exists():
        print(f"{image_path} file does not exist.")
        exit(1)
    image_path = str(image_path)
    save_image_path = args.save_image
    show_flag = args.show
    run(image_path, list_of_annots_from_model,output_dir, save_image_path, show_flag)
