from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import os
import random

def flip_images(images_folder, ground_truth_folder, images_flip_folder, ground_truth_flip_folder):
    for folder in [images_flip_folder, ground_truth_flip_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    for filename in os.listdir(images_folder):
        if filename.endswith(".tiff"):
            base_name = os.path.splitext(filename)[0]
            tiff_file_path = os.path.join(images_folder, filename)
            image = Image.open(tiff_file_path)

            ground_truth_file = os.path.join(ground_truth_folder, base_name + '.csv')
            if os.path.exists(ground_truth_file):
                labels = pd.read_csv(ground_truth_file)

                image_horizontal = image.transpose(Image.FLIP_LEFT_RIGHT)
                horizontal_tiff_path = os.path.join(images_flip_folder, base_name + '_horizontal.tiff')
                image_horizontal.save(horizontal_tiff_path)

                horizontal_csv_path = os.path.join(ground_truth_flip_folder, base_name + '_horizontal.csv')
                labels.to_csv(horizontal_csv_path, index=False)

                image_vertical = image.transpose(Image.FLIP_TOP_BOTTOM)
                vertical_tiff_path = os.path.join(images_flip_folder, base_name + '_vertical.tiff')
                image_vertical.save(vertical_tiff_path)

                vertical_csv_path = os.path.join(ground_truth_flip_folder, base_name + '_vertical.csv')
                labels.to_csv(vertical_csv_path, index=False)

                image_diagonal = image.rotate(90)
                diagonal_tiff_path = os.path.join(images_flip_folder, base_name + '_diagonal.tiff')
                image_diagonal.save(diagonal_tiff_path)

                diagonal_csv_path = os.path.join(ground_truth_flip_folder, base_name + '_diagonal.csv')
                labels.to_csv(diagonal_csv_path, index=False)
				
def erase_cells(image, n):
    image_array = np.array(image)
    white_mask = image_array > 200  # Threshold can be modified
    white_area_coords = np.column_stack(np.where(white_mask))
    random.shuffle(white_area_coords)
    background_gray_value = int(np.mean(image_array[~white_mask]))
    for x, y in white_area_coords[:n]:
        ImageDraw.floodfill(image, (x, y), background_gray_value)
    return image