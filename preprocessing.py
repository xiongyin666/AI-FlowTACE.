import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
def denoise_normalize(image):
    denoise_image = cv2.GaussianBlur(image, (3, 3), 0)
    normalized_image = denoise_image.astype(np.float32) / 255.0
    return normalized_image

def process_image_in_folder(input_folder, output_folder):

    for root, dirs, files in os.walk(input_folder):
        print(root)
        print(dirs)
        print(files)
        for image_file in files:
            if image_file.endswith(('.png', '.jpg', '.tif', '.jpeg')):
                image_path = os.path.join(root, image_file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                normalized_image = denoise_normalize(image)
                output_path = os.path.join(output_folder, image_file)

                cv2.imwrite(output_path, (normalized_image * 255).astype(np.uint8))

    # image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.JPG', '.tif', '.jpeg'))]
    # for image_file in image_files:
    #     image_path = os.path.join(folder_path, image_file)
    #     print(image_path)
    #     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #
    #     normalized_image = denoise_normalize(image)
        # output_path = os.path.join(folder_path, 'processed_DSA', image_file)
        # if not os.path.exists(os.path.dirname(output_path)):
        #     os.makedirs(os.path.dirname(output_path))
        # output_path = os.path.join(output_folder, image_file)
        #
        # cv2.imwrite(output_path, (normalized_image * 255).astype(np.uint8))

input_folder = r'E:\DSA_AI\seg_data\images'
output_folder = r'E:\DSA_AI\seg_data\norm_patients'
process_image_in_folder(input_folder, output_folder)