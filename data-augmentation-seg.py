import os
import cv2
import albumentations as A
import numpy as np
import argparse


# 检查边界框的有效性
def is_valid_bbox(bbox):
    # bbox: [x_center, y_center, width, height]
    x_center, y_center, width, height = bbox
    x_min = x_center - width / 2
    x_max = x_center + width / 2
    y_min = y_center - height / 2
    y_max = y_center + height / 2
    return x_min < x_max and y_min < y_max


# 读取YOLO格式的标签文件
def load_yolo_labels(label_file):
    with open(label_file, 'r') as file:
        lines = file.readlines()

    bboxes = []
    class_labels = []

    for line in lines:
        parts = line.split()
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:])
        bboxes.append([x_center, y_center, width, height])
        class_labels.append(class_id)

    return bboxes, class_labels


# 增强和保存图像与标签的函数
def get_enhance_save(
        old_images_files,
        old_labels_files,
        enhance_images_files,
        enhance_labels_files,
        mid_name,
):
    # 设置数据增强方法（针对YOLO格式的增强）
    transform = A.Compose(
        [
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.5,
                contrast_limit=0.5,
                p=0.5),
            A.ColorJitter(p=0.9),
            A.RandomGamma(p=0.2),
            A.ChannelShuffle(p=0.5),
            A.ToGray(p=0.2),
            A.RandomCrop(height=256, width=256, p=0.5),
            A.Resize(height=512, width=512, p=1.0),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            min_area=5,
            min_visibility=0.2,
            label_fields=["class_labels"],
        )
    )

    label_files_name = os.listdir(old_labels_files)

    for name in label_files_name:
        label_file = os.path.join(old_labels_files, name)

        # 读取YOLO格式的标签文件
        bboxes, class_labels = load_yolo_labels(label_file)

        # 读取图像
        image_path = os.path.join(old_images_files, name.replace(".txt", ".jpg"))
        if not os.path.exists(image_path):
            image_path = image_path.replace(".jpg", ".png")

        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Image file {name} could not be read.")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 应用数据增强
        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        transformed_image = transformed["image"]
        transformed_bboxes = transformed["bboxes"]
        transformed_class_labels = transformed["class_labels"]

        # 过滤无效边界框
        valid_bboxes = []
        valid_class_labels = []
        for bbox, label in zip(transformed_bboxes, transformed_class_labels):
            if is_valid_bbox(bbox):
                valid_bboxes.append(bbox)
                valid_class_labels.append(label)

        if not valid_bboxes:
            print(f"Warning: No valid bounding boxes for {name}. Skipping.")
            continue

        # 保存增强后的图像和标签
        if not os.path.exists(enhance_images_files):
            os.mkdir(enhance_images_files)

        a, b = os.path.splitext(name)
        new_name = a + mid_name + b
        cv2.imwrite(
            os.path.join(enhance_images_files, new_name.replace(".txt", ".png")),
            transformed_image,
        )

        if not os.path.exists(enhance_labels_files):
            os.mkdir(enhance_labels_files)

        new_txt_file = open(os.path.join(enhance_labels_files, new_name), "w")

        # 写入新的YOLO格式标签
        for box, label in zip(valid_bboxes, valid_class_labels):
            new_class_num = label
            box = list(map(lambda x: "%.5f" % x, box))
            box.insert(0, str(new_class_num))
            new_txt_file.write(" ".join(box) + "\n")

        new_txt_file.close()


# 主函数
def main(args):
    root = args.root

    old_images_files = os.path.join(root, "images")
    old_labels_files = os.path.join(root, "labels")

    enhance_images_files = os.path.join(root, args.enhance_images_folder)
    enhance_labels_files = os.path.join(root, args.enhance_labels_folder)

    # 实现对传入的数据文件进行遍历读取，并进行数据增强
    try:
        get_enhance_save(
            old_images_files,
            old_labels_files,
            enhance_images_files,
            enhance_labels_files,
            args.mid_name,
        )
    except Exception as e:
        print(f"Error occurred while processing images and labels: {e}")
        raise  # 重新抛出异常，方便调试


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply augmentation to YOLO annotated data")
    parser.add_argument("--root", type=str, default=r"C:\Users\Administrator\Desktop\new_seg",
                        help="Root directory of the dataset")
    parser.add_argument("--enhance_images_folder", type=str, default="enhance_images",
                        help="Folder to store enhanced images")
    parser.add_argument("--enhance_labels_folder", type=str, default="enhance_labels",
                        help="Folder to store enhanced labels")
    parser.add_argument("--mid_name", type=str, default="_aug", help="Suffix to be added to enhanced files")
    args = parser.parse_args()
    main(args)
