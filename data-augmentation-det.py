import argparse
import albumentations as A
import cv2
import os

def get_enhance_save(
    old_images_files,
    old_labels_files,
    label_list,
    enhance_images_files,
    enhance_labels_files,
    mid_name,
):
    # 这里设置指定的数据增强方法
    # p 参数决定了该增强方法被应用的概率。
    # 例如，如果你的 Compose 中包含了5个增强方法，每个方法的概率都设置为0.2，
    # 那么在每次应用增强时，有大约20%的概率会选择其中一个增强方法。

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
            A.BBoxSafeRandomCrop(),
            # A.OneOf(
            #     [
            #         A.Blur(blur_limit=3, p=0.9),
            #         A.MedianBlur(blur_limit=3, p=0.9),
            #         A.GaussianBlur(blur_limit=3, p=0.9),
            #     ],
            #     p=0.9,
            # ),
            # A.OneOf(
            #     [
            #         A.CLAHE(clip_limit=2),
            #         A.HueSaturationValue(
            #             hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20
            #         ),
            #     ],
            #     p=0.9,
            # ),
            # A.OneOf(
            #     [
            #         A.OpticalDistortion(p=0.9),
            #         A.GridDistortion(p=0.9),
            #         A.ElasticTransform(p=0.9),
            #     ],
            #     p=0.2,
            # ),
            # A.ShiftScaleRotate(
            #     shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5
            # ),
            A.RandomGamma(p=0.2),
            A.ChannelShuffle(p=0.5),
            A.ToGray(p=0.2),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            min_area=5,
            min_visibility=0.2,
            label_fields=["class_labels"],
        ),
    )

    label_files_name = os.listdir(old_labels_files)

    for name in label_files_name:
        label_files = os.path.join(old_labels_files, name)

        yolo_b_boxes = open(label_files).read().splitlines()

        bboxes = []

        class_labels = []

        # 对一个txt文件的每一行标注数据进行处理
        for b_box in yolo_b_boxes:
            b_box = b_box.split(" ")
            m_box = b_box[1:5]

            m_box = list(map(float, m_box))

            m_class = b_box[0]

            bboxes.append(m_box)
            class_labels.append(label_list[int(m_class)])

        # 读取对应的图像
        image_path = os.path.join(old_images_files, name.replace(".txt", ".jpg"))
        if os.path.exists(image_path) is False:
            image_path = os.path.join(old_images_files, name.replace(".txt", ".png"))

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 调用上面定义的图像增强方法进行数据增强
        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        transformed_image = transformed["image"]
        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
        transformed_b_boxes = transformed["bboxes"]
        transformed_class_labels = transformed["class_labels"]

        # transformed_b_boxes = [clip_bbox(box) for box in transformed_b_boxes]
        # 先判断目标文件夹路径是否存在
        if os.path.exists(enhance_images_files) is False:
            os.mkdir(enhance_images_files)
        a, b = os.path.splitext(name)
        new_name = a + mid_name + b
        cv2.imwrite(
            os.path.join(enhance_images_files, new_name.replace(".txt", ".png")),
            transformed_image,
        )

        if os.path.exists(enhance_labels_files) is False:
            os.mkdir(enhance_labels_files)

        new_txt_file = open(os.path.join(enhance_labels_files, new_name), "w")

        new_bboxes = []

        for box, label in zip(transformed_b_boxes, transformed_class_labels):
            new_class_num = label_list.index(label)
            box = list(box)
            for i in range(len(box)):
                box[i] = str(("%.5f" % box[i]))
            box.insert(0, str(new_class_num))
            new_bboxes.append(box)

        for new_box in new_bboxes:
            for ele in new_box:
                if ele is not new_box[-1]:
                    new_txt_file.write(ele + " ")
                else:
                    new_txt_file.write(ele)

            new_txt_file.write("\n")

        new_txt_file.close()


def main(args):
    root = args.root

    old_images_files = os.path.join(root, "images")
    old_labels_files = os.path.join(root, "labels")

    enhance_images_files = os.path.join(root, args.enhance_images_folder)
    enhance_labels_files = os.path.join(root, args.enhance_labels_folder)

    # 这里设置数据集的类别
    label_list = [

        "target"
    ]
    # 实现对传入的数据文件进行遍历读取，并进行数据增强
    try:
        get_enhance_save(
            old_images_files,
            old_labels_files,
            label_list,
            enhance_images_files,
            enhance_labels_files,
            args.mid_name,
        )
    except Exception as e:
        print(f"Error: {e}")
#
#
#
#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply augmentation to YOLO annotated data by csdn迪菲赫尔曼")
    parser.add_argument("--root", type=str, default=r"C:\Users\Administrator\Desktop\new_seg", help="Root directory of the dataset")
    parser.add_argument("--enhance_images_folder", type=str, default="enhance_images", help="Folder to store enhanced images")
    parser.add_argument("--enhance_labels_folder", type=str, default="enhance_labels", help="Folder to store enhanced labels")
    parser.add_argument("--mid_name", type=str, default="_aug", help="Suffix to be added to enhanced files")
    parser.add_argument("--num_workers", type=int, default=4, help="并行进程数量")
    args = parser.parse_args()
    main(args)

