import os
import random
import shutil

# 定义路径
# images_dir = r'E:\DSA_AI\norm_patients\images'  # 图像文件夹
# labels_dir = r'E:\DSA_AI\norm_patients\labels'  # 标签文件夹
# output_dir = r'E:\DSA_AI\norm_patients'  # 划分后的数据保存目录
#
# # 创建输出文件夹
# train_dir = os.path.join(output_dir, 'train')
# val_dir = os.path.join(output_dir, 'val')
# test_dir = os.path.join(output_dir, 'test')
#
# # 在每个数据集目录下创建 'images' 和 'labels' 子文件夹
# os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
# os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
#
# os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
# os.makedirs(os.path.join(val_dir, 'labels'), exist_ok=True)
#
# os.makedirs(os.path.join(test_dir, 'images'), exist_ok=True)
# os.makedirs(os.path.join(test_dir, 'labels'), exist_ok=True)
#
# # 获取所有图片文件和标签文件
# image_files = [f.lower for f in os.listdir(images_dir) if f.endswith(('.JPG', '.png'))]
# label_files = [f.lower for f in os.listdir(labels_dir) if f.endswith('.txt')]
#
# # 确保每个图像文件都有对应的标签文件
# # image_set = set(image_files)
# # label_set = set(label_files)
# # print("Image files:", image_set)
# # print("Label files:", label_set)
# # assert image_set == label_set, "每个图片文件都必须有一个对应的标签文件"
#
# # 将图像和标签的配对信息存储在一个列表中
# data_pairs = [(img, img.replace('.JPG', '.txt').replace('.png', '.txt')) for img in image_files]
#
# # 随机打乱数据
# random.shuffle(data_pairs)
#
# # 划分数据集的比例
# train_size = int(len(data_pairs) * 0.8)
# val_size = int(len(data_pairs) * 0.1)
# test_size = len(data_pairs) - train_size - val_size
#
# # 划分数据集
# train_pairs = data_pairs[:train_size]
# val_pairs = data_pairs[train_size:train_size + val_size]
# test_pairs = data_pairs[train_size + val_size:]
#
#
# # 定义一个函数来复制文件
# def copy_files(pairs, target_images_dir, target_labels_dir):
#     for img_file, lbl_file in pairs:
#         img_src = os.path.join(images_dir, img_file)
#         lbl_src = os.path.join(labels_dir, lbl_file)
#
#         # 复制图像和标签文件到目标文件夹
#         shutil.copy(img_src, os.path.join(target_images_dir, img_file))
#         shutil.copy(lbl_src, os.path.join(target_labels_dir, lbl_file))
#
#
# # 复制文件到相应的文件夹
# copy_files(train_pairs, os.path.join(train_dir, 'images'), os.path.join(train_dir, 'labels'))
# copy_files(val_pairs, os.path.join(val_dir, 'images'), os.path.join(val_dir, 'labels'))
# copy_files(test_pairs, os.path.join(test_dir, 'images'), os.path.join(test_dir, 'labels'))
#
# print(f"数据集已成功划分为：\n训练集: {len(train_pairs)}\n验证集: {len(val_pairs)}\n测试集: {len(test_pairs)}")


# 设置图像和标签文件夹路径
image_folder = r'E:\DSA_AI\norm_patients\images'  # 图像文件夹路径
label_folder = r'E:\DSA_AI\norm_patients\labels'  # 标签文件夹路径

# 设置输出文件夹
output_train = r'E:\DSA_AI\norm_patients\train'  # 训练集文件夹
output_val = r'E:\DSA_AI\norm_patients\val'  # 验证集文件夹
output_test = r'E:\DSA_AI\norm_patients\test'  # 测试集文件夹

# 获取图像文件和标签文件列表
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.JPG', '.png'))]
label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]

# 创建数据集划分
train_ratio = 0.8  # 70% 作为训练集
val_ratio = 0.1    # 20% 作为验证集
test_ratio = 0.1   # 10% 作为测试集

# 确保每个图片文件有对应的标签文件
data_pairs = [(img, img.replace('.JPG', '.txt').replace('.png', '.txt')) for img in image_files]
data_pairs = [pair for pair in data_pairs if pair[1] in label_files]  # 只保留有标签的图像

# 打乱数据
random.shuffle(data_pairs)

# 根据比例划分数据集
total_count = len(data_pairs)
train_count = int(total_count * train_ratio)
val_count = int(total_count * val_ratio)
test_count = total_count - train_count - val_count

train_data = data_pairs[:train_count]
val_data = data_pairs[train_count:train_count + val_count]
test_data = data_pairs[train_count + val_count:]

# 创建目标文件夹
os.makedirs(output_train, exist_ok=True)
os.makedirs(output_val, exist_ok=True)
os.makedirs(output_test, exist_ok=True)

# 拷贝文件到目标文件夹
def copy_files(data, output_folder):
    for img, lbl in data:
        img_path = os.path.join(image_folder, img)
        lbl_path = os.path.join(label_folder, lbl)
        # 拷贝图像
        shutil.copy(img_path, os.path.join(output_folder, 'images', img))
        # 拷贝标签
        shutil.copy(lbl_path, os.path.join(output_folder, 'labels', lbl))

# 创建子文件夹用于存储图像和标签
os.makedirs(os.path.join(output_train, 'images'), exist_ok=True)
os.makedirs(os.path.join(output_train, 'labels'), exist_ok=True)
os.makedirs(os.path.join(output_val, 'images'), exist_ok=True)
os.makedirs(os.path.join(output_val, 'labels'), exist_ok=True)
os.makedirs(os.path.join(output_test, 'images'), exist_ok=True)
os.makedirs(os.path.join(output_test, 'labels'), exist_ok=True)

# 拷贝数据到训练集、验证集、测试集
copy_files(train_data, output_train)
copy_files(val_data, output_val)
copy_files(test_data, output_test)

print(f"数据集划分完成：")
print(f"训练集：{len(train_data)} 张图像")
print(f"验证集：{len(val_data)} 张图像")
print(f"测试集：{len(test_data)} 张图像")
