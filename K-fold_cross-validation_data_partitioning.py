# 导入所需的模块和库
import datetime  # 用于处理日期和时间
import shutil  # 用于文件和目录操作
from pathlib import Path  # 用于处理文件路径
from collections import Counter  # 用于计数可哈希对象的出现次数
import sys
sys.path.append(r'D:\Projects\pythonProject\YOLO11')
import yaml  # 用于读写YAML文件（一种用于配置文件的格式）
import numpy as np  # 用于数值计算
import pandas as pd  # 用于数据处理和分析
from ultralytics import YOLO  # 用于使用YOLO算法进行目标检测
from sklearn.model_selection import KFold  # 用于进行K折交叉验证

dataset_path = Path(r"E:\DSA_AI\patients_seg_data\internal_data\seg_data") # replace with 'path/to/dataset' for your custom data
labels = sorted(dataset_path.rglob("*labels/*.txt")) # all data in 'labels'

yaml_file = Path(r"E:\DSA_AI\patients_seg_data\internal_data\seg_data\my.yaml")

with open(yaml_file, 'r', encoding="utf8") as y:
    classes = yaml.safe_load(y)['names']

print(type(classes))
cls_idx = sorted(classes)
indx = [l.stem for l in labels]
labels_df = pd.DataFrame([], columns=cls_idx, index=indx)

for label in labels:
    lbl_counter = Counter()
    with open(label, 'r') as lf:
        lines = lf.readlines()

    for l in lines:
        lbl_counter[int(l.split(' ')[0])] += 1
    labels_df.loc[label.stem] = lbl_counter
labels_df = labels_df.fillna(0.0)
ksplit = 5
kf = KFold(n_splits=ksplit, shuffle=True, random_state=20)

kfolds = list(kf.split(labels_df))
folds = [f'split_{n}' for n in range(1, ksplit + 1)]
folds_df = pd.DataFrame(index=indx, columns=folds)

for idx, (train, val) in enumerate(kfolds, start=1):
    folds_df[f'split_{idx}'].loc[labels_df.iloc[train].index] = 'train'
    folds_df[f'split_{idx}'].loc[labels_df.iloc[val].index] = 'val'

fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)

for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
    train_totals = labels_df.iloc[train_indices].sum()

    val_totals = labels_df.iloc[val_indices].sum()
    ratio = val_totals / (train_totals + 1E-7)
    fold_lbl_distrb.loc[f'split_{n}'] = ratio


save_path = Path(dataset_path / f'{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val')
save_path.mkdir(parents=True, exist_ok=True)

images = sorted(dataset_path.rglob("*images/*.png"))  # 可根据需要更改文件扩展名
# print(images)


ds_yamls = []

for split in folds_df.columns:
    # 创建当前子集的目录
    split_dir = save_path / split
    split_dir.mkdir(parents=True, exist_ok=True)

    (split_dir / 'train' / 'images').mkdir(parents=True, exist_ok=True)
    (split_dir / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
    (split_dir / 'val' / 'images').mkdir(parents=True, exist_ok=True)
    (split_dir / 'val' / 'labels').mkdir(parents=True, exist_ok=True)

    dataset_yaml = split_dir / f'{split}_dataset.yaml'
    ds_yamls.append(dataset_yaml)

    # 将数据集信息写入数据集配置文件
    with open(dataset_yaml, 'w') as ds_y:
        yaml.safe_dump({
            'path': save_path.as_posix(),  # 数据集根目录路径
            'train': 'train',  # 训练集所在的子目录名称
            'val': 'val',  # 验证集所在的子目录名称
            'names': classes  # 数据集类别信息
        }, ds_y)


for image, label in zip(images, labels):
    for split, k_split in folds_df.loc[image.stem].items():
        # 获取目标目录的路径
        img_to_path = save_path / split / k_split / 'images'  # 图像文件的目标目录
        lbl_to_path = save_path / split / k_split / 'labels'  # 标签文件的目标目录
        shutil.copy(image, img_to_path / image.name)  # 复制图像文件
        shutil.copy(label, lbl_to_path / label.name)  # 复制标签文件


