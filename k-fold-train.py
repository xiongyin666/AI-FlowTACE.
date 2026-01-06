import sys
sys.path.append(r'D:\Projects\pythonProject\YOLO11')
from ultralytics import YOLO
import os

def main():
    base_dir = r"E:\DSA_AI\patients_seg_data\internal_data\seg_data\2025-06-20_5-Fold_Cross-val"
    ds_yamls = [os.path.join(base_dir, f'split_{i}', f'split_{i}_dataset.yaml') for i in range(4, 6)]

    results = {}
    for k, dataset_yaml in enumerate(ds_yamls, 1):
        print(f'====== Training Fold {k+3} ======')
        model = YOLO(r'D:\Projects\pythonProject\ultralytics-mainfenge\ultralytics\cfg\models\v8\yolov8-seg.yaml')
        model.train(data=dataset_yaml,
                    epochs=250,
                    imgsz=512,
                    batch=8,
                    save=True,  # (bool) 保存训练检查点和预测结果
                    save_period=-1,  # (int) 每x周期保存检查点（如果小于1则禁用）
                    cache=False,  # (bool) True/ram、磁盘或False。使用缓存加载数据
                    device=0,  # (int | str | list, optional) 运行的设备，例如 cuda device=0 或 device=0,1,2,3 或 device=cpu
                    workers=4,  # (int) 数据加载的工作线程数（每个DDP进程）
                    project='runs/test_xiong_seg/20250619',  # (str, optional) 项目名称
                    name=f'yolo8_seg_{k+3}',  # (str, optional) 实验名称，结果保存在'project/name'目录下
                    exist_ok=False,  # (bool) 是否覆盖现有实验
                    pretrained=True,  # (bool | str) 是否使用预训练模型（bool），或从中加载权重的模型（str）
                    optimizer='Adam',  # (str) 要使用的优化器，选择=[SGD，Adam，Adamax，AdamW，NAdam，RAdam，RMSProp，auto]
                    verbose=True,  # (bool) 是否打印详细输出
                    seed=0,  # (int) 用于可重复性的随机种子
                    deterministic=True,  # (bool) 是否启用确定性模式
                    single_cls=False,  # (bool) 将多类数据训练为单类
                    rect=False,  # (bool) 如果mode='train'，则进行矩形训练，如果mode='val'，则进行矩形验证
                    cos_lr=False,  # (bool) 使用余弦学习率调度器
                    close_mosaic=0,  # (int) 在最后几个周期禁用马赛克增强
                    resume=False,  # (bool) 从上一个检查点恢复训练
                    amp=True,  # (bool) 自动混合精度（AMP）训练，选择=[True, False]，True运行AMP检查
                    fraction=1.0,  # (float) 要训练的数据集分数（默认为1.0，训练集中的所有图像）
                    profile=False,  # (bool) 在训练期间为记录器启用ONNX和TensorRT速度
                    freeze=None,  # (int | list, 可选) 在训练期间冻结前 n 层，或冻结层索引列表。
                    # 分割
                    overlap_mask=True,  # (bool) 训练期间是否应重叠掩码（仅适用于分割训练）
                    mask_ratio=4,  # (int) 掩码降采样比例（仅适用于分割训练）
                    # 分类
                    dropout=0.0,  # (float) 使用丢弃正则化（仅适用于分类训练）
                    # 超参数 ----------------------------------------------------------------------------------------------
                    lr0=0.01,  # (float) 初始学习率（例如，SGD=1E-2，Adam=1E-3）
                    lrf=0.005,  # (float) 最终学习率（lr0 * lrf）
                    momentum=0.937,  # (float) SGD动量/Adam beta1
                    weight_decay=0.0005,  # (float) 优化器权重衰减 5e-4
                    warmup_epochs=3.0,  # (float) 预热周期（分数可用）
                    warmup_momentum=0.8,  # (float) 预热初始动量
                    warmup_bias_lr=0.1,  # (float) 预热初始偏置学习率
                    box=7.5,  # (float) 盒损失增益
                    cls=0.5,  # (float) 类别损失增益（与像素比例）
                    dfl=1.5,  # (float) dfl损失增益
                    pose=12.0,  # (float) 姿势损失增益
                    kobj=1.0,  # (float) 关键点对象损失增益
                    label_smoothing=0.0,  # (float) 标签平滑（分数）
                    nbs=64,  # (int) 名义批量大小
                    hsv_h=0.015,  # (float) 图像HSV-Hue增强（分数）
                    hsv_s=0.7,  # (float) 图像HSV-Saturation增强（分数）
                    hsv_v=0.4,  # (float) 图像HSV-Value增强（分数）
                    degrees=0.0,  # (float) 图像旋转（+/- deg）
                    translate=0.1,  # (float) 图像平移（+/- 分数）
                    scale=0.5,  # (float) 图像缩放（+/- 增益）
                    shear=0.0,  # (float) 图像剪切（+/- deg）
                    perspective=0.0,  # (float) 图像透视（+/- 分数），范围为0-0.001
                    flipud=0.0,  # (float) 图像上下翻转（概率）
                    fliplr=0.5,  # (float) 图像左右翻转（概率）
                    mosaic=1.0,  # (float) 图像马赛克（概率）
                    mixup=0.0,  # (float) 图像混合（概率）
                    copy_paste=0.0,  # (float) 分割复制-粘贴（概率）
        )

        results[k] = model.metrics  # save output metrics for further analysis
        # print(results[k])
        # with open(f'results_fold_{k}.json', 'w', encoding='utf-8') as f:
        #     json.dump(model.metrics, f, ensure_ascii=False, indent=2)
        print(f'====== Fold {k} 完成 ======')

if __name__ == '__main__':
    main()