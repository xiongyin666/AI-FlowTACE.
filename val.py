import sys
sys.path.append(r'D:\Projects\pythonProject\ultralytics-mainfenge')
from ultralytics import YOLO
import os
# import socket
# import torchvision.models as models
# import torch
# import math
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == '__main__':
    model = YOLO(r'D:\Projects\pythonProject\YOLO11\runs\new_data_test\20250323\v11_250\weights\best.pt')

    metrics = model.val(
        val=True,  # (bool) 在训练期间进行验证/测试
        data=r'E:\DSA_AI\norm_patients\my.yaml',
        # data=r'D:\Projects\pythonProject\ultralytics-mainfenge\target_seg\data.yaml',
        split='test',  # (str) 用于验证的数据集拆分，例如'val'、'test'或'train'
        batch=1,  # (int) 每批的图像数量（-1 为自动批处理
        imgsz=512,  # 输入图像的大小，可以是整数或w，h
        device=0,  # 运行的设备，例如 cuda device=0 或 device=0,1,2,3 或 device=cpu
        workers=4,  # 数据加载的工作线程数（每个DDP进程）
        save_json=False,  # 保存结果到JSON文件
        save_hybrid=False,  # 保存标签的混合版本（标签 + 额外的预测）
        conf=0.30,  # 检测的目标置信度阈值（默认为0.25用于预测，0.001用于验证）
        iou=0.5,  # 非极大值抑制 (NMS) 的交并比 (IoU) 阈值
        project='runs/test_xiong',  # 项目名称（可选）
        name='val_fast0.3_att',  # 实验名称，结果保存在'project/name'目录下（可选）
        max_det=300,  # 每张图像的最大检测数
        half=False,  # 使用半精度 (FP16)
        dnn=False,  # 使用OpenCV DNN进行ONNX推断
        plots=True,  # 在训练/验证期间保存图像
        save=False
    )
    print(f"mAP50-95: {metrics.box.map}") # map50-95
    print(f"mAP50: {metrics.box.map50}")  # map50
    print(f"mAP75: {metrics.box.map75}")  # map75
    speed_metrics = metrics.speed
    total_time = sum(speed_metrics.values())
    fps = 1000 / total_time
    print(f"FPS: {fps}") # FPS
