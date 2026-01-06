import numpy as np
import os
from ultralytics import YOLO
import time
import glob
import cv2
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pandas as pd


det_model = YOLO(r"D:\Projects\pythonProject\ultralytics-mainfenge\runs\test_xiong\20250525\v8att_2502\weights\best.pt")
seg_model = YOLO(r"D:\Projects\pythonProject\ultralytics-mainfenge\runs\test_xiong_seg\the_best\v8_seg300\weights\best.pt")  # YOLOv8n模型

frame = r"C:\Users\Administrator\Desktop\1"
img_paths = glob.glob(frame + r'\*.png')  # 或png，根据你图片类型

count = 0
index = 0
command = 1
data = {
    "Image Index": [],
    "Inference Time (s)": [],
    "Command": []
}
df = pd.DataFrame(data)
start_time = time.time()
for img_path in img_paths:
    # print(img_path)
    t1 = time.time()
    results = det_model.predict(
        img_path,
        device=0,
        save=False,  # 保存预测结果
        imgsz=512,  # 输入图像的大小，可以是整数或w，h
        conf=0.1,  # 用于检测的目标置信度阈值（默认为0.25，用于预测，0.001用于验证）
        iou=0.45,  # 非极大值抑制 (NMS) 的交并比 (IoU) 阈值
        show=False,  # 如果可能的话，显示结果
        project='runs/predict',  # 项目名称（可选）
        name='detect',  # 实验名称，结果保存在'project/name'目录下（可选）
        save_txt=False,  # 保存结果为 .txt 文件
        save_conf=False,  # 保存结果和置信度分数
        save_crop=False,  # 保存裁剪后的图像和结果
        show_labels=False,  # 在图中显示目标标签
        show_conf=False,  # 在图中显示目标置信度分数
        vid_stride=1,  # 视频帧率步长
        line_width=1,  # 边界框线条粗细（像素）
        visualize=False,  # 可视化模型特征
        augment=False,  # 对预测源应用图像增强
        agnostic_nms=False,  # 类别无关的NMS
        retina_masks=False,  # 使用高分辨率的分割掩码
        show_boxes=True  # 在分割预测中显示边界框image)
    )
    for i, r in enumerate(results):
        detected_area = r.plot()

        # 使用分割模型对检测到的区域进行分割
        segmented_result = seg_model.predict(
            detected_area,
            device=0,
            save=True,  # 保存预测结果
            imgsz=512,  # 输入图像的大小，可以是整数或w，h
            conf=0.25,  # 用于检测的目标置信度阈值（默认为0.25，用于预测，0.001用于验证）
            iou=0.45,  # 非极大值抑制 (NMS) 的交并比 (IoU) 阈值
            show=False,  # 如果可能的话，显示结果
            project=r"C:\Users\Administrator\Desktop\1",  # 项目名称（可选）
            name='1',  # 实验名称，结果保存在'project/name'目录下（可选）
            save_txt=False,  # 保存结果为 .txt 文件
            save_conf=False,  # 保存结果和置信度分数
            save_crop=False,  # 保存裁剪后的图像和结果
            show_labels=False,  # 在图中显示目标标签
            show_conf=False,  # 在图中显示目标置信度分数
            vid_stride=1,  # 视频帧率步长
            line_width=1,  # 边界框线条粗细（像素）
            visualize=False,  # 可视化模型特征
            augment=False,  # 对预测源应用图像增强
            agnostic_nms=False,  # 类别无关的NMS
            retina_masks=False,  # 使用高分辨率的分割掩码
            show_boxes=False,  # 在分割预测中显示边界框
        )
        # print(f'第{index + 1}张分割完成')
        print('-' * 50)
        t2 = time.time()
        inference_time = t2 - t1

        print(f'第{index + 1}张分割完成')
        print('-' * 50)

        detected_in_current_frame = False
        for j, s in enumerate(segmented_result):
            if index % 15 < 5:
                box = s.boxes
                cls = box.cls

                if cls.numel() >= 1:
                    detected_in_current_frame = True
                    count += 1
                    print(f'开始count是多少{count}')

                    if count >= 3:
                        command = command + 1
                        if command >= 5:
                            command = 5
                        # 发送数据给服务器
                        # ser.write(('W' + str(command)).encode('utf-8'))
                        print(f'给药指令为W+{command}----------------------------------------------------------------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                        count = 0
                        print(f'最后count是多少{count}')
                else:
                    count = 0  # 重置计数器

            elif index % 15 == 14:  # 等待10帧
                count = 0  # 重置计数器

        if not detected_in_current_frame:
            count = 0
    df = df._append({
        "Image Index": index,
        "Inference Time (s)": inference_time,
        "Command": f"W+{command}"
    }, ignore_index=True)

    print(f"[{index}] 当前图片推理时间: {inference_time:.3f} 秒")
    print(f'给药指令为W+{command}')
    index += 1

df.to_excel(r'C:\Users\Administrator\Desktop\1.xlsx', index=False)

print("推理时间和给药指令已保存到 Excel 文件中！")