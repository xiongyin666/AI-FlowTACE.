import socket
import struct
import time
import cv2
import numpy as np
import serial
from collections import deque
from ultralytics import YOLO
from PIL import Image
# 接收图像数据的函数
# server_ip = "127.0.0.1"  # 替换为 C++ 服务器的 IP 地址
# server_port = 6000  # 替换为 C++ 服务器的端口号
# port = 'COM4'
# baudrate = 9600
# detect_model = YOLO(r'D:\Projects\pythonProject\ultralytics-mainfenge\runs\test_xiong\20250112\haloatt_iou_2503\weights\best.pt')
# segment_model = YOLO(r'D:\Projects\pythonProject\ultralytics-mainfenge\runs\model\segment\seg_best.pt')

# 1. 创建 TCP 客户端套接字
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 启用 Keep-Alive
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

# 设置超时（例如：30秒）
client_socket.settimeout(3600)

# 2. 连接到 C++ 服务器
def TCP(server_ip, server_port, detect_model, segment_model):
    try:
        print(f"Attempting to connect to {server_ip}:{server_port} ")
        client_socket.connect((server_ip, server_port))
        print(f"Connected to {server_ip}:{server_port}")

    except socket.error as e:
        print(f"Connection failed on attempt : {e}")

    # 3. 接收图像的宽度和高度（假设是8字节）
    START = str(input('START'))
    ser = serial.Serial(port, baudrate)
    ser.write(START.encode('utf-8'))
    print(f'发送START指令...')
    time.sleep(1)

    command = 1
    ser.write(('W' + str(command)).encode('utf-8'))
    print(f'发送{command}挡位')

    count = 0
    count_not_detected = 0
    index = 0
    cnt = 0

    STOP = False
    while True:
        command = 1
        # ser.write(('W' + str(command)).encode('utf-8'))
        # print(f'发送{command}挡位')
        # if ser.in_waiting > 0:
        #     response = ser.read(ser.in_waiting).decode('gbk')
        #     if response == '接下来需要我干什么':
        #         print('收到停止指令')
        #         STOP = True
        # if STOP:
        #     print('停止')
        #     break

        image_info = client_socket.recv(8)  # 假设宽度和高度各占 4 字节，共 8 字节
        if len(image_info) != 8:
            print("Error: Failed to receive image dimensions.")
            client_socket.close()


        # 解包宽度和高度
        nwidth, nheight = struct.unpack('<II', image_info)
        print(f"Received image dimensions: width={nwidth}, height={nheight}")

        # 4. 接收图像数据
        image_size = nwidth * nheight  # 假设每个像素占 2 字节（16 位图像）
        print(f'尺寸：{image_size}')
        image_data = bytearray(image_size)
        bytes_received = 0
        # frame = 1 / 50
        while bytes_received < image_size:
            try:
                chunk = client_socket.recv(image_size - bytes_received)
                if not chunk:
                    print("Error: Failed to receive image data.")
                    client_socket.close()

                # image_data += chunk
                image_data[bytes_received:bytes_received+len(chunk)] = chunk
                bytes_received += len(chunk)
                time.sleep(0.1)
            except socket.error as e:
                print(f"Error receiving data: {e}")
                client_socket.close()

        array_uint8 = np.frombuffer(image_data, dtype=np.uint8)
        image = array_uint8.reshape((nheight, nwidth))

        im = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        print(f"存储第 {cnt}张图")
        cnt += 1
        print(type(im))
        # im = Image.fromarray(im)
        results = detect_model.predict(
            im,
            save=False,  # 保存预测结果
            imgsz=512,  # 输入图像的大小，可以是整数或w，h
            conf=0.25,  # 用于检测的目标置信度阈值（默认为0.25，用于预测，0.001用于验证）
            iou=0.45,  # 非极大值抑制 (NMS) 的交并比 (IoU) 阈值
            show=True,  # 如果可能的话，显示结果
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
            segmented_result = segment_model.predict(
                detected_area,
                save=False,  # 保存预测结果
                imgsz=512,  # 输入图像的大小，可以是整数或w，h
                conf=0.25,  # 用于检测的目标置信度阈值（默认为0.25，用于预测，0.001用于验证）
                iou=0.50,  # 非极大值抑制 (NMS) 的交并比 (IoU) 阈值
                show=True,  # 如果可能的话，显示结果
                project='runs/predict',  # 项目名称（可选）
                name='0729',  # 实验名称，结果保存在'project/name'目录下（可选）
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
                show_boxes=True,  # 在分割预测中显示边界框
            )
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
                        print('开始count是多少{}'.format(count))
                        if count >= 3:
                            command = command + 1
                            if command > 10:
                                command = 10
                            # 发送数据给服务器
                            # ser.write(('W' + str(command)).encode('utf-8'))
                            # # 等待服务端响应
                            # if ser.in_waiting > 0:
                            #     response = ser.read(ser.in_waiting).decode('utf-8')
                            #     print(f'指令是command是{command}')
                            #     print(f'收到来自服务器的回应：{response}')
                            #     print('此时流速过大，注射泵降至：第{}挡'.format(command))

                            count = 0
                            print('最后count是多少{}'.format(count))

                    else:  # 未检测到目标
                        count_not_detected += 1  # 重置计数器
                        print(f'未检测到的count_not_detected是{count_not_detected}')
                        if count_not_detected >= 3:
                            command = command + 1
                            if command > 5:
                                command = 5
                            # 发送数据给服务器
                            ser.write(('W' + str(command)).encode('utf-8'))
                            if ser.in_waiting > 0:
                                response = ser.read(ser.in_waiting).decode('utf-8')
                                print(f'收到来自服务器的回应：{response}')
                                print('目前无反流Micro pump 适当增速至：第{}挡'.format(command))
                                count_not_detected = 0  # 重置计数器
                                print('最后count_not_detected是多少{}'.format(count_not_detected))

                elif index % 15 == 14:  # 等待10帧
                    count = 0  # 重置计数器
                    count_not_detected = 0

                # 如果当前帧没有检测到目标，也需要重置计数器
            if not detected_in_current_frame:
                count = 0

        index += 1


# 主函数，运行 Python 客户端
if __name__ == '__main__':
    server_ip = "127.0.0.1"  # 替换为 C++ 服务器的 IP 地址
    server_port = 6000  # 替换为 C++ 服务器的端口号
    port = 'COM4'
    baudrate = 9600
    detect_model = YOLO(r'D:\Projects\pythonProject\YOLO11\runs\new_data_test\20250320\v11\weights\best.pt')
    segment_model = YOLO(r'D:\Projects\pythonProject\ultralytics-mainfenge\runs\model\segment\seg_best.pt')
    TCP(server_ip, server_port, detect_model, segment_model)
    # TCP(server_ip, server_port, port, baudrate, detect_model, segment_model)
    time.sleep(0.01)


# 为什么变化过来的python脚本没有体现在 m_bstartRec 为 true 时进入循环，这表示接收图像数据的过程已经开始。使用 ZeroMemory() 将 pRecvImageInfo 中的内存初始化为零，以确保数据没有残留
# 使用 recv() 函数从服务器接收 8 字节的数据。通常这 8 字节包含图像的宽度和高度,如果成功接收到 8 字节数据，则提取图像的宽度和高度。
# memcpy(&nWidth, pRecvImageInfo, 4) 将前 4 字节复制到 nWidth 中，表示图像宽度。
# memcpy(&nHeight, pRecvImageInfo + 4, 4) 将后 4 字节复制到 nHeight 中，表示图像高度。
# 格式化输出图像的信息，并显示在调试窗口。计算图像数据的大小。这里假设每个像素使用 2 字节（可能是 16 位图像）。分配一个足够大的缓冲区 pImageBuf 来存储图像数据。循环接收图像数据，直到接收到完整的图像数据。
# 每次调用 recv() 从服务器接收剩余的数据，并更新 bytesInBuf 变量。