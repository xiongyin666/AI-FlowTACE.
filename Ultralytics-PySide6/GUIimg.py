import cv2, sys
import numpy as np
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from ultralytics import YOLO
from collections import Counter


class GradientWidget(QWidget):
    def paintEvent(self, event):
        painter = QPainter(self)
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0, QColor(240, 248, 255))
        gradient.setColorAt(1, QColor(230, 240, 250))
        painter.fillRect(self.rect(), gradient)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.xiaolian_ui()
        self.model1 = None
        self.model2 = None
        self.result = None
        self.con = 0.25

    def xiaolian_ui(self):
        self.setFixedSize(1280, 590)
        self.setMinimumSize(1280, 590)
        self.setMaximumSize(1280, 590)
        self.setWindowTitle('TACE automatic drug delivery assistance recognition program')
        self.setWindowIcon(QIcon(r"D:\Projects\pythonProject\YOLO11\Ultralytics-PySide6-main\图标_20250627175738.png"))

        # 主布局
        main_widget = GradientWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # ===== 左侧控制面板 =====
        left_panel = QWidget()
        left_panel.setMaximumWidth(220)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(8)

        # Logo
        logo = QLabel("""<div style='text-align: center; font-family: Arial; 
								font-size: 10pt; font-weight: bold; color: #2c3e50;'>
								TACE drug reflux segmentation</div>""")
        left_layout.addWidget(logo)

        # 操作按钮
        btn_group = QGroupBox("Operation panel ")
        btn_group.setStyleSheet("""
					QGroupBox {
						border: 1px solid #95a5a6;
						border-radius: 4px;
						margin-top: 6px;
					}
					QGroupBox::title {
						subcontrol-origin: margin;
						left: 6px;
						color: #34495e;
						font: bold 11px;
					}
				""")
        btn_layout = QVBoxLayout()
        buttons = [
            ("Load detection model", self.load_model1),
            ("Load segmentation model", self.load_model2),
            ("Load data", self.select_image),
            ("Stop", self.stop_detect)
        ]
        for text, slot in buttons:
            btn = QPushButton(text)
            btn.setFixedHeight(32)
            btn.setStyleSheet(f"""
						QPushButton {{
							font: 12px 'Microsoft YaHei';
							background: '#f8f9fa' ;
							border: 1px solid #ced4da;
							border-radius: 3px;
							padding: 4px;
						}}
						QPushButton:hover {{ background: #e9ecef; }}
					""")
            btn.clicked.connect(slot)
            btn_layout.addWidget(btn)
        btn_group.setLayout(btn_layout)
        left_layout.addWidget(btn_group)

        # 修改状态指示部分为信息输出框
        status_group = QGroupBox("Identification")
        status_group.setStyleSheet(
            "QGroupBox { border: 1px solid gray; border-radius: 5px; margin-top: 1ex; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 3px; }")
        status_layout = QVBoxLayout()
        self.output_text = QTextEdit(self)
        self.output_text.setReadOnly(True)
        status_layout.addWidget(self.output_text)
        status_group.setLayout(status_layout)
        left_layout.addWidget(status_group)
        left_layout.addStretch()

        # 退出按钮
        exit_btn = QPushButton("Exit the system")
        exit_btn.setFixedHeight(30)
        exit_btn.setStyleSheet("""
					QPushButton {
						background: #e74c3c;
						color: white;
						font: bold 12px 'Microsoft YaHei';
						border-radius: 3px;
						padding: 4px;
					}
					QPushButton:hover { background: #c0392b; }
				""")
        exit_btn.clicked.connect(self.close)
        left_layout.addWidget(exit_btn)

        # ===== 右侧显示区域 =====
        right_panel = QWidget()
        right_layout = QHBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        def create_display(title):
            box = QWidget()
            box.setStyleSheet("""
						background: white;
						border: 1px solid #bdc3c7;
						border-radius: 5px;
					""")
            layout = QVBoxLayout(box)
            layout.setContentsMargins(0, 0, 0, 0)

            # 标题栏
            title_bar = QLabel(title)
            title_bar.setStyleSheet("""
						background: #f8f9fa;
						color: #2c3e50;
						font: bold 30px 'Microsoft YaHei';
						padding: 6px;
						border-bottom: 1px solid #ced4da;
					""")
            title_bar.setAlignment(Qt.AlignCenter)
            layout.addWidget(title_bar)

            # 图像显示
            img_label = QLabel()
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setMinimumSize(500, 500)
            img_label.setMaximumSize(500, 500)
            img_label.setStyleSheet("background: #2c3e50;")
            layout.addWidget(img_label)

            return box, img_label

        self.cam_box, self.label1 = create_display("Original picture")
        self.result_box, self.label2 = create_display("Forecast results")
        right_layout.addWidget(self.cam_box)
        right_layout.addWidget(self.result_box)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)
        self.setCentralWidget(main_widget)

    def load_model1(self):
        self.result = None
        model_path, _ = QFileDialog.getOpenFileName(self, "Select model file", filter='*.pt')
        if model_path:
            self.model1 = YOLO(model_path)
            self.output_text.append(f"Model 1 has been loaded: {model_path}")

    def load_model2(self):
        self.result = None
        model_path, _ = QFileDialog.getOpenFileName(self, "Select model file", filter='*.pt')
        if model_path:
            self.model2 = YOLO(model_path)
            self.output_text.append(f"Model 2 has been loaded: {model_path}")

    # def select_image(self):
    #     self.result = None
    #     image_path, fileType = QFileDialog.getOpenFileName(self, "选择图片文件", filter='*.jpg *.png *.bmp *.mp4 *.avi *.mov')
    #     if image_path:
    #         img = cv2.imread(image_path)
    #         self.detect_image(img)

    def select_image(self):
        self.result = None
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select image or video files",
            filter="Image and video files (*.jpg *.png *.bmp *.mp4 *.avi *.mov);;Picture files (*.jpg *.png *.bmp);;Video file  (*.mp4 *.avi *.mov)"
        )
        if file_path:
            if file_path.lower().endswith(('.mp4', '.avi', '.mov')):
                self.detect_video(file_path)
            else:
                img = cv2.imread(file_path)
                self.detect_image(img)

    def stop_detect(self):
        self.result = None
        img = cv2.cvtColor(np.zeros((580, 550), np.uint8), cv2.COLOR_BGR2RGB)
        img = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
        self.label1.setPixmap(QPixmap.fromImage(img))
        self.label2.setPixmap(QPixmap.fromImage(img))
        self.display_statistics()

    def close(self):
        exit()

    def detect_video(self, video_path):
        if self.model1 is None or self.model2 is None:
            self.output_text.setText("Please load two models first!")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.output_text.setText("Unable to open video file")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 用检测模型推理
            results1 = self.model1.predict(frame, conf=False, save=False, show_boxes=True, show_labels=False,
                                           line_width=1)
            # 将检测结果画到图像上
            detected_area = results1[0].plot()

            # 用分割模型推理
            results2 = self.model2.predict(detected_area, save=False, show_boxes=False, show_labels=False)
            annotated_image = results2[0].plot()

            # 转换为RGB QImage后显示到label2
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            h, w, ch = annotated_image.shape
            qimage2 = QImage(annotated_image.data, w, h, 3 * w, QImage.Format_RGB888)
            pixmap2 = QPixmap.fromImage(qimage2)
            pixmap2 = pixmap2.scaled(self.label2.size(), Qt.AspectRatioMode.IgnoreAspectRatio)
            self.label2.setPixmap(pixmap2)

            QApplication.processEvents()  # 保持界面刷新

        cap.release()
        self.output_text.setText("Video detection completed")

    def detect_image(self, img):
        if self.model1 is not None and self.model2 is None:
            frame = img
            results1 = self.model1.predict(frame,
                          save=False,  # 保存预测结果
                          imgsz=512,  # 输入图像的大小，可以是整数或w，h
                          conf=0.25,  # 用于检测的目标置信度阈值（默认为0.25，用于预测，0.001用于验证）
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

            for i, r in enumerate(results1):
                detected_area = r.plot()

                # 使用分割模型对检测到的区域进行分割
                results2 = self.model2.predict(detected_area,
                    save=False,  # 保存预测结果
                    imgsz=512,  # 输入图像的大小，可以是整数或w，h
                    conf=0.2,  # 用于检测的目标置信度阈值（默认为0.25，用于预测，0.001用于验证）
                    iou=0.3,  # 非极大值抑制 (NMS) 的交并比 (IoU) 阈值
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
                    show_boxes=False  # 在分割预测中显示边界框image)
                    )

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height1, width1, channel1 = frame.shape
                bytesPerLine1 = 3 * width1
                qimage1 = QImage(image_rgb.data, width1, height1, bytesPerLine1, QImage.Format_RGB888)
                pixmap1 = QPixmap.fromImage(qimage1)
                self.label1.setPixmap(pixmap1.scaled(self.label1.size(), Qt.AspectRatioMode.IgnoreAspectRatio))
                annotated_image = results2[0].plot()
                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)  # 转换为 RGB
                height2, width2, channel2 = annotated_image.shape
                bytesPerLine2 = 3 * width2
                qimage2 = QImage(annotated_image.data, width2, height2, bytesPerLine2, QImage.Format_RGB888)
                pixmap2 = QPixmap.fromImage(qimage2)
                pixmap2 = pixmap2.scaled(self.label2.size(), Qt.AspectRatioMode.IgnoreAspectRatio)
                self.result = results2
                self.label2.setPixmap(pixmap2)
                self.display_statistics()

    def stat(self):
        detected_classes = []
        target_range = {0}  # 修改为自己的类别
        if self.result == None:
            return None
        for r in self.result:
            classes = r.boxes.cls.cpu().numpy().astype(int).tolist()
            # 筛选出在目标范围内的类别，并去重
            detected_classes.extend([cls for cls in classes if cls in target_range])
        class_counts = Counter(detected_classes)
        class_counts_dic = dict(class_counts)
        return class_counts_dic

    def display_statistics(self):
        class_counts = self.stat()
        if class_counts == None:
            self.output_text.setText('')
            return
        # 修改class_labels为自己的类别对应关系，可中文
        class_labels = {
            0: "target"
        }
        # 构建输出字符串
        output_string = ""
        for class_id, count in class_counts.items():
            label = class_labels.get(class_id, f"Class{class_id}")  # 如果没有找到标签，则使用默认标签
            output_string += f"{label}: {count} \n"

        self.output_text.setText(output_string)

    def closeEvent(self, event: QEvent):
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())