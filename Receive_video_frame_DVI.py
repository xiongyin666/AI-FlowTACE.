import os

import cv2
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)


save_folder = r"E:\采集卡"
if not cap.isOpened():
    print('设备无法打开')
    exit()
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print('无法读取')
        break
    cv2.imshow('usb3.0 capture card output', frame)
    frame_filename = os.path.join(save_folder, f"frame_{frame_count:04d}.png")
    cv2.imwrite(frame_filename, frame)
    print(f"保存: {frame_filename}")
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# import cv2
# for i in range(5):
#     cap = cv2.VideoCapture(i)
#     if cap.isOpened():
#         print(f'{i}可用')
#         cap.release()
#