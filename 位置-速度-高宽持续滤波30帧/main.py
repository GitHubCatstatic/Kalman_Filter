# 卡尔曼滤波的运用
# 概述：追踪一个运动的物体，通过卡尔曼滤波来估计物体的位置
# 背景：视觉模块：maixcam，被追踪的物体：apriltags标签
# 思路：利用maixcam的find_apriltags()函数来检测apriltags标签，返回标签的位置信息作为观测值
# 利用卡尔曼滤波来估计标签的位置，并画出和apriltags相同高宽的检测框
from maix import image, camera, display, app, time
import numpy as np

# 创建显示对象
disp = display.Display()

# 创建摄像头对象
cam = camera.Camera(360, 240)

# 卡尔曼滤波参数初始化
# 状态向量 [x, y, vx, vy, w, h, vw, vh]
a = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
c = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=float)

# 初始时间间隔
dt = 1

# 状态转移矩阵
F = np.array([[1, 0, 3*dt, 0, 0, 0, 0, 0],
              [0, 1, 0, 3*dt, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 2*dt, 0],
              [0, 0, 0, 0, 0, 1, 0, 2*dt],
              [0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 1]], dtype=float)

# 观测矩阵
H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 1]], dtype=float)

# 过程噪声协方差
Q = np.eye(8, dtype=float) * 0.1

# 观测噪声协方差
R = np.eye(8, dtype=float) * 0.1

# 状态协方差矩阵
P = np.eye(8, dtype=float) * 100

current_time = time.time_ms()
last_time = current_time

x_pos = 0 # 预测x的位置
y_pos = 0 # 预测y的位置
vx = 0 # 速度
vy = 0
vw = 0
vh = 0
w = 0 # 宽高
h = 0
x = 0
y = 0

cnt = 0

def kalman_filter(a, P, F, H, Q, R, c, detected):
    # 预测步骤
    a = np.dot(F, a)
    P = np.dot(F, np.dot(P, F.T)) + Q

    if detected:
        # 计算卡尔曼增益
        S = np.dot(H, np.dot(P, H.T)) + R
        K = np.dot(P, np.dot(H.T, np.linalg.inv(S)))

        # 更新步骤
        b = c - np.dot(H, a)
        a = a + np.dot(K, b)
        P = P - np.dot(K, np.dot(H, P))

    return a, P

# 更新主循环的状态和观测
while not app.need_exit():
    img = cam.read()
    tags = img.find_apriltags()

    current_time = time.time_ms()
    dt = (current_time - last_time) # 保持毫秒单位
    last_time = current_time

    detected = False
    if tags:
        for tag in tags:
            img.draw_rect(tag.x(), tag.y(), tag.w(), tag.h(), image.COLOR_GREEN)
            vx = (tag.x() - x) / dt
            vy = (tag.y() - y) / dt
            vw = (tag.w() - w) / dt
            vh = (tag.h() - h) / dt
            x, y = tag.x(), tag.y()
            w, h = tag.w(), tag.h()
            detected = True
            cnt = 0

    c = np.array([x, y, vx, vy, w, h, vw, vh], dtype=float)
    if cnt < 30:
        a, P = kalman_filter(a, P, F, H, Q, R, c, detected)
        x_pos, y_pos, vx, vy, w, h, vw, vh = int(a[0]), int(a[1]), a[2], a[3], int(a[4]), int(a[5]), a[6], a[7]
    if not detected:
        cnt += 1
    img.draw_rect(x_pos, y_pos, w, h, image.COLOR_RED)
    disp.show(img)
