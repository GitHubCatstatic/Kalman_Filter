# 卡尔曼滤波的运用：位置滤波
# 概述：追踪一个运动的物体，通过卡尔曼滤波来估计物体的位置
# 背景：视觉模块：maixcam，被追踪的物体：apriltags标签
# 思路：利用maixcam的find_apriltags()函数来检测apriltags标签，返回标签的位置信息作为观测值
# 利用卡尔曼滤波来估计标签的位置，并画出和apriltags相同高宽的检测框
from maix import image, camera, display, app, time
import numpy as np

# 创建显示对象
disp = display.Display()

# 创建摄像头对象
cam = camera.Camera(332, 224)

# 卡尔曼滤波参数初始化
# 状态向量 [x, y]
a = np.array([0, 0], dtype=float)
c = np.array([0, 0], dtype=float)

# 状态转移矩阵:描述了状态向量如何从一个时间步转移到下一个时间步
F = np.array([[1, 0],
              [0, 1]], dtype=float)

# 观测矩阵：这是将状态向量映射到观测向量的矩阵
H = np.array([[1, 0],
              [0, 1]], dtype=float)

# 过程噪声协方差
Q = np.array([[1, 0],
              [0, 1]], dtype=float) * 0.1

# 观测噪声协方差
R = np.array([[1, 0],
              [0, 1]], dtype=float) * 0.1

# 状态协方差矩阵
P = np.eye(2, dtype=float) * 100  # 单位矩阵

# 参数初始化
current_time = 0
prev_time = time.time_ms()

x_pos = 0  # 预测的x的位置
y_pos = 0  # 预测的y的位置
w = 0  # 宽
h = 0  # 高
x = 0
y = 0

# 为了在中途检测不到标签时继续预测，需要进行以下修改：
# 1.维持预测：即使在没有观测值（即未检测到标签）时，依旧使用卡尔曼滤波进行预测步骤，但跳过更新步骤。
# 2.记录标签检测状态：在每次检测到标签时记录状态，未检测到标签时则维持上一状态的预测。
# 但是位置预测不包含速度信息所以，虽然继续预测了，但是还是在原地不动
detected = False  # 标记是否检测到标签

def kalman_filter(a, P, F, H, Q, R, c, detected):
    # 预测步骤
    a = np.dot(F, a)  # 状态预测
    P = np.dot(F, np.dot(P, F.T)) + Q  # 误差协方差预测

    if detected:
        # 计算卡尔曼增益
        S = np.dot(H, np.dot(P, H.T)) + R  # 观测预测协方差
        K = np.dot(P, np.dot(H.T, np.linalg.inv(S)))  # 卡尔曼增益

        # 更新步骤
        b = c - np.dot(H, a)  # 观测残差
        a = a + np.dot(K, b)  # 状态更新
        P = P - np.dot(K, np.dot(H, P))  # 误差协方差更新

    return a, P  # 返回更新后的状态向量和误差协方差矩阵

while not app.need_exit():
    img = cam.read()  # 获取图像
    tags = img.find_apriltags()  # 检测apriltags标签

    current_time = time.time_ms()
    dt = (current_time - prev_time) / 1000  # 计算时间间隔
    prev_time = current_time

    detected = False
    if tags:
        for tag in tags:
            img.draw_rect(tag.x(), tag.y(), tag.w(), tag.h(), image.COLOR_GREEN)
            x, y = tag.x(), tag.y()
            w, h = tag.w(), tag.h()
            c = np.array([x, y], dtype=float)  # 观测值
            detected = True

    a, P = kalman_filter(a, P, F, H, Q, R, c, detected)
    # 计算检测框的位置和大小
    x_pos, y_pos = int(a[0]), int(a[1])
    img.draw_rect(x_pos, y_pos, w, h, image.COLOR_RED)
    disp.show(img)
    # print(f"time: {time.time_ms() - current_time}ms, fps: {1000 / (time.time_ms() - current_time)}")
