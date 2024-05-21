import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
# img = imread('geisel.jpg')
image = cv2.imread('geisel.jpg', cv2.IMREAD_GRAYSCALE)

# 应用高斯滤波器进行平滑
blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)

# 使用Canny函数进行边缘检测
edges = cv2.Canny(blurred_image, threshold1=100, threshold2=200)

# 显示原始图像和检测到的边缘
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(edges, cmap='gray'), plt.title('Canny Edges')
plt.show()
