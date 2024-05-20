import os
import numpy as np
import imageio
import matplotlib.pyplot as plt


def imshift(img, k, l):
    # Px of high, width
    img_high, img_width = img.shape
    shifted_img = np.zeros((img_high, img_width))
    # double loops implement the periodical boundary conditions
    for i in range(img_high):
        for j in range(img_width):
            shifted_img[(i - k) % img_high, (j - l) % img_width] = img[i, j]

    return shifted_img


# Load the image
x = imageio.imread('windmill.png')
y = imageio.imread('lake.png')
a = 2.0
b = 3.0
k, l = 100, -50
# Left-hand side of the equation
left = imshift(a * x + b * y, k, l)
# print(f"left: { left }")
# Right-hand side of the equation
right = a * imshift(x, k, l) + b * imshift(y, k, l)
# print(f"right: { right }")
# Check if linear
result = np.array_equal(left, right)
print(f"Linearity Check: { result }")
