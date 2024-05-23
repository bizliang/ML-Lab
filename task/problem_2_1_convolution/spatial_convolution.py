import os

import numpy as np
import imageio
import matplotlib.pyplot as plt
import time

def imshift(img, k, l):
    # Px of high, width
    img_high, img_width = img.shape
    shifted_img = np.zeros((img_high, img_width))
    # double loops implement the periodical boundary conditions
    for i in range(img_high):
        for j in range(img_width):
            shifted_img[(i - k) % img_high, (j - l) % img_width] = img[i, j]

    return shifted_img

def imkernel(tau, s1, s2):
    '''
    The kernel (window) function already, you can use that function to generate your kernel.

    Note: the function is slightly difference than just a matrix. As you can see it returned a
    lambda function object. You need to assign location of the kernel, then it will return
    specified value in that location of the kernel.

    For example, we want a 3x3 window, (note: in this function, we said the center point to be
    location (0,0), so s1 is the absolution distance to the center point, for example: s1 means
    from -1 to 1):

    nu = imkernel(tau,s1,s1);   #<------- generated a 3x3 window funtion
    nu(-1,-1)  #<--------- this will return the top left corner value of the kernel
    nu(0,0)  #<--------- this will return the center value of the kernel

    '''
    # 2x + 2y
    # equation = lambda x, y: 2x + 2y
    # equation(1 , 2)

    w = lambda i, j: np.exp(-(i ** 2 + j ** 2) / (2 * tau ** 2))
    # normalization
    i, j = np.mgrid[-s1:s1, -s2:s2]
    Z = np.sum(w(i, j))
    nu = lambda i, j: w(i, j) / Z * (np.absolute(i) <= s1 & np.absolute(j) <= s2)
    # Note: The return value is a lambda function, which require (i,j) input.
    # nu = w(i,j)/Z
    return nu


def imconvolve_naive(im, nu, s1, s2):
    (n1, n2) = im.shape
    xconv = np.zeros((n1, n2))
    # Iterate over each element of source image
    for i in range(s1, n1 - s1):
        for j in range(s2, n2 - s2):
            sum_val = 0
            # Iterate over each element of the convolution kernel
            for k in range(-s1, s1 + 1):
                for l in range(-s2, s2 + 1):
                    # Sum
                    sum_val += im[i + k, j + l] * nu(k, l)
            xconv[i, j] = sum_val

    return xconv


# Create imconvolve_spatial function
def imconvolve_spatial(im, nu, s1, s2):
    (n1, n2) = im.shape
    # ksize 是卷积核中元素的总数, 表示从 -s1 到 s1 和 -s2 到 s2 的所有整数点。
    ksize = (2 * s1 + 1) * (2 * s2 + 1)
    xstack = np.zeros((n1, n2, ksize))

    idx = 0
    for k in range(-s1, s1 + 1):
        for l in range(-s2, s2 + 1):
            # xstack[:, :, idx] 选择了一个二维数组，这个二维数组对应 xstack 在第三维度索引为 idx 的那一层。
            # imshift(im, k, l) 函数通过将图像 im 按照 (k, l) 的偏移量进行平移，实现周期性边界条件。
            xstack[:, :, idx] = imshift(im, k, l)
            idx += 1

    xconv = np.zeros((n1, n2))
    idx = 0
    for k in range(-s1, s1 + 1):
        for l in range(-s2, s2 + 1):
            # 利用 xstack 中预先计算好的位移图像，与对应的卷积核元素进行乘积并累加。
            xconv += xstack[:, :, idx] * nu(k, l)
            idx += 1

    return xconv



def imshow(img, convolution_img):
    # Display  (a) Original
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(' (a) Original')
    axes[0].axis('off')

    # Display  (b) Periodical
    axes[1].imshow(convolution_img, cmap='gray')
    axes[1].set_title(' (b) Convolution')
    axes[1].axis('off')

    plt.show()


def imsave(save_path, shifted_img):
    if os.path.exists(save_path):
        # Remove the existing file if it exists
        os.remove(save_path)
    # Normalize shifted_img to range [0, 255] and convert to uint8
    shifted_img = ((shifted_img - shifted_img.min()) / (shifted_img.max() - shifted_img.min()) * 255).astype(np.uint8)
    imageio.imwrite(save_path, shifted_img)


# Sample call and plotting code
tau = 1
s1 = 3
s2 = 3
# Load an example image
im = imageio.imread('windmill.png')
nu = imkernel(tau, s1, s2)
# convolution
start_time = time.time()
convolution_image_naive = imconvolve_naive(im, nu, s1, s2)
time_naive = time.time() - start_time
print(f"Naive convolution time: {time_naive} seconds")

# Spatial convolution
start_time = time.time()
convolution_image_spatial = imconvolve_spatial(im, nu, s1, s2)
time_spatial = time.time() - start_time
print(f"Spatial convolution time: {time_spatial} seconds")

# Display images
imshow(im, convolution_image_naive)
imshow(im, convolution_image_spatial)

output_image_name = 'windmill-convolution.png'
imsave(output_image_name, convolution_image_naive)


# 只需两层循环就能完成卷积操作，相比传统的四层循环（对图像和卷积核的每一个元素进行逐个计算）效率更高。
