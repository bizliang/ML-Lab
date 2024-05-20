import os

import numpy as np
import imageio
import matplotlib.pyplot as plt
import time


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
convolution_image = imconvolve_naive(im, nu, s1, s2)
time_naive = time.time() - start_time
print(f"Naive convolution time: {time_naive} seconds")

imshow(im, convolution_image)

output_image_name = 'windmill-convolution.png'
imsave(output_image_name, convolution_image)
