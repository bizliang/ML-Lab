import numpy as np
from imageio import imread, imwrite
import matplotlib.pyplot as plt
from scipy import signal
import os


def rgb2gray(rgb):
    """
    Convert an RGB image to grayscale.

    Parameters:
    rgb (ndarray): RGB image.

    Returns:
    ndarray: Grayscale image.
    """
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def canny_edge(img, te):
    """
    Perform Canny edge detection on a grayscale image.

    Parameters:
    img (ndarray): Grayscale image.
    te (float): Threshold value for edge detection.

    Returns:
    tuple: Contains the smoothed image, gradient magnitude image,
           non-maximum suppression (NMS) image, and thresholded edge image.
    """
    # Step 1: Smoothing
    gaussian_kernel = np.array([[2, 4, 5, 4, 2],
                                [4, 9, 12, 9, 4],
                                [5, 12, 15, 12, 5],
                                [4, 9, 12, 9, 4],
                                [2, 4, 5, 4, 2]]) / 159

    # Apply Gaussian filter to smooth the image and reduce noise
    smoothed_image = signal.convolve2d(img, gaussian_kernel, boundary='symm', mode='same')

    # Step 2: Finding Gradients
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Convolve the smoothed image with Sobel kernels to find gradients
    Gx = signal.convolve2d(smoothed_image, kx, boundary='symm', mode='same')
    Gy = signal.convolve2d(smoothed_image, ky, boundary='symm', mode='same')

    # Compute the gradient magnitude and direction
    magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
    direction = np.arctan2(Gy, Gx)

    # Step 3: Non-maximum Suppression (NMS)
    nms_image = np.zeros_like(magnitude)
    angle = direction * 180. / np.pi
    angle[angle < 0] += 180

    # Iterate over the image to apply NMS
    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            try:
                q = 255
                r = 255

                # Determine the neighbors to compare based on gradient direction
                # Angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = magnitude[i, j + 1]
                    r = magnitude[i, j - 1]
                # Angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = magnitude[i + 1, j - 1]
                    r = magnitude[i - 1, j + 1]
                # Angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = magnitude[i + 1, j]
                    r = magnitude[i - 1, j]
                # Angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = magnitude[i - 1, j - 1]
                    r = magnitude[i + 1, j + 1]

                # Keep the pixel value if it is a local maximum
                if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                    nms_image[i, j] = magnitude[i, j]
                else:
                    nms_image[i, j] = 0
            except IndexError as e:
                pass

    # Step 4: Thresholding
    threshold_image = np.zeros_like(nms_image)
    strong_pixel = np.max(nms_image)

    # Apply thresholding to keep strong edges
    for i in range(nms_image.shape[0]):
        for j in range(nms_image.shape[1]):
            if nms_image[i, j] >= te * strong_pixel:
                threshold_image[i, j] = 255

    return smoothed_image, magnitude, nms_image, threshold_image


def imshow(title, img, target_image):
    # Display images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(target_image, cmap='gray')
    axes[1].set_title(title)
    axes[1].axis('off')

    plt.show()


def imsave(save_path, img):
    if os.path.exists(save_path):
        # Remove the existing file if it exists
        os.remove(save_path)
    # Normalize img to range [0, 255] and convert to uint8
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    imwrite(save_path, img)


# Import image
img = imread('geisel.jpg')

# Convert to Gray scale image
img = rgb2gray(img)
plt.imshow(img, cmap='gray')
plt.show()

# Select Threshold
# if te < 0.1, the image will contain more noise, and the edge of it is very noisy
# if te > 0.5, the image will only keep the strong edge, many significant px in image's structure might ignore
# if te > 1.0, will output black image, because all px in image was not considered as edge
te = 0.15

# Sample call
smoothed_image, magnitude, nms_image, threshold_image = canny_edge(img, te)

# imshow
imshow('Smoothed Image', img, smoothed_image)
imshow('Gradient Magnitude Image', img, magnitude)
imshow('NMS Image', img, nms_image)
imshow('Final Edge Image', img, threshold_image)

# imsave
imsave('geisel_canny_smoothed_image.jpg', smoothed_image)
imsave('geisel_canny_magnitude.jpg', magnitude)
imsave('geisel_canny_nms_image.jpg', nms_image)
imsave('geisel_canny_threshold_image.jpg', threshold_image)

# The value for te
print(f'The threshold value (te) used: {te}')
