import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter, median_filter

# Read the image and convert it to a numpy array
image = Image.open("windmill.png")
image_np = np.array(image)

# If the image is grayscale, ensure it has a shape of (Height, Width, 1)
if len(image_np.shape) == 2:
    image_np = image_np[:, :, np.newaxis]

# Display the original image
plt.figure(figsize=(6, 6))
plt.imshow(image_np.squeeze(), cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.show()

# Function to add Gaussian noise to an image
def add_gaussian_noise(image, mean=0, var=0.01):
    sigma = var**0.5
    gaussian_noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + gaussian_noise * 255
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

# Add Gaussian noise to the image
noisy_image = add_gaussian_noise(image_np)

# Display the noisy image
plt.figure(figsize=(6, 6))
plt.imshow(noisy_image.squeeze(), cmap='gray')
plt.title('Image with Gaussian Noise')
plt.axis('off')
plt.show()

# Function to apply a Gaussian filter to remove noise using SciPy
def scipy_gaussian_filter(image, sigma=1):
    if len(image.shape) == 3 and image.shape[2] == 3:
        filtered_image = np.zeros_like(image)
        for i in range(3):
            filtered_image[:, :, i] = gaussian_filter(image[:, :, i], sigma=sigma)
    else:
        filtered_image = gaussian_filter(image, sigma=sigma)
    return filtered_image

# Function to apply a Median filter to remove noise using SciPy
def scipy_median_filter(image, size=3):
    if len(image.shape) == 3 and image.shape[2] == 3:
        filtered_image = np.zeros_like(image)
        for i in range(3):
            filtered_image[:, :, i] = median_filter(image[:, :, i], size=size)
    else:
        filtered_image = median_filter(image, size=size)
    return filtered_image

# Apply Gaussian filter to the noisy image using SciPy
scipy_gaussian_filtered_image = scipy_gaussian_filter(noisy_image, sigma=1)

# Apply Median filter to the noisy image using SciPy
scipy_median_filtered_image = scipy_median_filter(noisy_image, size=3)

# Display the filtered images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(scipy_gaussian_filtered_image.squeeze(), cmap='gray')
plt.title('Gaussian Filtered Image (SciPy)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(scipy_median_filtered_image.squeeze(), cmap='gray')
plt.title('Median Filtered Image (SciPy)')
plt.axis('off')

plt.show()

# Print the shapes of the filtered images
print("Gaussian filtered image shape:", scipy_gaussian_filtered_image.shape)
print("Median filtered image shape:", scipy_median_filtered_image.shape)
