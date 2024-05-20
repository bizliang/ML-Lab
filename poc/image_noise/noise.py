import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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
    # Generate Gaussian noise
    sigma = var ** 0.5
    gaussian_noise = np.random.normal(mean, sigma, image.shape)

    # Add the noise to the image
    noisy_image = image + gaussian_noise * 255
    noisy_image = np.clip(noisy_image, 0, 255)  # Clip values to be in [0, 255]

    return noisy_image.astype(np.uint8)


# Add Gaussian noise to the image
noisy_image = add_gaussian_noise(image_np)

# Display the noisy image
plt.figure(figsize=(6, 6))
plt.imshow(noisy_image.squeeze(), cmap='gray')
plt.title('Image with Gaussian Noise')
plt.axis('off')
plt.show()


# Function to apply a simple mean filter to remove noise
def mean_filter(image, kernel_size=3):
    padded_image = np.pad(image, ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2), (0, 0)),
                          mode='constant')
    filtered_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                filtered_image[i, j, k] = np.mean(padded_image[i:i + kernel_size, j:j + kernel_size, k])

    return filtered_image.astype(np.uint8)


# Apply mean filter to the noisy image
filtered_image = mean_filter(noisy_image)

# Display the filtered image
plt.figure(figsize=(6, 6))
plt.imshow(filtered_image.squeeze(), cmap='gray')
plt.title('Filtered Image (Mean Filter)')
plt.axis('off')
plt.show()
