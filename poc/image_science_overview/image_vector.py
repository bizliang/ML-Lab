import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Read the image and convert it to a numpy array

image = Image.open("windmill.png")
image_np = np.array(image)

# Check the shape of the image (Height, Width, [Channels])
print("Image shape:", image_np.shape)

# If the image is colored (Height, Width, 3)
if len(image_np.shape) == 3:
    # Extract each color channel (R, G, B)
    red_channel = image_np[:, :, 0]
    green_channel = image_np[:, :, 1]
    blue_channel = image_np[:, :, 2]

    # Display each color channel
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(red_channel, cmap='Reds')
    plt.title('Red Channel')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(green_channel, cmap='Greens')
    plt.title('Green Channel')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(blue_channel, cmap='Blues')
    plt.title('Blue Channel')
    plt.axis('off')

    plt.show()

    # Convert to grayscale image by averaging the color channels
    gray_image = np.mean(image_np, axis=2)  # Calculate the mean to get the grayscale image

else:
    # The image is already grayscale
    gray_image = image_np

# Display the grayscale image
plt.figure(figsize=(6, 6))
plt.imshow(gray_image, cmap='gray')
plt.title('Gray Image')
plt.axis('off')
plt.show()

# Print the shape of the grayscale image (Height, Width)
print("Gray image shape:", gray_image.shape)

# Convert the image to a vector
image_vector = gray_image.flatten()  # Flatten the 2D array to a 1D vector

# Print the vector and its shape
print("Image vector shape:", image_vector.shape)
print("Image vector:", image_vector)
