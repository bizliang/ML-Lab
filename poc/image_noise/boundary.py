import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Read the image and convert it to a numpy array
image = Image.open("windmill.png")
image_np = np.array(image)

# Display the original image
plt.figure(figsize=(6, 6))
plt.imshow(image_np, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.show()

# Function to apply different padding techniques
def apply_padding(image, padding_size, mode):
    if mode == 'zero':
        padded_image = np.pad(image, pad_width=padding_size, mode='constant', constant_values=0)
    elif mode == 'mirror':
        padded_image = np.pad(image, pad_width=padding_size, mode='reflect')
    elif mode == 'expand':
        padded_image = np.pad(image, pad_width=padding_size, mode='edge')
    return padded_image

# Define padding size
padding_size = ((10, 10), (10, 10))  # Pad 10 pixels on each side

# Apply zero padding
zero_padded_image = apply_padding(image_np, padding_size, 'zero')

# Apply mirror padding
mirror_padded_image = apply_padding(image_np, padding_size, 'mirror')

# Apply expand padding
expand_padded_image = apply_padding(image_np, padding_size, 'expand')

# Display the padded images
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(zero_padded_image, cmap='gray')
plt.title('Zero Padding')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mirror_padded_image, cmap='gray')
plt.title('Mirror Padding')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(expand_padded_image, cmap='gray')
plt.title('Expand Padding')
plt.axis('off')

plt.show()
