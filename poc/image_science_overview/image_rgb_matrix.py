from PIL import Image
import numpy as np

# Open the image file
img = Image.open("windmill.png")

# Convert the image to RGB mode
img = img.convert("RGB")

# Use numpy to get the array representation of the pixel data
rgb_matrix = np.array(img)

# Print the shape of the RGB matrix, which will be displayed as (height, width, 3), where 3 corresponds to the RGB channels
print(rgb_matrix.shape)

# If you want to view the RGB value of a specific pixel, you can directly access it through indexing
x = 10  # Example x-coordinate
y = 20  # Example y-coordinate
print(f"Pixel at ({x}, {y}) in numpy array: R={rgb_matrix[y, x, 0]}, G={rgb_matrix[y, x, 1]}, B={rgb_matrix[y, x, 2]}")
