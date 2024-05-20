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


def imshow(img, shifted_img):
    # Display  (a) Original
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(' (a) Original')
    axes[0].axis('off')

    # Display  (b) Periodical
    axes[1].imshow(shifted_img, cmap='gray')
    axes[1].set_title(' (b) Periodical')
    axes[1].axis('off')

    plt.show()

def imsave(save_path, shifted_img):
    if os.path.exists(save_path):
        # Remove the existing file if it exists
        os.remove(save_path)
    # Normalize shifted_img to range [0, 255] and convert to uint8
    shifted_img = ((shifted_img - shifted_img.min()) / (shifted_img.max() - shifted_img.min()) * 255).astype(np.uint8)
    imageio.imwrite(save_path, shifted_img)

image_input = 'windmill.png'
k, l = 100, -50
# Load the image
x = imageio.imread(image_input)
# shifting the image in the direction (k, l)
shifted_x = imshift(x, k, l)

shifted_img_output = 'windmill-shift.png'
imsave(shifted_img_output, shifted_x)
imshow(x, shifted_x)
# shift it back in the direction ( k, l)
shifted_back_x = imshift(shifted_x, -k, -l)
# Check
result = np.array_equal(x, shifted_back_x)

print(f"shifted_back_x_result: { result }")
