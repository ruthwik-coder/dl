# 4 Write a program in deep learning to apply image processing operations such as Histogram equalization, Thresholding, Edge detection, Data augmentation, Morphological Operations. CO1
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('image.jpg')
if img is None:
    # Creating a dummy image if 'image.jpg' doesn't exist for demonstration
    img = (np.random.rand(400, 400, 3) * 255).astype(np.uint8)

# Convert to RGB for Matplotlib and Gray for processing
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- 1. Histogram Equalization ---
# Increases contrast by distributing intensity
equalized_img = cv2.equalizeHist(img_gray)

# --- 2. Thresholding ---
# Converts image to binary (black and white)
_, thresh_img = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

# --- 3. Edge Detection (Canny) ---
edges = cv2.Canny(img_gray, 100, 200)

# --- 4. Morphological Operations ---
# Dilation (adds pixels to boundaries) and Erosion (removes pixels)
kernel = np.ones((5, 5), np.uint8)
dilation = cv2.dilate(thresh_img, kernel, iterations=1)
erosion = cv2.erode(thresh_img, kernel, iterations=1)

# --- 5. Data Augmentation ---
# Common DL augmentations: Flip and Rotation
flipped = cv2.flip(img_rgb, 1)  # Horizontal flip
rows, cols = img_gray.shape
M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1) # 45-degree rotation
rotated = cv2.warpAffine(img_rgb, M, (cols, rows))

# --- Visualization ---
titles = [
    'Original', 'Equalized', 'Thresholding',
    'Edges', 'Dilation', 'Erosion',
    'Flipped', 'Rotated'
]
images = [
    img_rgb, equalized_img, thresh_img,
    edges, dilation, erosion,
    flipped, rotated
]

plt.figure(figsize=(16, 10))
for i in range(len(images)):
    plt.subplot(2, 4, i+1)
    # Display gray images in gray cmap, others in RGB
    if len(images[i].shape) == 2:
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
