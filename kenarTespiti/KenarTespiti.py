import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image in grayscale
img = cv2.imread('kare.png', cv2.IMREAD_GRAYSCALE)

# Check if the image loaded successfully
if img is None:
    print("Hata: Resim doğru şekilde yüklenemedi.")
else:
    # Define the horizontal derivative filter
    kernel = np.array([[-1, 1]])

    # Apply the filter using OpenCV's filter2D function
    horizontal_derivative = cv2.filter2D(img, -1, kernel)

    # Display the result with adjusted contrast
    plt.subplot(1, 2, 1)
    plt.title("Orijinal Resim")
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)

    plt.subplot(1, 2, 2)
    plt.title("Yatay Türev")
    plt.imshow(horizontal_derivative, cmap='gray', vmin=0, vmax=255)
    plt.show()
