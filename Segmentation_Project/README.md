This project demonstrates how deep learning can be used to segment disaster-affected regions from satellite images. The workflow includes:

Loading and preprocessing satellite images

Applying U-Net architecture for segmentation

Training, validating, and evaluating the model

Generating prediction masks

Comparing model output with ground truth

Below is a high-level preview of the project structure:

📁 Disaster Image Segmentation
│
├── data/
│   ├── images/        # Input satellite images
│   ├── masks/         # Annotation masks
│
├── models/
│   └── unet_model.h5  # Saved trained model
│
├── notebooks/
│   └── training.ipynb # Main model training notebook
│
└── README.md





import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import MeanIoU
# Parameters
IMG_HEIGHT = 128
IMG_WIDTH = 128
NUM_IMAGES = 100

# Arrays for images and masks
images = np.zeros((NUM_IMAGES, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
masks = np.zeros((NUM_IMAGES, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)

# Create random "flood regions" (circles)
for i in range(NUM_IMAGES):
    x, y = np.random.randint(32, 96, size=2)
    r = np.random.randint(10, 30)
    yy, xx = np.meshgrid(np.arange(IMG_HEIGHT), np.arange(IMG_WIDTH))
    circle = ((xx - x)**2 + (yy - y)**2) <= r**2

    # Random background
    images[i] = np.random.rand(IMG_HEIGHT, IMG_WIDTH, 3)

    # Mask
    masks[i, circle, 0] = 1.0

print("Images shape:", images.shape, "Masks shape:", masks.shape)
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(images[0])
plt.title("Synthetic Image")
plt.subplot(1,2,2)
plt.imshow(masks[0].squeeze(), cmap='gray')
plt.title("Mask")
plt.show()


