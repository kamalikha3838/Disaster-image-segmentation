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

