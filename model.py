import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load image pairs (SAR and optical) and preprocess
def load_image_pairs(sar_image_path, optical_image_path, image_size=(256, 256)):
    sar_image = cv2.imread(sar_image_path, cv2.IMREAD_GRAYSCALE)
    sar_image = cv2.resize(sar_image, image_size) / 255.0  # Normalize to [0, 1]
    sar_image = np.expand_dims(sar_image, axis=-1)  # Add channel dimension

    optical_image = cv2.imread(optical_image_path, cv2.IMREAD_COLOR)
    optical_image = cv2.resize(optical_image, image_size) / 255.0  # Normalize to [0, 1]

    return sar_image, optical_image

# Load dataset
def load_dataset(sar_dir, optical_dir):
    sar_images = []
    optical_images = []
    for file_name in os.listdir(sar_dir):
        sar_path = os.path.join(sar_dir, file_name)
        optical_path = os.path.join(optical_dir, file_name)
        if os.path.exists(optical_path):
            sar_img, optical_img = load_image_pairs(sar_path, optical_path)
            sar_images.append(sar_img)
            optical_images.append(optical_img)

    return np.array(sar_images), np.array(optical_images)

# Load the training and testing datasets
train_sar, train_optical = load_dataset('data/train/sar_images', 'data/train/optical_images')
test_sar, test_optical = load_dataset('data/test/sar_images', 'data/test/optical_images')

# Build the U-Net model for SAR-to-optical image translation
def build_unet(input_shape=(256, 256, 1)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)

    # Decoder
    u5 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    outputs = layers.Conv2D(3, (1, 1), activation='sigmoid')(c7)  # Output RGB image

    model = models.Model(inputs, outputs)
    return model

model = build_unet()
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Training the model
history = model.fit(
    train_sar, train_optical,
    validation_split=0.1,
    epochs=50,
    batch_size=16,
    shuffle=True
)

# Function to plot predictions
def plot_sample_predictions(model, test_sar, test_optical, num_samples=5):
    predictions = model.predict(test_sar[:num_samples])
    for i in range(num_samples):
        plt.figure(figsize=(15, 5))

        # Original SAR Image
        plt.subplot(1, 3, 1)
        plt.imshow(test_sar[i].squeeze(), cmap='gray')
        plt.title("SAR Image")

        # Predicted Colorized Image
        plt.subplot(1, 3, 2)
        plt.imshow(predictions[i])
        plt.title("Colorized Image")

        # Ground Truth Optical Image
        plt.subplot(1, 3, 3)
        plt.imshow(test_optical[i])
        plt.title("Optical Image")

        plt.show()

plot_sample_predictions(model, test_sar, test_optical)

# Save the model
model.save('sar_colorization_model.h5')
