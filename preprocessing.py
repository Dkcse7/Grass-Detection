import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Read the CSV file
csv_file_path = 'C:\\Users\\DIGVIJAY\\OneDrive\\Desktop\\grass\\training\\training\\Grass.csv'
data_df = pd.read_csv(csv_file_path)

# Split the 'Number;Grass' column into separate columns for filenames and labels
data_df[['Filename', 'Label']] = data_df['Number;Grass'].str.split(';', expand=True)

# Define paths to your images directory
image_dir = 'C:\\Users\\DIGVIJAY\\OneDrive\\Desktop\\grass\\training\\training\\image'

# Create ImageDataGenerator for preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,  # Scale pixel values to [0, 1]
    rotation_range=20,  # Randomly rotate images by up to 20 degrees
    width_shift_range=0.2,  # Randomly shift images horizontally by up to 20%
    height_shift_range=0.2,  # Randomly shift images vertically by up to 20%
    shear_range=0.2,  # Randomly apply shear transformation
    zoom_range=0.2,  # Randomly zoom in/out on images by up to 20%
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode='nearest'  # Fill in missing pixels using the nearest available value
)

# Flow images from directory and preprocess them
generator = datagen.flow_from_dataframe(
    dataframe=data_df,
    directory=image_dir,
    x_col='Filename',
    y_col='Label',
    target_size=(150, 150),  # Resize images to 150x150 pixels
    batch_size=32,
    class_mode='raw',  # Return raw numerical labels
    shuffle=True  # Shuffle the order of images
)

# Extract features from preprocessed images
X_features = []
y_labels = []

for i in range(len(generator)):
    batch_images, batch_labels = generator[i]
    features = batch_images  # You can replace this with feature extraction from a pre-trained model
    X_features.append(features)
    y_labels.append(batch_labels)

# Convert lists to numpy arrays
X_features = np.concatenate(X_features)
y_labels = np.concatenate(y_labels)

# Print shape of feature matrix and label vector
print('Feature matrix shape:', X_features.shape)
print('Label vector shape:', y_labels.shape)
