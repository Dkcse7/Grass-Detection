import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler

# Read the CSV file
csv_file_path = 'C:\\Users\\DIGVIJAY\\OneDrive\\Desktop\\grass\\training\\training\\Grass.csv'
data_df = pd.read_csv(csv_file_path)

# Split the 'Number;Grass' column into separate columns for filenames and labels
data_df[['Filename', 'Label']] = data_df['Number;Grass'].str.split(';', expand=True)

# Prepare data
image_filenames = data_df['Filename'].tolist()
labels = data_df['Label'].astype(int).tolist()  # Convert labels to integers

# Convert 'Label' column to integers
data_df['Label'] = data_df['Label'].astype(int)

# Define paths to your images directory
image_dir = 'C:\\Users\\DIGVIJAY\\OneDrive\\Desktop\\grass\\training\\training\\image'

# Preprocess and augment data
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,  # Increased rotation range
    width_shift_range=0.4,  # Increased width shift range
    height_shift_range=0.4,  # Increased height shift range
    shear_range=0.4,  # Increased shear range
    zoom_range=0.4,  # Increased zoom range
    horizontal_flip=True
)

# Define image size and batch size
image_size = (224, 224)  # ResNet input size
batch_size = 20

# Load and preprocess images
generator = datagen.flow_from_dataframe(
    dataframe=data_df,
    directory=image_dir,
    x_col="Filename",
    y_col="Label",
    target_size=image_size,
    batch_size=batch_size,
    class_mode='raw' 
)

# Load pre-trained ResNet model
base_model = ResNet50(weights='imagenet', include_top=False)

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# Combine base model and custom head
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze layers in base model
for layer in base_model.layers:
    layer.trainable = False

# Define learning rate scheduler
def lr_scheduler(epoch):
    return 0.001 * np.exp(-0.1 * epoch)

lr_schedule = LearningRateScheduler(lr_scheduler)

# Define the learning rate
learning_rate = 0.01

# Create the Adam optimizer with the specified learning rate
optimizer = Adam(learning_rate=learning_rate)

# Compile the model
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model   
history = model.fit(
    generator,
    steps_per_epoch=len(data_df) // batch_size,
    epochs=50,
    callbacks=[lr_schedule]  # Apply learning rate scheduler
)

# Evaluate the model
test_loss, test_acc = model.evaluate(generator, verbose=2)
print('\nTest accuracy:', test_acc)
