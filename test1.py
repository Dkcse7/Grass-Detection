import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Read the Excel file
excel_file_path = 'C:\\Users\\DIGVIJAY\\OneDrive\\Desktop\\grass\\training\\training\\Grass1.xlsx'
data_df = pd.read_excel(excel_file_path)

# Define paths to your training and testing images directories
train_dir = 'C:\\Users\\DIGVIJAY\\OneDrive\\Desktop\\grass\\training\\training\\image'
test_dir = 'C:\\Users\\DIGVIJAY\\OneDrive\\Desktop\\grass\\test'
data_df['Label'] = data_df['Label'].astype(str)

# Data Augmentation and Preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2 # Reserve 20% of data for validation
)

# Load and preprocess images
train_generator = datagen.flow_from_dataframe(
    dataframe=data_df,
    directory=train_dir,
    x_col="Number",
    y_col="Label",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',  # Use 'binary' class mode for binary classification
    subset='training'
)

valid_generator = datagen.flow_from_dataframe(
    dataframe=data_df,
    directory=train_dir,
    x_col="Number",
    y_col="Label",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
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

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,
    validation_data=valid_generator,
    validation_steps=len(valid_generator)
)

# Evaluate the model
test_loss, test_acc = model.evaluate(valid_generator, verbose=2)
print('\nValidation accuracy:', test_acc)
