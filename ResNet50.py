import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# Parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
DATASET_PATH = r"D:\Eye Scan\DATASET_101"  # Update this path
# Data Generator
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)


# Build Model
base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
    ModelCheckpoint("final_fundus_model.h5", save_best_only=True, monitor='val_accuracy')
]

# Train
model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=callbacks)
model.save("final_fundus_model.h5")
print("✅ Model training complete and saved.")
# Train
model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=callbacks)
model.save("final_fundus_model.h5")
print("✅ Model training complete and saved.")
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("final_fundus_model_finetuned.keras")

# Display model summary
model.summary()

# Check output layer info
output_layer = model.layers[-1]
print("Output layer activation:", output_layer.activation.__name__)
print("Number of output units (classes):", output_layer.output_shape[-1])
