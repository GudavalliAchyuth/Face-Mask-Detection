# scripts/train_mask_detector.py

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Initialize hyperparameters
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

DIRECTORY = os.path.join(os.path.dirname(__file__), "..", "dataset")
CATEGORIES = ["with_mask", "without_mask"]

print("[INFO] Loading images...")

data = []
labels = []

# 2. Loop over the image categories and preprocess the data
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    if not os.path.exists(path):
        print(f"[WARNING] Path missing: {path}. Please add images here.")
        continue
        
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        try:
            # Load image and resize to 224x224 (required size for MobileNetV2)
            image = load_img(img_path, target_size=(224, 224))
            image = img_to_array(image)
            image = preprocess_input(image) # Scales pixels between -1 and 1

            data.append(image)
            labels.append(category)
        except Exception as e:
            # Skip corrupted or non-image files smoothly
            continue

# Convert data and labels to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Partition data into 80% training and 20% testing splits
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# Construct the training image generator for data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# 3. Load the MobileNetV2 network, ensuring the head FC layer sets are left off
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Construct the head of the model that will be placed on top of the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# Place the head FC model on top of the base model (this becomes the actual model we train)
model = Model(inputs=baseModel.input, outputs=headModel)

# Loop over all layers in the base model and freeze them so they won't be updated during the training process
for layer in baseModel.layers:
    layer.trainable = False

# 4. Compile and Train the Model
print("[INFO] Compiling model...")
opt = Adam(learning_rate=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] Training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS
)

# 5. Evaluate the network
print("[INFO] Evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# For each image in the testing set, find the index of the label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# Show a beautifully formatted classification report (Precision, Recall, F1-score)
# TO THIS (fixed keyword argument):
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

# Serialize and save the model to disk
print("[INFO] Saving mask detector model...")
model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(model_dir, exist_ok=True)
model.save(os.path.join(model_dir, "mask_detector.h5"))

print("[INFO] Done! Model saved successfully.")