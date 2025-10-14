import tensorflow as tf
import pandas as pd
import numpy as np
import os
from glob import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# ========= CONFIG =========
IMG_SIZE = 224   # you can change to 299 if using InceptionV3
BATCH_SIZE = 16
EPOCHS = 20
dataset_dir = "C:/Users/Seetha/Skinalytic-AI/skin_dataset_combined"

# ========= DATA PREP =========
def build_dataframe_from_folder(dataset_dir):
    image_paths = glob(os.path.join(dataset_dir, "*", "*"))
    valid_exts = [".jpg", ".jpeg", ".png", ".bmp"]
    data = []
    for path in image_paths:
        if os.path.splitext(path)[1].lower() not in valid_exts:
            continue
        label = os.path.basename(os.path.dirname(path))
        data.append({"image_path": path, "label": label})
    df = pd.DataFrame(data)
    df["low"] = df['label'].astype('category').cat.codes
    return df

def decode_img(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0   # normalize
    return img, label

def make_dataset(df, batch_size=BATCH_SIZE, shuffle=True):
    paths = df["image_path"].values
    labels = df["low"].values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(decode_img, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# Build dataframe
df = build_dataframe_from_folder(dataset_dir)
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["low"], random_state=42)

train_ds = make_dataset(train_df, shuffle=True)
val_ds = make_dataset(test_df, shuffle=False)
num_classes = df["low"].nunique()

# ========= DATA AUGMENTATION =========
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.1),
])

# ========= HYBRID MODEL =========
resnet_base = tf.keras.applications.ResNet50V2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
)
mobilenet_base = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
)

# Freeze most layers at start
for layer in resnet_base.layers[:-40]:
    layer.trainable = False
for layer in mobilenet_base.layers[:-40]:
    layer.trainable = False

# Input & augmentation
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)

# Extract features
x1 = resnet_base(x, training=False)
x1 = tf.keras.layers.GlobalAveragePooling2D()(x1)
x1 = tf.keras.layers.Dropout(0.3)(x1)

x2 = mobilenet_base(x, training=False)
x2 = tf.keras.layers.GlobalAveragePooling2D()(x2)
x2 = tf.keras.layers.Dropout(0.3)(x2)

# Learnable fusion (additive + dense)
x1_dense = tf.keras.layers.Dense(512, activation='relu')(x1)
x2_dense = tf.keras.layers.Dense(512, activation='relu')(x2)
x = tf.keras.layers.Add()([x1_dense, x2_dense])
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)

# Classifier
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

# ========= COMPILE =========
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6),
    tf.keras.callbacks.ModelCheckpoint("best_hybrid_model.keras", save_best_only=True)
]

model.compile(optimizer=optimizer,
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# ========= TRAIN (Stage 1: Frozen base) =========
history = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)

# ========= FINE-TUNE (Stage 2: Unfreeze deeper layers) =========
for layer in resnet_base.layers[-60:]:
    layer.trainable = True
for layer in mobilenet_base.layers[-60:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history_ft = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)