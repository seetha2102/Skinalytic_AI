import tensorflow as tf 
import pandas as pd
import numpy as np
import os
from glob import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ========= CONFIG =========
IMG_SIZE = 299   # Increased from 224 → 299 for better detail
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

# ========= CLASS WEIGHTS =========
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df["low"]),
    y=train_df["low"]
)
class_weights = dict(enumerate(class_weights))
print("\nClass Weights:", class_weights)

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

resnet_base.trainable = False
mobilenet_base.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)

x1 = resnet_base(x, training=False)
x1 = tf.keras.layers.GlobalAveragePooling2D()(x1)

x2 = mobilenet_base(x, training=False)
x2 = tf.keras.layers.GlobalAveragePooling2D()(x2)

x = tf.keras.layers.Concatenate()([x1, x2])
x = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

# ========= COMPILE & TRAIN =========
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("best_hybrid_model.keras", save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, verbose=1)
]

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# ========= EVALUATION =========
results = model.evaluate(val_ds, verbose=2)
print(f"\nFinal Validation Loss: {results[0]:.4f}, Validation Accuracy: {results[1]:.4f}\n")

# ========= SAVE =========
model.save("hybrid_resnet_mobilenet_skin_classifier_final.keras")
print("\nHybrid model fine-tuned & saved ✅\n")
