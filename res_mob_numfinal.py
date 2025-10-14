import tensorflow as tf
import pandas as pd
import numpy as np
import os
import shap
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# ========= CONFIG =========
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 1
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
    df["low"] = df["label"].astype("category").cat.codes
    return df

def decode_img(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
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

# Compute class weights for imbalance
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(df["low"]), y=df["low"]
)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# ========= DATA AUGMENTATION =========
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.1),
])

# ========= HYBRID MODEL (RESNET & MOBILENET) =========
L2 = tf.keras.regularizers.l2(1e-4)

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

# Learnable fusion
fused = tf.keras.layers.Concatenate()([
    tf.keras.layers.Dense(512, activation="relu")(x1),
    tf.keras.layers.Dense(512, activation="relu")(x2)
])

x = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=L2)(fused)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

# ========= COMPILE =========
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6),
    tf.keras.callbacks.ModelCheckpoint("best_hybrid_model.keras", save_best_only=True)
]

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ========= TRAIN (Stage 1) =========
history = model.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS, callbacks=callbacks,
    class_weight=class_weights
)

# ========= FINE-TUNING (Stage 2) =========
resnet_base.trainable = True
mobilenet_base.trainable = True

# Freeze earlier layers to retain low-level features
for layer in resnet_base.layers[:-30]:
    layer.trainable = False
for layer in mobilenet_base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_ft = model.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS, callbacks=callbacks,
    class_weight=class_weights
)

# ========= EVALUATE =========
results = model.evaluate(val_ds, verbose=2)
print(f"\nFinal Validation Loss: {results[0]:.4f}, Validation Accuracy: {results[1]:.4f}\n")

# ========= METRICS =========
y_true, y_pred = [], []
for images, labels in val_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

print("\nClassification Report:\n", classification_report(y_true, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))

# model.summary()

# ---------- GRAD-CAM ----------

print("\nGenerating Grad-CAM visualizations...")

# Pick one image from validation dataset
for images, labels in val_ds.take(1):
    img = images[0].numpy()       # float32 in [0,1]
    true_label = labels[0].numpy()

# --- IMPORTANT: pick the tensor in the top-level graph that is the ResNet feature-map ---
# Use the input to the global average pool (this is (None,7,7,2048) in your model)
conv_tensor = model.get_layer("global_average_pooling2d").input   # <-- top-level tensor (7,7,2048)

# Build grad model that maps model.input -> (conv_tensor, model.output)
grad_model = tf.keras.Model(inputs=model.inputs, outputs=[conv_tensor, model.output])

# Debug: check shapes
print("grad_model outputs shapes:", [o.shape for o in grad_model.outputs])
print("full model output shape:", model.output_shape)
print("img dtype, min, max:", img.dtype, np.min(img), np.max(img))
print("img shape:", img.shape)

# Preprocess image for ResNet50V2
img_batch = np.expand_dims(img, axis=0)                    # (1,H,W,C)
img_batch = tf.keras.applications.resnet_v2.preprocess_input(img_batch)

# Compute gradients
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_batch)      # conv_outputs (1,7,7,2048), predictions (1,num_classes)
    print("conv_outputs.shape:", conv_outputs.shape, "predictions.shape:", predictions.shape)
    preds = predictions[0]
    pred_index = tf.argmax(preds)                          # tensor scalar

    # class score: scalar (for chosen class)
    class_channel = preds[pred_index]

# Convert pred_index to int (debug print)
pred_index_int = int(pred_index.numpy())
print("Predicted class index:", pred_index_int, "predicted probs:", preds.numpy())

# Gradients of class score wrt conv feature maps
grads = tape.gradient(class_channel, conv_outputs)         # shape (1,7,7,2048)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))       # shape (2048,)

# Build heatmap
conv_outputs = conv_outputs[0]                             # (7,7,2048)
heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)  # (7,7)
heatmap = tf.maximum(heatmap, 0)

heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-10)
heatmap = heatmap.numpy()

# maxh = tf.reduce_max(heatmap)
# if maxh == 0:
#     heatmap = tf.zeros_like(heatmap)
# else:
#     heatmap /= maxh
# heatmap = heatmap.numpy()
print("Heatmap stats:", heatmap.shape, "min/max:", heatmap.min(), heatmap.max())

# Resize + overlay
orig_uint8 = np.uint8(img * 255)
heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

superimposed = cv2.addWeighted(orig_uint8, 0.6, heatmap_colored, 0.4, 0)

# Show
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(orig_uint8)
plt.title(f"Original (true={true_label})")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
plt.title("Grad-CAM (ResNet branch)")
plt.axis("off")
plt.show()


# Resize heatmap to image size
heatmap = np.uint8(255 * heatmap)
jet = plt.get_cmap("jet")
jet_colors = jet(np.arange(256))[:, :3]
jet_heatmap = jet_colors[heatmap]
jet_heatmap = tf.image.resize(np.expand_dims(jet_heatmap, 0), (img.shape[0], img.shape[1]))[0]

# Superimpose heatmap on original image
superimposed_img = jet_heatmap * 0.4 + img
superimposed_img = np.uint8(255 * superimposed_img / np.max(superimposed_img))

plt.figure(figsize=(6, 6))
plt.imshow(superimposed_img)
plt.axis('off')
plt.title(f"Grad-CAM - Predicted class {pred_index}")
plt.show()


# ======== SHAP ===========
# Take one batch from validation dataset
for images, labels in val_ds.take(1):
    sample_images = images[:10].numpy()
    break

# Define explainer
explainer = shap.GradientExplainer(model, sample_images)

# Get SHAP values for predictions
shap_values = explainer.shap_values(sample_images)

# Visualize one explanation
shap.image_plot(shap_values, sample_images)


# ========= SAVE =========
model.save("hybrid_resnet_mobilenet_final_optimized.keras")
print("\n✅ Final optimized hybrid model saved successfully!\n")

# ========= PLOT LEARNING CURVES =========
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.plot(history_ft.history["val_accuracy"], label="Val Acc (FT)")
plt.legend(); plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.plot(history_ft.history["val_loss"], label="Val Loss (FT)")
plt.legend(); plt.title("Loss")
plt.show()
