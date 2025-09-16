# Import Libraries
# -------------------------------
import numpy as np
import tensorflow as tf
# noinspection PyUnresolvedReferences
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

# -------------------------------
# Loading the Dataset
# -------------------------------
input_path = "C:/IA_Projects/Pneumonia_Detection_Project/chest_xray/chest_xray/"
train_ds = tf.keras.utils.image_dataset_from_directory(
    input_path + 'train',
    color_mode='rgb',
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=123
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    input_path + 'test',
    color_mode='rgb',
    batch_size=32,
    image_size=(256, 256),
    shuffle=False,
    seed=123
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    input_path + 'val',
    color_mode='rgb',
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=123
)


# -------------------------------
# Normalize Images
# -------------------------------
def normalize(image, label):
    return tf.cast(image / 255.0, tf.float32), label


train_ds = train_ds.map(normalize)
val_ds = val_ds.map(normalize)
test_ds = test_ds.map(normalize)
# -------------------------------
# Data Augmentation
# -------------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)])
# -------------------------------
# Compute Class Weights
# -------------------------------
y_train = np.concatenate([y for x, y in train_ds], axis=0)
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weights_dict = {i: w for i, w in enumerate(class_weights)}
# ----------------------------------
# Transfer Learning -ResNet50
# ----------------------------------
base_resnet = ResNet50(weights="imagenet", include_top=False, input_shape=(256, 256, 3))
base_resnet.trainable = False

resnet_model = Sequential([
    base_resnet,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

resnet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_resnet = resnet_model.fit(
    train_ds, validation_data=val_ds, epochs=20, class_weight=class_weights_dict,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
    ]
)

# -------------------------------
# Fine-Tuning ResNet50
# -------------------------------
base_resnet.trainable = True
for layer in base_resnet.layers[:-30]:
    layer.trainable = False

resnet_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy'])
history_resnet_ft = resnet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weights_dict,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)])

# Sauvegarde du modèle
resnet_model.save('resnet50_pneumonia_model.h5')
print("Modèle sauvegardé avec succès!")

# -------------------------------
# Model Evaluation
# -------------------------------
test_ds = tf.keras.utils.image_dataset_from_directory(
    input_path + 'test',
    color_mode='rgb',
    batch_size=32,
    image_size=(256, 256),
    shuffle=False,
    seed=123
)
test_ds = test_ds.map(normalize)

y_pred = resnet_model.predict(test_ds)
y_pred_classes = (y_pred > 0.5).astype("int32")
y_true = np.concatenate([y for x, y in test_ds], axis=0)

print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=["Normal", "Pneumonia"]))