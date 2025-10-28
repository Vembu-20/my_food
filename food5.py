'''
# Transfer Learning on Cats vs Dogs (fixed)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, math, zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

# 1) Download + extract dataset (use the returned path)
url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
filename = "cats_and_dogs_filtered.zip"
zip_path = tf.keras.utils.get_file(fname=filename, origin=url, extract=False)

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(path=os.path.dirname(zip_path))

base_dir = os.path.join(os.path.dirname(zip_path), "cats_and_dogs_filtered")
train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")

# 2) Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
validation_datagen = ImageDataGenerator(rescale=1./255)

# NOTE: train_generator must read from train_dir (was validation_dir by mistake)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode="binary",
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode="binary",
    shuffle=False
)

# 3) Pretrained VGG16 base
conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(150, 150, 3))
conv_base.trainable = False

# 4) Classifier head
model = tf.keras.models.Sequential([
    conv_base,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=2e-5),
    metrics=["accuracy"]
)

# Steps computed from dataset sizes (more robust)
# steps_per_epoch = math.ceil(train_generator.samples / train_generator.batch_size)
# validation_steps = math.ceil(validation_generator.samples / validation_generator.batch_size)

history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=50
)

# 5) Sample predictions on one validation batch
x, y_true = next(validation_generator)
y_pred = model.predict(x)
class_names = ['cat', 'dog']

for i in range(len(x)):
    plt.imshow(x[i])
    pred_label = class_names[int(round(float(y_pred[i][0])))]
    true_label = class_names[int(y_true[i])]
    plt.title(f'Pred: {pred_label} | True: {true_label}')
    plt.axis('off')
    plt.show()

# 6) Plot accuracy & loss
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(1, len(acc) + 1)

plt.figure()
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

plt.figure()
plt.plot(epochs, loss, "ro", label="Training loss")
plt.plot(epochs, val_loss, "r", label="Validation loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

print("Class indices:", train_generator.class_indices)