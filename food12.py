'''
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import imageio
import glob
import os
from skimage.draw import polygon  # <-- ADDED THIS IMPORT

# --- 1. Constants and Hyperparameters ---
IMG_SHAPE = (64, 64, 1)  # Image shape (height, width, channels)
NOISE_DIM = 100          # Dimension of the noise vector for the generator
BUFFER_SIZE = 1200       # Buffer size for shuffling the dataset
BATCH_SIZE = 128         # Batch size for training
EPOCHS = 100             # Number of training epochs
NUM_IMAGES = 1200        # Number of synthetic shape images to generate

# --- 2. Data Generation: Create Synthetic Shapes ---
def generate_shapes(num_images):
    """
    Generates a dataset of simple geometric shapes (squares, circles, triangles).
    """
    images = np.zeros((num_images, IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2]), dtype=np.float32)
    for i in range(num_images):
        # Choose a random shape
        shape_type = np.random.choice(['square', 'circle', 'triangle'])
        img = np.zeros((IMG_SHAPE[0], IMG_SHAPE[1]), dtype=np.float32)

        # Random size and position
        size = np.random.randint(15, 35)
        x = np.random.randint(5, IMG_SHAPE[0] - size - 5)
        y = np.random.randint(5, IMG_SHAPE[1] - size - 5)

        if shape_type == 'square':
            img[x:x+size, y:y+size] = 1.0

        elif shape_type == 'circle':
            center_x, center_y = x + size // 2, y + size // 2
            radius = size // 2
            Y, X = np.ogrid[:IMG_SHAPE[0], :IMG_SHAPE[1]]
            dist_from_center = np.sqrt((X - center_y) ** 2 + (Y - center_x) ** 2)
            mask = dist_from_center <= radius
            img[mask] = 1.0

        elif shape_type == 'triangle':
            p1 = (x, y + size // 2)
            p2 = (x + size, y)
            p3 = (x + size, y + size)
            rr, cc = polygon([p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]], img.shape)
            img[rr, cc] = 1.0

        images[i] = np.expand_dims(img, axis=-1)

    return images

# Generate and prepare the dataset
print("Generating synthetic shape data...")
real_images = generate_shapes(NUM_IMAGES)
# Normalize the images to [-1, 1] as it is standard for GANs with tanh activation
real_images = (real_images - 0.5) * 2.0
train_dataset = tf.data.Dataset.from_tensor_slices(real_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
print("Data generation complete.")

# --- 3. Build the Generator Model ---
def make_generator_model():
    model = tf.keras.Sequential()
    # Start with a dense layer that projects the noise vector
    model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(NOISE_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 256)))
    assert model.output_shape == (None, 8, 8, 256)

    # Upsample to 16x16
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Upsample to 32x32
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Upsample to 64x64, producing the final image
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2])

    return model

# --- 4. Build the Discriminator Model ---
def make_discriminator_model():
    model = tf.keras.Sequential()
    # Downsample from 64x64
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=IMG_SHAPE))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # Downsample to 16x16
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # Downsample to 8x8
    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # Flatten and produce a single output for classification (real or fake)
    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# Create instances of the generator and discriminator
generator = make_generator_model()
discriminator = make_discriminator_model()

# --- 5. Define Loss Functions and Optimizers ---
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# --- 6. The Training Loop ---
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

# --- 7. Training and Visualization ---
def train(dataset, epochs):
    if not os.path.exists('gan_generated_images'):
        os.makedirs('gan_generated_images')

    seed = tf.random.normal([16, NOISE_DIM])

    for epoch in range(epochs):
        gen_loss_epoch = []
        disc_loss_epoch = []

        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
            gen_loss_epoch.append(gen_loss)
            disc_loss_epoch.append(disc_loss)

        if (epoch + 1) % 5 == 0:
            generate_and_save_images(generator, epoch + 1, seed)

        print(f'Epoch {epoch + 1}, Gen Loss: {np.mean(gen_loss_epoch):.4f}, Disc Loss: {np.mean(disc_loss_epoch):.4f}')

    generate_and_save_images(generator, epochs, seed)

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 0.5 + 0.5, cmap='gray')
        plt.axis('off')

    plt.savefig(f'gan_generated_images/image_at_epoch_{epoch:04d}.png')
    plt.show()

# Start the training process
print("\nStarting GAN training...")
train(train_dataset, EPOCHS)
print("Training complete.")

# --- 8. Create a GIF of the Training Process ---
print("\nCreating training progress GIF...")
anim_file = 'dcgan_shapes.gif'
with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('gan_generated_images/image*.png')
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
print(f"GIF saved as {anim_file}")'''