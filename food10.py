'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def generate_real_samples(batch_size, image_size):
    """Generates real shapes squares or circles with size 28x28"""
    shapes = np.zeros((batch_size, image_size * image_size))  # Placeholder for shapes
    for i in range(batch_size):
        shape_type = np.random.choice(['circle', 'square'])
        shape = np.zeros((image_size, image_size))  # 28x28 block image

        if shape_type == 'square':
            size = np.random.randint(5, 15)  # Random square size #10,20
            start = np.random.randint(5, image_size - size - 5)
            shape[start:start + size, start:start + size] = 1  # white square

        else:
            radius = np.random.randint(5, 10)
            center = np.random.randint(10, image_size - 10, size=2)
            for y in range(image_size):
                for x in range(image_size):
                    if (x - center[0]) ** 2 + (y - center[1]) ** 2 < radius ** 2:
                        shape[y, x] = 1  # white circle

        shapes[i] = shape.flatten()  # Flatten the 28x28 image to 784 pixels
    return shapes


def plot_samples(samples, title="Synthetic Shapes"):
    plt.figure(figsize=(8, 8))
    for i, sample in enumerate(samples[:16]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(sample.reshape(28, 28), cmap="gray")
        plt.axis('off')
    plt.suptitle(title)
    plt.show()


def train_gan(generator, discriminator, image_dim, latent_dim, batch_size=64, epochs=10000, lr=0.0002):
    # Loss and optimizers
    criterion = nn.BCELoss()
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_g = optim.Adam(generator.parameters(), lr=lr)

    for epoch in range(epochs):
        # Train Discriminator
        real_samples = generate_real_samples(batch_size, image_dim)
        real_samples = torch.tensor(real_samples, dtype=torch.float32)
        real_labels = torch.ones(batch_size, 1)

        latent_space_samples = torch.randn(batch_size, latent_dim)
        fake_samples = generator(latent_space_samples)
        fake_labels = torch.zeros(batch_size, 1)

        discriminator_real_loss = criterion(discriminator(real_samples), real_labels)
        discriminator_fake_loss = criterion(discriminator(fake_samples.detach()), fake_labels)
        discriminator_loss = discriminator_real_loss + discriminator_fake_loss

        optimizer_d.zero_grad()
        discriminator_loss.backward()
        optimizer_d.step()

        # Train Generator
        fake_samples = generator(latent_space_samples)
        generator_loss = criterion(discriminator(fake_samples), real_labels)

        optimizer_g.zero_grad()
        generator_loss.backward()
        optimizer_g.step()

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}/{epochs} | D Loss: {discriminator_loss.item()} | G Loss: {generator_loss.item()}")
            plot_samples(fake_samples.detach().numpy())


class Generator(nn.Module):
    def __init__(self, latent_dim, image_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, image_dim),
            nn.Sigmoid()  # Output values between 0 and 1
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, image_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(image_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output a probability between 0 and 1
        )

    def forward(self, x):
        return self.model(x)


image_size = 28
image_dim = image_size * image_size
latent_dim = 100

generator = Generator(latent_dim, image_dim)
discriminator = Discriminator(image_dim)

train_gan(generator, discriminator, image_size, latent_dim, epochs=10000, batch_size=64)
'''