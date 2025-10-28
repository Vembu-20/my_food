
# Different optimizers

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad
import time

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)
    return (x_train, y_train_cat), (x_test, y_test_cat), y_test

def build_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(10, activation="softmax")
    ])
    return model

# Train models with different optimizers
def train_and_evaluate(optimizers, x_train, y_train, x_test, y_test):
    history_dict = {}
    results = {}
    for name, optimizer in optimizers.items():
        print(f"\nTraining with {name}...")
        model = build_model()
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        start = time.time()
        history = model.fit(x_train, y_train,
                            epochs=10,
                            batch_size=128,
                            validation_split=0.1,
                            verbose=0)
        end = time.time()
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

        results[name] = {
            "model": model,
            "Train Accuracy": history.history['accuracy'][-1],
            "Val Accuracy": history.history['val_accuracy'][-1],
            "Train Loss": history.history['loss'][-1],
            "Val Loss": history.history['val_loss'][-1],
            "Test Accuracy": test_acc,
            "Training Time": round(end - start, 2)
        }
        history_dict[name] = history

    return results, history_dict

def plot_metrics(history_dict):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    for name, hist in history_dict.items():
        plt.plot(hist.history['val_accuracy'], label=f'{name} (val)')
        plt.plot(hist.history['loss'], linestyle='--', label=f'{name} (train)')
    plt.title("Loss vs Epoch")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    for name, hist in history_dict.items():
        plt.plot(hist.history['val_accuracy'], label=f'{name} (val)')
        plt.plot(hist.history['accuracy'], linestyle='--', label=f'{name} (train)')
    plt.title("Accuracy vs Epoch")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.show()

def display_predictions (model, x_test, y_test_labels):
    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis = 1)
    plt.figure(figsize=(10, 10))

    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(x_test[i], cmap='gray')
        pred = predicted_classes[i]
        actual = y_test_labels[i]
        color = 'green' if pred == actual else 'red'
        symbol = "correct" if pred == actual else "incorrect"
        plt.title(f"P: {pred} / A:{actual}\n{symbol}", color=color, fontsize=10)
        plt.axis('off')

    plt.suptitle("Prediction vs Actual (First 25 Test Images)", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test), y_test_labels = load_data()

    optimizers = {
        "SGD": SGD(),
        "Adam": Adam(),
        "RMSprop": RMSprop(),
        "Adagrad": Adagrad()
    }
    results, history_dict = train_and_evaluate(optimizers, X_train, y_train, X_test, y_test)

    summary = pd.DataFrame([
        {"optimizer": name, **{k: v for k, v in res.items() if k != "model"}}
        for name, res in results.items()
    ])
    summary = summary.sort_values(by="Val Accuracy", ascending=False)

    print("\nOptimizer Comparison Table:")
    print(summary.to_string(index=False))

    best_name = summary.iloc[0]["optimizer"]
    print(f"\nBest Optimizer: {best_name}")
    print(f"   -> Validation Accuracy: {summary.iloc[0]['Val Accuracy']:.4f}")
    print(f"   -> Test Accuracy: {summary.iloc[0]['Test Accuracy']:.4f}")
    print(f"   -> Training Time: {summary.iloc[0]['Training Time']:.2f} seconds")
    plot_metrics(history_dict)
    best_model = results[best_name]["model"]
    display_predictions(best_model, X_test, y_test_labels)