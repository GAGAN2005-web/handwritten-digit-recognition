import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Load MNIST Dataset
# ==========================================

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Original training shape:", x_train.shape)
print("Original testing shape:", x_test.shape)


# ==========================================
# 2. Normalize Pixel Values
# ==========================================

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# ==========================================
# 3. Flatten Images
# ==========================================

x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

print("After preprocessing:")
print("Training shape:", x_train.shape)
print("Testing shape:", x_test.shape)


# ==========================================
# 4. Build MLP Model
# ==========================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),

    tf.keras.layers.Dense(128, activation="relu"),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(64, activation="relu"),

    tf.keras.layers.Dense(10, activation="softmax")
])


# ==========================================
# 5. Compile Model
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# 6. Display Model Summary
# ==========================================

model.summary()


# ==========================================
# 7. Train Model
# ==========================================

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)


# ==========================================
# 8. Evaluate Model
# ==========================================

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy * 100, "%")


# ==========================================
# 9. Save Model
# ==========================================

model.save("models/mnist_mlp.keras")

print("\nModel saved successfully!")

# ==========================================
# 10. Plot Training and Validation Accuracy
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.show()


# ==========================================
# 11. Plot Training and Validation Loss
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.show()