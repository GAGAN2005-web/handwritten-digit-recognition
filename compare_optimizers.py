import tensorflow as tf
import pandas as pd

# ==========================================
# 1. Load MNIST Dataset
# ==========================================

(x_train, y_train), (x_test, y_test) = \
    tf.keras.datasets.mnist.load_data()

# Normalize
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten
x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)


# ==========================================
# 2. Function to Create MLP
# ==========================================

def create_model():

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(784,)),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(
            64,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            10,
            activation="softmax"
        )
    ])

    return model


# ==========================================
# 3. Define Optimizers
# ==========================================

optimizers = {
    "SGD": tf.keras.optimizers.SGD(
        learning_rate=0.01
    ),

    "RMSProp": tf.keras.optimizers.RMSprop(
        learning_rate=0.001
    ),

    "Adam": tf.keras.optimizers.Adam(
        learning_rate=0.001
    )
}


# ==========================================
# 4. Train and Compare
# ==========================================

results = []

for name, optimizer in optimizers.items():

    print("\n" + "=" * 50)
    print(f"Training with {name}")
    print("=" * 50)

    model = create_model()

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        x_train,
        y_train,
        epochs=3,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )

    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    results.append({
        "Optimizer": name,
        "Test Accuracy": test_accuracy * 100,
        "Test Loss": test_loss
    })

    print(
        f"{name} Test Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )


# ==========================================
# 5. Display Comparison
# ==========================================

results_df = pd.DataFrame(results)

print("\n\nOptimizer Comparison")
print(results_df.to_string(index=False))