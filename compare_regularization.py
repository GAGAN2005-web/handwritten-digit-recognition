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
# 2. Model WITHOUT Batch Normalization
# ==========================================

def create_model_without_bn():

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
# 3. Model WITH Batch Normalization
# ==========================================

def create_model_with_bn():

    model = tf.keras.Sequential([

        tf.keras.layers.Input(shape=(784,)),

        tf.keras.layers.Dense(128),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Activation("relu"),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(64),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Activation("relu"),

        tf.keras.layers.Dense(
            10,
            activation="softmax"
        )
    ])

    return model


# ==========================================
# 4. Train and Evaluate
# ==========================================

results = []


# ---- Without Batch Normalization ----

print("\nTraining model WITHOUT Batch Normalization")

model1 = create_model_without_bn()

model1.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model1.fit(
    x_train,
    y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

loss1, accuracy1 = model1.evaluate(
    x_test,
    y_test,
    verbose=0
)

results.append({
    "Model": "Without Batch Normalization",
    "Test Accuracy": accuracy1 * 100,
    "Test Loss": loss1
})


# ---- With Batch Normalization ----

print("\nTraining model WITH Batch Normalization")

model2 = create_model_with_bn()

model2.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model2.fit(
    x_train,
    y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

loss2, accuracy2 = model2.evaluate(
    x_test,
    y_test,
    verbose=0
)

results.append({
    "Model": "With Batch Normalization",
    "Test Accuracy": accuracy2 * 100,
    "Test Loss": loss2
})


# ==========================================
# 5. Display Results
# ==========================================

results_df = pd.DataFrame(results)

print("\nRegularization Comparison")
print(results_df.to_string(index=False))