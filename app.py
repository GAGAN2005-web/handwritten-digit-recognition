import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ==========================================
# Load Model
# ==========================================

model = tf.keras.models.load_model(
    "models/mnist_mlp.keras"
)


# ==========================================
# Load MNIST Test Data
# ==========================================

(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered"
)


# ==========================================
# Custom Styling
# ==========================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: gray;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .prediction-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        background-color: #f0f2f6;
        margin-top: 20px;
    }

    .digit {
        font-size: 70px;
        font-weight: bold;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# Title
# ==========================================

st.markdown(
    '<div class="main-title">🔢 Handwritten Digit Recognition</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Deep Learning project using an MLP trained on MNIST'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================
# Model Information
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset", "MNIST")

with col2:
    st.metric("Model", "MLP")

with col3:
    st.metric("Accuracy", "97.71%")


st.divider()


# ==========================================
# Upload Image
# ==========================================

st.subheader("📤 Upload a Digit")

uploaded_file = st.file_uploader(
    "Upload a handwritten digit image",
    type=["png", "jpg", "jpeg"]
)


# ==========================================
# Prediction Function
# ==========================================

def predict_digit(image):

    # Convert to grayscale
    image = image.convert("L")

    # Resize to MNIST format
    image = image.resize((28, 28))

    # Convert to NumPy array
    image_array = np.array(image)

    # Invert if necessary
    if image_array.mean() > 127:
        image_array = 255 - image_array

    # Normalize
    image_array = image_array.astype("float32") / 255.0

    # Flatten
    image_array = image_array.reshape(1, 784)

    # Prediction
    prediction = model.predict(
        image_array,
        verbose=0
    )

    predicted_digit = int(np.argmax(prediction))
    probabilities = prediction[0]
    confidence = float(np.max(probabilities) * 100)

    return predicted_digit, confidence, probabilities


# ==========================================
# Uploaded Image Prediction
# ==========================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("🖼️ Uploaded Image")

    st.image(
        image,
        width=250
    )

    predicted_digit, confidence, probabilities = predict_digit(image)

    st.markdown(
        f"""
        <div class="prediction-box">

        <div>Predicted Digit</div>

        <div class="digit">
        {predicted_digit}
        </div>

        <div>
        Confidence: <b>{confidence:.2f}%</b>
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # ==========================================
    # Probability Chart
    # ==========================================

    st.subheader("📊 Prediction Probabilities")

    chart_data = {
        "Digit": list(range(10)),
        "Probability": probabilities * 100
    }

    st.bar_chart(
        chart_data,
        x="Digit",
        y="Probability"
    )


# ==========================================
# Built-in Model Test
# ==========================================

st.divider()

st.subheader("🧪 Test with MNIST Sample")

if st.button("Generate Test Prediction"):

    index = np.random.randint(0, len(x_test))

    test_image = x_test[index]

    actual_digit = int(y_test[index])

    test_input = (
        test_image.astype("float32") / 255.0
    )

    test_input = test_input.reshape(1, 784)

    prediction = model.predict(
        test_input,
        verbose=0
    )

    predicted_digit = int(
        np.argmax(prediction)
    )

    confidence = float(
        np.max(prediction) * 100
    )

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            test_image,
            width=200,
            caption=f"Actual Digit: {actual_digit}"
        )

    with col2:

        st.metric(
            "Predicted Digit",
            predicted_digit
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        if predicted_digit == actual_digit:

            st.success("✅ Correct Prediction!")

        else:

            st.error("❌ Incorrect Prediction")