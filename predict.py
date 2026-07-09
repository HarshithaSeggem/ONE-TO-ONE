# ==========================================
# Heartbeat Prediction Module
# ==========================================

import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ------------------------------------------
# Load Trained Model
# ------------------------------------------

MODEL_PATH = "rnn_model.keras"
ENCODER_PATH = "label_encoder.pkl"

model = load_model(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# ------------------------------------------
# Heartbeat Class Names
# ------------------------------------------

CLASS_NAMES = {
    0: "Normal Beat",
    1: "Supraventricular Beat",
    2: "Ventricular Beat",
    3: "Fusion Beat",
    4: "Unknown Beat"
}

# ------------------------------------------
# Prediction Function
# ------------------------------------------

def predict_heartbeat(ecg_signal):

    # Convert input to numpy array
    ecg_signal = np.asarray(ecg_signal, dtype=np.float32)

    # Flatten in case input is nested
    ecg_signal = ecg_signal.flatten()

    # Validate length
    if len(ecg_signal) != 187:
        raise ValueError(
            f"Expected 187 ECG values, but got {len(ecg_signal)}"
        )

    # Reshape for SimpleRNN
    ecg_signal = ecg_signal.reshape(1, 187, 1)

    # Predict
    prediction = model.predict(ecg_signal, verbose=0)

    probabilities = prediction[0]

    predicted_index = int(np.argmax(probabilities))

    confidence = float(probabilities[predicted_index] * 100)

    # Decode original label
    original_label = int(
        label_encoder.inverse_transform([predicted_index])[0]
    )

    predicted_class = CLASS_NAMES.get(
        original_label,
        f"Class {original_label}"
    )

    return predicted_class, confidence, probabilities


# ------------------------------------------
# Test the Module
# ------------------------------------------

if __name__ == "__main__":

    sample = np.random.rand(187)

    predicted_class, confidence, probabilities = predict_heartbeat(sample)

    print("\nPrediction Result")
    print("-----------------------------")
    print("Predicted Class :", predicted_class)
    print("Confidence      :", round(confidence, 2), "%")

    print("\nClass Probabilities")

    for i, probability in enumerate(probabilities):
        print(
            f"{CLASS_NAMES[i]:30s} : {probability * 100:.2f}%"
        )