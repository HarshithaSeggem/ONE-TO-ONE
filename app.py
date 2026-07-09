import streamlit as st
import pandas as pd
import numpy as np
from predict import predict_heartbeat

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Heartbeat Classification using RNN",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>
.main{
    background-color:#F4F8FB;
}

.title{
    text-align:center;
    font-size:45px;
    font-weight:bold;
    color:#D32F2F;
}

.subtitle{
    text-align:center;
    font-size:18px;
    color:gray;
    margin-bottom:30px;
}

.result-card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.15);
}

.footer{
    text-align:center;
    color:gray;
    margin-top:30px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Load Dataset
# ----------------------------
try:
    test_data = pd.read_csv("mitbih_test.csv", header=None)
except Exception as e:
    st.error(f"Dataset not found: {e}")
    st.stop()

# ----------------------------
# Class Names
# ----------------------------
CLASS_NAMES = {
    0: "Normal Beat",
    1: "Supraventricular Beat",
    2: "Ventricular Beat",
    3: "Fusion Beat",
    4: "Unknown Beat"
}

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:

    st.image("https://img.icons8.com/color/96/heart-with-pulse.png", width=90)

    st.title("Heartbeat AI")

    st.markdown("---")

    st.write("### Model")

    st.success("Simple RNN")

    st.write("### Dataset")

    st.info("MIT-BIH Arrhythmia Dataset")

    st.write("### Input")

    st.write("ECG Signal (187 Features)")

    st.write("### Output")

    st.write("Heartbeat Class")

# ----------------------------
# Header
# ----------------------------
st.markdown("<div class='title'>❤️ Heartbeat Classification using RNN</div>",
unsafe_allow_html=True)

st.markdown("<div class='subtitle'>Artificial Intelligence based ECG Heartbeat Prediction</div>",
unsafe_allow_html=True)

st.markdown("---")

# ----------------------------
# Sample Selection
# ----------------------------
sample_index = st.slider(
    "Select ECG Sample",
    0,
    len(test_data)-1,
    0
)

sample = test_data.iloc[sample_index,:-1].values

actual_label = int(test_data.iloc[sample_index,-1])

actual_class = CLASS_NAMES[actual_label]

col1,col2 = st.columns([2,1])

with col1:

    st.write("### ECG Signal")

    st.line_chart(sample)

with col2:

    st.write("### Actual Class")

    st.info(actual_class)

    predict_button = st.button(
        "🔍 Predict Heartbeat",
        use_container_width=True
    )
    # ----------------------------
# Prediction
# ----------------------------

if predict_button:

    try:

        predicted_class, confidence, probabilities = predict_heartbeat(sample)

        st.markdown("---")

        col3, col4 = st.columns(2)

        with col3:

            st.markdown(
                """
                <div class="result-card">
                <h3>Prediction Result</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.success(f"Predicted Class : {predicted_class}")

            st.metric(
                label="Confidence",
                value=f"{confidence:.2f}%"
            )

        with col4:

            st.markdown(
                """
                <div class="result-card">
                <h3>Actual Result</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.info(actual_class)

            if predicted_class == actual_class:
                st.success("✅ Correct Prediction")
            else:
                st.error("❌ Incorrect Prediction")

        st.markdown("---")

        st.subheader("Prediction Probabilities")

        probability_df = pd.DataFrame(
            {
                "Heartbeat Class": list(CLASS_NAMES.values()),
                "Probability": probabilities
            }
        )

        st.bar_chart(
            probability_df.set_index("Heartbeat Class")
        )

        st.markdown("---")

        st.subheader("ECG Signal Values")

        st.dataframe(
            pd.DataFrame(
                sample,
                columns=["ECG Value"]
            ),
            use_container_width=True
        )

    except Exception as e:

        st.error(f"Prediction Error : {e}")

# ----------------------------
# Footer
# ----------------------------

st.markdown("---")

st.markdown(
    """
    <div class='footer'>
        ❤️ Heartbeat Classification using Simple RNN<br>
        Developed with Streamlit • TensorFlow • Python
    </div>
    """,
    unsafe_allow_html=True
)