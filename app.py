import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Brainwave Emotion Detection App",
    page_icon="brain.png",  # <-- colorful brain icon (place this file in the same folder)
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------------- CUSTOM STYLING ----------------------
st.markdown("""
    <style>
    /* Global Background */
    body {
        background: linear-gradient(135deg, #e3f2fd, #ede7f6);
        font-family: 'Poppins', sans-serif;
        color: #2c3e50;
    }
    .main {
        background: white;
        border-radius: 25px;
        padding: 2.5rem;
        box-shadow: 0px 8px 25px rgba(0, 0, 0, 0.1);
        margin-top: 1.5rem;
        margin-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #1f3b73;
        text-align: center;
        font-weight: 700;
    }
   h1 {
    font-size: 2.2rem;
    color: white !important;
    -webkit-background-clip: unset !important;
    -webkit-text-fill-color: white !important;
}

    .stButton>button {
        border-radius: 12px;
        background: linear-gradient(90deg, #4c8bf5, #6a5acd);
        color: white;
        height: 3rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #6a5acd, #4c8bf5);
        transform: scale(1.05);
    }
    .emotion-box {
        background: linear-gradient(135deg, #e3f2fd, #c5cae9);
        padding: 1.2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        color: #1f3b73;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    footer {
        text-align: center;
        font-size: 0.9rem;
        color: #555;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------- SIDEBAR ----------------------
st.sidebar.title("💡 Emotion Insights")
st.sidebar.markdown("""
### 🧠 **Emotion Guide**
Each predicted state reflects your dominant mental and emotional activity derived from EEG signals.

- 😔 **Depressed** — Low energy, emotional imbalance, reduced engagement.  
- 😢 **Sad** — Reflective, withdrawn, moderate alpha wave dominance.  
- 😌 **Calm** — Stable, relaxed, balanced alpha & theta rhythms.  
- 😄 **Happy** — Positive arousal, elevated beta activity, strong engagement.  

🧩 The model decodes EEG wave patterns to identify these emotional states.
""")

# ---------------------- HEADER ----------------------
col1, col2 = st.columns([1, 8])
with col1:
    st.image("brain.png", width=90)  # Make sure brain.png is in the same folder as app.py
with col2:
    st.markdown("<h1 style='color: white; font-size: 2.2rem; font-weight: 700;'>Brainwave Emotion Detection App</h1>", unsafe_allow_html=True)

st.markdown("Upload a preprocessed EEG `.npy` file to detect your **dominant emotional state** using deep learning analysis of brainwave signals.")

# ---------------------- LOAD MODEL ----------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("final_model.keras")

with st.spinner("🔄 Loading AI model..."):
    model = load_model()
st.success("✅ Model loaded successfully!")

# ---------------------- FILE UPLOAD ----------------------
uploaded_file = st.file_uploader("📤 Upload your EEG `.npy` file", type=["npy"])

if uploaded_file is not None:
    try:
        data = np.load(uploaded_file, allow_pickle=True)
        st.write(f"📁 **Uploaded file shape:** {data.shape}")

        # --- Fix shape ---
        data = np.squeeze(data)
        if data.ndim == 2:
            data = np.expand_dims(data, axis=0)
        elif data.ndim == 3 and data.shape[0] != 1:
            data = data[0:1, :, :]

        if data.shape[-1] != 63 and data.shape[-2] == 63:
            data = np.transpose(data, (0, 2, 1))
        if data.shape[-1] < 63:
            pad_width = 63 - data.shape[-1]
            data = np.pad(data, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
        if data.shape[-1] > 63:
            data = data[:, :, :63]

        st.write(f"✅ **Processed EEG shape:** {data.shape}")

        # ---------------------- PREDICT ----------------------
        preds = model.predict(data)
        predicted_class = np.argmax(preds, axis=1)[0]
        emotions = ['depressed', 'sad', 'calm', 'happy']
        predicted_emotion = emotions[predicted_class]

        # ---------------------- RESULT BOX ----------------------
        st.markdown(f"""
        <div class="emotion-box">
             <b>Predicted Emotion:</b> {predicted_emotion.upper()}
        </div>
        """, unsafe_allow_html=True)

        # ---------------------- CONFIDENCE CHART ----------------------
        st.markdown("<h2 style='color:white;'>📊 Prediction Confidence</h2>")
        fig, ax = plt.subplots(figsize=(6, 3))
        bars = ax.bar(emotions, preds[0], color=['#e57373', '#64b5f6', '#81c784', '#fff176'])
        ax.set_ylabel("Confidence")
        ax.set_ylim(0, 1)
        ax.set_title("Emotion Probability Distribution")
        for bar, val in zip(bars, preds[0]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha='center', fontsize=9)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

else:
    st.info("⬆️ Please upload an EEG `.npy` file to continue.")

# ---------------------- FOOTER ----------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<footer>
📦 <b>Model Version:</b> v1.0 | 🎯 <b>Accuracy:</b> 85.12% | 🗓️ Updated: Oct 2025  
👩‍💻 Developed by <b>Gopika</b> — Brainwave Emotion Recognition using Deep Learning 
</footer>
""", unsafe_allow_html=True)
