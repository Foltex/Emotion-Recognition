import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import cv2
import pandas as pd

st.set_page_config(
    page_title="Deep SER Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B4B4B;
        text-align: center;
        margin-bottom: 20px;
    }
    .result-box {
        
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        border: 2px solid #4CAF50;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("About Project")
    st.info(
        """
        **Deep SER** is a voice-based emotion detection application using Deep Learning (ResNet50).
        
        **How to use:**
        1. Prepare an audio file (Wav/Mp3).
        2. Upload it in the panel.
        3. Click the 'Analyze' button.
        """
    )
    st.markdown("---")
    st.subheader("Team Members")
    st.text("1. Reuben\n2. Gavriel\n3. Neil")
    st.markdown("---")
    st.caption("2025 Deep Learning Project")

st.markdown('<div class="main-header">Voice Emotion Recognition</div>', unsafe_allow_html=True)
st.write("<p style='text-align: center;'>Sentiment and emotion analysis from short audio recordings using artificial intelligence.</p>", unsafe_allow_html=True)
st.markdown("---")

@st.cache_resource
def load_resources():
    model = load_model('Resnet50_Model.h5')
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, le

try:
    model, le = load_resources()
except Exception as e:
    st.error(f"Failed to load model! Error: {e}")

def preprocess_audio(audio_file):
    y, sr = librosa.load(audio_file, duration=5.0)
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mels_db = librosa.power_to_db(mels, ref=np.max)
    img_resized = cv2.resize(mels_db, (128, 128))
    img_norm = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min())
    img_rgb = np.stack((img_norm,)*3, axis=-1)
    return np.expand_dims(img_rgb, axis=0)

col_upload, col_result = st.columns([1, 2])

with col_upload:
    st.subheader("1. Upload Audio")
    uploaded_file = st.file_uploader("Format: WAV / MP3", type=["wav", "mp3"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        analyze_btn = st.button("Analyze Now", use_container_width=True, type="primary")
    else:
        analyze_btn = False

with col_result:
    st.subheader("2. Analysis Result")
    
    if analyze_btn and uploaded_file is not None:
        with st.spinner('Analyzing audio...'):
            try:
                processed_data = preprocess_audio(uploaded_file)
                prediction = model.predict(processed_data)
                probs = prediction[0]
                
                df_probs = pd.DataFrame({"Emotion": le.classes_, "Percentage": probs})
                df_probs = df_probs.sort_values(by="Percentage", ascending=False)
                
                top_emotion = df_probs.iloc[0]['Emotion']
                top_conf = df_probs.iloc[0]['Percentage']

                st.markdown(f"""
                <div class="result-box">
                    <h4 style="margin:0; color:#555;">Detected Emotion:</h4>
                    <h1 style="font-size: 3.5rem; margin: 10px 0;">{top_emotion.upper()}</h1>
                    <h3 style="color:#4CAF50;">Confidence: {top_conf*100:.2f}%</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### Probability Details")
                
                for _, row in df_probs.iterrows():
                    col_label, col_bar, col_pct = st.columns([2, 6, 2])
                    with col_label:
                        st.write(f"**{row['Emotion'].title()}**")
                    with col_bar:
                        st.progress(float(row['Percentage']))
                    with col_pct:
                        st.write(f"{row['Percentage']*100:.1f}%")

            except Exception as e:
                st.error(f"Error: {e}")
    
    elif not analyze_btn and uploaded_file is not None:
        st.info("Click 'Analyze Now' to see the results.")
    else:
        st.warning("Please upload an audio file in the left panel.")