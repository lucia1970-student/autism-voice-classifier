import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from scipy.stats import variation
from evolved_model import EvolvedNN, winner, config
import neat
from collections import defaultdict
import pandas as pd
from io import StringIO

st.set_page_config(page_title="Autism Voice Classifier", layout="centered")
st.title("Autism Voice Classifier Dashboard")

# Custom HTML/CSS for the banner
custom_html = """
<div class="banner">
    <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRUhT5ANMhyQD_SqnxVATfMlLI4G2Aqpo5SruzQS0z-6g&usqp=CAE&s" alt="Banner Image">
</div>
<style>
    .banner {
        display: flex;
        justify-content: center;
        width: 208px;
        height: 160px;
        overflow: hidden;
    }
    .banner img {
        width: 100%;
        object-fit: cover;
    }
</style>
"""
# Display the custom HTML
st.components.v1.html(custom_html)

prediction_summary = []

# Load trained model
neat_net = neat.nn.FeedForwardNetwork.create(winner, config)
model = EvolvedNN(neat_net, winner)
model.load_state_dict(torch.load("evolved_neat_model.pth", map_location=torch.device('cpu')))
model.eval()

st.markdown("Upload a `.wav` file or manually adjust features to predict likelihood of Autism.")

# ========== AUDIO FILE UPLOAD ==========
audio_file = st.file_uploader("Upload a .wav file", type=["wav"])

def extract_features_from_audio(file):
    y, sr = librosa.load(file, sr=None)

    # 1. f0 estimation using librosa.yin
    f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
    f0_mean = np.mean(f0)

    # 2. Jitter = relative variation in f0
    jitter = variation(f0)  # standard deviation / mean

    # 3. Shimmer = variation in amplitude envelope
    amplitude_env = np.abs(y)
    shimmer = variation(amplitude_env)

    # 4. HNR (approximate): harmonic-to-noise ratio via energy stats
    signal_energy = np.sum(y ** 2)
    noise_energy = np.sum((y - np.mean(y)) ** 2)
    hnr = 10 * np.log10(signal_energy / (noise_energy + 1e-6))
    hnr = np.clip(hnr, 0, 100)  # avoid wild spikes

    return [f0_mean, jitter, shimmer, hnr]

features = None
if audio_file:
    try:
        features = extract_features_from_audio(audio_file)
        st.success("Audio processed. Features extracted and ready for prediction.")
        st.write(f"**Extracted Features:**\n\nf0 = {features[0]:.3f}, jitter = {features[1]:.4f}, shimmer = {features[2]:.4f}, HNR = {features[3]:.2f}")
    except Exception as e:
        st.error(f"Error processing audio: {e}")

# ========== MANUAL FEATURE SLIDERS ==========
st.header("Or enter features manually")
feat1 = st.slider("f0 (Fundamental Frequency)", 50.0, 400.0, 150.0)
feat2 = st.slider("Jitter", 0.0, 0.2, 0.02)
feat3 = st.slider("Shimmer", 0.0, 0.2, 0.03)
feat4 = st.slider("Mean HNR", 0.0, 100.0, 20.0)

manual_input = [feat1, feat2, feat3, feat4]

# ========== PREDICTION =========
if st.button("Predict"):
    input_data = features if features else manual_input
    input_tensor = torch.tensor([input_data], dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = torch.argmax(output, dim=1).item()
        label = "Autistic" if pred_class == 1 else "Non-Autistic"
        confidence = torch.softmax(output, dim=1).squeeze().tolist()

        # Show prediction results
        st.success(f"**Predicted Class:** {label}")
        st.info(f"Confidence: Non-Autistic = {confidence[0]:.2f}, Autistic = {confidence[1]:.2f}")

        # Display explainability table
        st.subheader("Prediction Details")
        feature_names = ["f0 (Hz)", "Jitter", "Shimmer", "Mean HNR (dB)"]
        df = pd.DataFrame({
            "Feature": feature_names,
            "Value": input_data
        })
        st.table(df)

        # Add to prediction summary
        prediction_summary.append({
            "Audio File": audio_file.name if audio_file else "Manual Entry",
            "f0": input_data[0],
            "Jitter": input_data[1],
            "Shimmer": input_data[2],
            "Mean HNR": input_data[3],
            "Prediction": label,
            "Confidence_NonAutistic": confidence[0],
            "Confidence_Autistic": confidence[1]
        })
# ========== FEATURE IMPORTANCE ==========
st.header("Feature Importance (Input Sensitivity)")
input_sensitivity = defaultdict(float)
input_labels = {-1: "f0 (Hz)", -2: "Jitter", -3: "Shimmer", -4: "HNR (dB)"}

for (in_node, out_node), conn in winner.connections.items():
    if conn.enabled and in_node < 0:
        input_sensitivity[in_node] += abs(conn.weight)

input_labels = {-1: "f0", -2: "jitter", -3: "shimmer", -4: "HNR"}
labels = [input_labels[i] for i in sorted(input_sensitivity.keys())]
sensitivities = [input_sensitivity[i] for i in sorted(input_sensitivity.keys())]

fig, ax = plt.subplots()
ax.bar(labels, sensitivities, color='skyblue')
ax.set_title("Input Sensitivity (Total Absolute Weight)")
ax.set_ylabel("Influence")
ax.set_xlabel("Features")
st.pyplot(fig)

# ========== EXPORT SUMMARY ==========
if prediction_summary:
    st.subheader("Download Prediction Report")
    summary_df = pd.DataFrame(prediction_summary)
    csv = summary_df.to_csv(index=False)
    st.download_button("Download CSV Report", csv, "prediction_summary.csv", "text/csv")

                           
