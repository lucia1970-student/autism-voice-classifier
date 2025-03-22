import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as pl
import neat
import neat.nn
from evolved_model import EvolvedNN, winner, config  # import your model and genome

# Load trained model
neat_net = neat.nn.FeedForwardNetwork.create(winner, config)
model = EvolvedNN(neat_net, winner)
model.load_state_dict(torch.load("evolved_neat_model.pth"))
model.eval()

st.title("Autism Voice Classifier Dashboard")

# Input
st.header("Input Voice Features")
feat1 = st.slider("Feature 1", -1.0, 1.0, 0.0)
feat2 = st.slider("Feature 2", -1.0, 1.0, 0.0)
feat3 = st.slider("Feature 3", -1.0, 1.0, 0.0)
feat4 = st.slider("Feature 4", -1.0, 1.0, 0.0)

if st.button("Predict"):
    input_tensor = torch.tensor([[feat1, feat2, feat3, feat4]], dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
        label = "Autistic" if prediction == 1 else "Non-Autistic"
        st.success(f"Predicted Class: **{label}**")

# Feature Importance
st.header("Feature Importance (Input Sensitivity)")

from collections import defaultdict

input_sensitivity = defaultdict(float)
for (in_node, out_node), conn in winner.connections.items():
    if conn.enabled and in_node < 0:
        input_sensitivity[in_node] += abs(conn.weight)

input_labels = {-1: "feat1", -2: "feat2", -3: "feat3", -4: "feat4"}
labels = [input_labels[i] for i in sorted(input_sensitivity.keys())]
sensitivities = [input_sensitivity[i] for i in sorted(input_sensitivity.keys())]

fig, ax = plt.subplots()
ax.bar(labels, sensitivities, color='lightblue')
ax.set_title("Input Sensitivity")
ax.set_ylabel("Total Absolute Weight")
st.pyplot(fig)
                           
