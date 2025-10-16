import sys
from pathlib import Path

# Add project root to path for src imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import cv2
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from src.model import SingleCamMultiDronePoseNet


@st.cache_resource
def load_model():
    """Load and cache the model for predictions"""
    model = SingleCamMultiDronePoseNet()
    model.load_state_dict(torch.load("weights/best_model.pt", map_location="cpu"))
    model.eval()
    return model


def main():
    """Main Streamlit UI function"""
    st.title("Multi-Drone Pose Estimation")
    
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
    if not uploaded_file:
        st.info("Upload a JPG or PNG image to get predictions.")
        return

    try:
        # Process image
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        img = cv2.resize(img, (224, 224))
        img_tensor = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        # Get predictions
        with torch.no_grad():
            preds = model(img_tensor)[0]  # [4, 3]

        # Display results
        st.image(img, caption="Input Image", use_column_width=True)

        # Show predictions in table
        results = []
        for i, pred in enumerate(preds):
            results.append({
                "Drone": f"Drone {i+1}",
                "X (m)": f"{pred[0]:.2f}",
                "Y (m)": f"{pred[1]:.2f}", 
                "Z (m)": f"{pred[2]:.2f}"
            })
        st.table(results)

        # Plot top-down view
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = ["r", "g", "b", "y"]
        for i, pred in enumerate(preds):
            ax.scatter(pred[0], pred[1], c=colors[i], label=f"Drone {i+1}")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Estimated Drone Positions (Top-Down View)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")


if __name__ == "__main__":
    main()
