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

        # Display input image and predictions side by side
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(img, caption="Input Image", use_container_width=True)
            
            # Detailed position table
            st.subheader("Drone Positions")
            results = []
            for i, pred in enumerate(preds):
                results.append({
                    "Drone": f"Drone {i+1}",
                    "X (m)": f"{pred[0]:.2f}",
                    "Y (m)": f"{pred[1]:.2f}", 
                    "Z (m)": f"{pred[2]:.2f}",
                    "Height": f"{abs(pred[2]):.2f}m",
                    "Distance": f"{np.sqrt(pred[0]**2 + pred[1]**2):.2f}m"
                })
            st.table(results)

        with col2:
            # Plot top-down view
            st.subheader("Top-Down View")
            fig1, ax1 = plt.subplots(figsize=(8, 8))
            colors = ['r', 'g', 'b', 'y']
            
            # Draw coordinate grid
            ax1.grid(True, linestyle='--', alpha=0.6)
            ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            
            # Plot drones with numbers
            for i, pred in enumerate(preds):
                ax1.scatter(pred[0], pred[1], c=colors[i], s=100, label=f'Drone {i+1}')
                ax1.annotate(f'D{i+1}', (pred[0], pred[1]), 
                           xytext=(5, 5), textcoords='offset points')
            
            ax1.set_xlabel('X Position (m)')
            ax1.set_ylabel('Y Position (m)')
            ax1.set_title('Drone Positions (Top View)')
            ax1.legend()
            st.pyplot(fig1)

        # 3D visualization
        st.subheader("3D View")
        fig2 = plt.figure(figsize=(10, 8))
        ax2 = fig2.add_subplot(111, projection='3d')
        
        # Plot drones in 3D
        for i, pred in enumerate(preds):
            ax2.scatter(pred[0], pred[1], pred[2], c=colors[i], s=100, label=f'Drone {i+1}')
            # Draw vertical lines to ground
            ax2.plot([pred[0], pred[0]], [pred[1], pred[1]], [0, pred[2]], 
                    c=colors[i], linestyle='--', alpha=0.5)
            
        # Set labels and title
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Y Position (m)')
        ax2.set_zlabel('Height (m)')
        ax2.set_title('Drone Positions (3D View)')
        
        # Add ground plane grid
        x_range = np.array([min(preds[:,0]), max(preds[:,0])])
        y_range = np.array([min(preds[:,1]), max(preds[:,1])])
        margin = 1.0  # Add 1m margin
        x_grid, y_grid = np.meshgrid(
            np.linspace(x_range.min() - margin, x_range.max() + margin, 10),
            np.linspace(y_range.min() - margin, y_range.max() + margin, 10)
        )
        z_grid = np.zeros_like(x_grid)
        ax2.plot_surface(x_grid, y_grid, z_grid, alpha=0.1, color='gray')
        
        ax2.legend()
        ax2.grid(True)
        
        # Adjust 3D view angle
        ax2.view_init(elev=20, azim=45)
        st.pyplot(fig2)
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")


if __name__ == "__main__":
    main()
