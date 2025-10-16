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

        # Split predictions into position and orientation
        positions = preds[:, :3]  # [4, 3]
        quaternions = preds[:, 3:]  # [4, 4]

        # Convert quaternions to Euler angles for visualization
        def quaternion_to_euler(q):
            # q = [qx, qy, qz, qw]
            qx, qy, qz, qw = q
            
            # Roll (x-axis rotation)
            sinr_cosp = 2 * (qw * qx + qy * qz)
            cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
            roll = np.arctan2(sinr_cosp, cosr_cosp)
            
            # Pitch (y-axis rotation)
            sinp = 2 * (qw * qy - qz * qx)
            pitch = np.arcsin(sinp)
            
            # Yaw (z-axis rotation)
            siny_cosp = 2 * (qw * qz + qx * qy)
            cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
            yaw = np.arctan2(siny_cosp, cosy_cosp)
            
            return np.array([roll, pitch, yaw]) * 180.0 / np.pi  # Convert to degrees

        # Show predictions in table
        results = []
        for i, (pos, quat) in enumerate(zip(positions, quaternions)):
            euler = quaternion_to_euler(quat)
            results.append({
                "Drone": f"Drone {i+1}",
                "X (m)": f"{pos[0]:.2f}",
                "Y (m)": f"{pos[1]:.2f}", 
                "Z (m)": f"{pos[2]:.2f}",
                "Roll (°)": f"{euler[0]:.1f}",
                "Pitch (°)": f"{euler[1]:.1f}",
                "Yaw (°)": f"{euler[2]:.1f}"
            })
        st.table(results)

        # Create 3D plot with pose visualization
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = ['r', 'g', 'b', 'y']
        arrow_length = 0.5  # Length of orientation arrows
        
        for i, (pos, quat) in enumerate(zip(positions, quaternions)):
            # Plot drone position
            ax.scatter(pos[0], pos[1], pos[2], c=colors[i], s=100, label=f'Drone {i+1}')
            
            # Calculate orientation vectors using quaternion
            def quat_rotate(q, v):
                qx, qy, qz, qw = q
                x, y, z = v
                
                # Apply quaternion rotation
                wx = qw * x + qy * z - qz * y
                wy = qw * y + qz * x - qx * z
                wz = qw * z + qx * y - qy * x
                
                xx = qx * x + qy * y + qz * z
                
                return np.array([
                    2 * (xx * qx + wx * qw - wy * qz + wz * qy),
                    2 * (xx * qy + wy * qw - wz * qx + wx * qz),
                    2 * (xx * qz + wz * qw - wx * qy + wy * qx)
                ])
            
            # Draw orientation arrows
            forward = quat_rotate(quat, [arrow_length, 0, 0])
            up = quat_rotate(quat, [0, 0, arrow_length])
            
            # Plot orientation vectors
            ax.quiver(pos[0], pos[1], pos[2], 
                     forward[0], forward[1], forward[2],
                     colors[i], alpha=0.6)
            ax.quiver(pos[0], pos[1], pos[2], 
                     up[0], up[1], up[2],
                     colors[i], alpha=0.3)
        
        # Set labels and title
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Estimated Drone Poses (3D View)')
        ax.legend()
        
        # Auto-scale axes
        positions = positions.numpy()
        max_range = np.array([
            positions[:,0].max() - positions[:,0].min(),
            positions[:,1].max() - positions[:,1].min(),
            positions[:,2].max() - positions[:,2].min()
        ]).max() / 2.0
        
        mid_x = (positions[:,0].max() + positions[:,0].min()) * 0.5
        mid_y = (positions[:,1].max() + positions[:,1].min()) * 0.5
        mid_z = (positions[:,2].max() + positions[:,2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        st.pyplot(fig)
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")


if __name__ == "__main__":
    main()
