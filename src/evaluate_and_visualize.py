import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import SingleCamMultiDronePoseDataset
from model import SingleCamMultiDronePoseNet

# Settings
base_dir = Path(r"d:\Pose Estimation of multi agent UAV")
cam_id = 25
batch_size = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model and weights
model = SingleCamMultiDronePoseNet().to(device)
model.load_state_dict(torch.load(base_dir / "weights" / "best_model.pt"))
model.eval()

# Create dataset and loader
dataset = SingleCamMultiDronePoseDataset(base_dir, cam_id)
val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

all_preds = []
all_targets = []

with torch.no_grad():
    for imgs, poses in val_loader:
        imgs, poses = imgs.to(device), poses.to(device)
        preds = model(imgs)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(poses.cpu().numpy())

all_preds = np.concatenate(all_preds, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

# Compute metrics
mae = np.mean(np.abs(all_preds - all_targets))
mse = np.mean((all_preds - all_targets) ** 2)
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")

# Plot error histogram
errors = np.linalg.norm(all_preds - all_targets, axis=1)
plt.figure()
plt.hist(errors, bins=30)
plt.xlabel('Position Error (m)')
plt.ylabel('Count')
plt.title('Distribution of Position Errors')
plt.show()

# Plot predicted vs true positions (scatter)
plt.figure()
plt.scatter(all_targets[:, 0], all_targets[:, 1], label='True', alpha=0.5)
plt.scatter(all_preds[:, 0], all_preds[:, 1], label='Predicted', alpha=0.5)
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Predicted vs True Positions')
plt.legend()
plt.show()
