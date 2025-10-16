import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import SingleCamMultiDronePoseDataset
from model import SingleCamMultiDronePoseNet

def evaluate():
    # Settings
    base_dir = Path(r"d:\Pose Estimation of multi agent UAV")
    cam_id = 25
    batch_size = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = SingleCamMultiDronePoseNet().to(device)
    weights_path = base_dir / "weights" / "best_model.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"No weights found at {weights_path}")
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    # Load dataset
    dataset = SingleCamMultiDronePoseDataset(base_dir, cam_id)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Collect predictions
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for imgs, poses in loader:
            imgs = imgs.to(device)
            preds = model(imgs)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(poses.numpy())

    all_preds = np.concatenate(all_preds, axis=0)    # [N, 4, 3]
    all_targets = np.concatenate(all_targets, axis=0) # [N, 4, 3]

    # Compute per-drone metrics
    for drone_id in range(4):
        drone_preds = all_preds[:, drone_id]     # [N, 3]
        drone_targets = all_targets[:, drone_id]  # [N, 3]
        
        mae = np.mean(np.abs(drone_preds - drone_targets), axis=0)
        rmse = np.sqrt(np.mean((drone_preds - drone_targets)**2, axis=0))
        
        print(f"\nDrone {drone_id+1} Metrics:")
        print(f"MAE (x,y,z): {mae[0]:.3f}, {mae[1]:.3f}, {mae[2]:.3f} meters")
        print(f"RMSE (x,y,z): {rmse[0]:.3f}, {rmse[1]:.3f}, {rmse[2]:.3f} meters")

        # Plot trajectory
        plt.figure(figsize=(10, 10))
        plt.plot(drone_targets[:, 0], drone_targets[:, 1], 'b-', label='Ground Truth')
        plt.plot(drone_preds[:, 0], drone_preds[:, 1], 'r--', label='Predicted')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title(f'Drone {drone_id+1} Trajectory')
        plt.legend()
        plt.axis('equal')
        plt.savefig(base_dir / f"results/drone{drone_id+1}_trajectory.png")
        plt.close()

    # Plot combined error distribution
    errors = np.linalg.norm(all_preds - all_targets, axis=2)  # [N, 4]
    plt.figure(figsize=(10, 6))
    plt.boxplot([errors[:, i] for i in range(4)], labels=[f'Drone {i+1}' for i in range(4)])
    plt.ylabel('Position Error (m)')
    plt.title('Position Error Distribution by Drone')
    plt.savefig(base_dir / "results/error_distribution.png")
    plt.close()

if __name__ == "__main__":
    evaluate()