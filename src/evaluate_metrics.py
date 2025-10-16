import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import SingleCamMultiDronePoseDataset
from model import SingleCamMultiDronePoseNet
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde


def compute_success_rate(errors, thresholds):
    """Compute success rate at different error thresholds."""
    success_rates = []
    for thresh in thresholds:
        success_rate = (errors <= thresh).mean() * 100
        success_rates.append(success_rate)
    return success_rates


def compute_temporal_consistency(predictions, window_size=5):
    """Compute temporal consistency as the average velocity change."""
    if len(predictions) < window_size + 1:
        return np.nan
    
    velocities = np.diff(predictions, axis=0)  # [N-1, 3]
    acc = np.diff(velocities, axis=0)  # [N-2, 3]
    return np.mean(np.linalg.norm(acc, axis=1))


def evaluate_and_visualize(base_dir=".", cam_id=25, batch_size=8, save_plots=True):
    """Comprehensive evaluation with enhanced metrics and visualizations."""
    base_dir = Path(base_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)

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

    # Error thresholds for success rate (in meters)
    thresholds = [0.1, 0.2, 0.5, 1.0, 2.0]

    # Per-drone detailed metrics
    print("\nPer-Drone Metrics:")
    print("-" * 50)
    
    for drone_id in range(4):
        drone_preds = all_preds[:, drone_id]     # [N, 3]
        drone_targets = all_targets[:, drone_id]  # [N, 3]
        
        # Basic metrics
        mae = np.mean(np.abs(drone_preds - drone_targets), axis=0)
        rmse = np.sqrt(np.mean((drone_preds - drone_targets)**2, axis=0))
        errors = np.linalg.norm(drone_preds - drone_targets, axis=1)
        
        # Success rates
        success_rates = compute_success_rate(errors, thresholds)
        
        # Temporal consistency
        temp_consistency = compute_temporal_consistency(drone_preds)
        
        print(f"\nDrone {drone_id+1} Metrics:")
        print(f"MAE (x,y,z): {mae[0]:.3f}, {mae[1]:.3f}, {mae[2]:.3f} meters")
        print(f"RMSE (x,y,z): {rmse[0]:.3f}, {rmse[1]:.3f}, {rmse[2]:.3f} meters")
        print(f"Mean Error: {np.mean(errors):.3f} meters")
        print(f"Median Error: {np.median(errors):.3f} meters")
        print(f"Temporal Consistency Score: {temp_consistency:.3f}")
        
        print("\nSuccess Rates:")
        for thresh, rate in zip(thresholds, success_rates):
            print(f"@ {thresh:.1f}m: {rate:.1f}%")

        if save_plots:
            # 3D Trajectory Plot
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(drone_targets[:, 0], drone_targets[:, 1], drone_targets[:, 2], 
                   'b-', label='Ground Truth', alpha=0.7)
            ax.plot(drone_preds[:, 0], drone_preds[:, 1], drone_preds[:, 2], 
                   'r--', label='Predicted', alpha=0.7)
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Y Position (m)')
            ax.set_zlabel('Z Position (m)')
            ax.set_title(f'Drone {drone_id+1} 3D Trajectory')
            ax.legend()
            plt.savefig(results_dir / f"drone{drone_id+1}_3d_trajectory.png")
            plt.close()

            # Error Distribution
            plt.figure(figsize=(8, 6))
            plt.hist(errors, bins=30, density=True, alpha=0.7)
            kde = gaussian_kde(errors)
            x_range = np.linspace(0, max(errors), 100)
            plt.plot(x_range, kde(x_range), 'r-', label='KDE')
            plt.xlabel('Position Error (m)')
            plt.ylabel('Density')
            plt.title(f'Drone {drone_id+1} Error Distribution')
            plt.legend()
            plt.savefig(results_dir / f"drone{drone_id+1}_error_dist.png")
            plt.close()

    # Overall metrics
    print("\nOverall Model Performance:")
    print("-" * 50)
    all_errors = np.linalg.norm(all_preds - all_targets, axis=2).flatten()
    overall_success_rates = compute_success_rate(all_errors, thresholds)
    
    print(f"Overall Mean Error: {np.mean(all_errors):.3f} meters")
    print(f"Overall Median Error: {np.median(all_errors):.3f} meters")
    print(f"Error Std Dev: {np.std(all_errors):.3f} meters")
    print("\nOverall Success Rates:")
    for thresh, rate in zip(thresholds, overall_success_rates):
        print(f"@ {thresh:.1f}m: {rate:.1f}%")

    if save_plots:
        # Combined error distribution
        plt.figure(figsize=(12, 6))
        plt.boxplot([np.linalg.norm(all_preds[:, i] - all_targets[:, i], axis=1) 
                    for i in range(4)],
                   labels=[f'Drone {i+1}' for i in range(4)])
        plt.ylabel('Position Error (m)')
        plt.title('Position Error Distribution by Drone')
        plt.savefig(results_dir / "overall_error_distribution.png")
        plt.close()

        # Save metrics to file
        with open(results_dir / "evaluation_metrics.txt", "w") as f:
            for drone_id in range(4):
                errors = np.linalg.norm(all_preds[:, drone_id] - all_targets[:, drone_id], axis=1)
                f.write(f"\nDrone {drone_id+1} Metrics:\n")
                f.write(f"Mean Error: {np.mean(errors):.3f} meters\n")
                f.write(f"Median Error: {np.median(errors):.3f} meters\n")
                f.write(f"Error Std Dev: {np.std(errors):.3f} meters\n")
                f.write("\nSuccess Rates:\n")
                success_rates = compute_success_rate(errors, thresholds)
                for thresh, rate in zip(thresholds, success_rates):
                    f.write(f"@ {thresh:.1f}m: {rate:.1f}%\n")

    return {
        'mean_error': np.mean(all_errors),
        'median_error': np.median(all_errors),
        'std_error': np.std(all_errors),
        'success_rates': dict(zip(thresholds, overall_success_rates))
    }


if __name__ == "__main__":
    base_dir = Path(r"d:\Pose Estimation of multi agent UAV")
    metrics = evaluate_and_visualize(base_dir=base_dir, save_plots=True)
    print("\nEvaluation complete! Check the 'results' directory for detailed plots and metrics.")