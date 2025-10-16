import torch
from torch.utils.data import DataLoader
from dataset import SingleCamMultiDronePoseDataset
from model import SingleCamMultiDronePoseNet
import torch.optim as optim
from pathlib import Path

def train():
    # Settings
    base_dir = Path(r"d:\Pose Estimation of multi agent UAV")
    cam_id = 25  # <-- Change to match your available files
    batch_size = 8
    epochs = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset
    dataset = SingleCamMultiDronePoseDataset(base_dir, cam_id)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4)
    
    # Create model
    model = SingleCamMultiDronePoseNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    loss_fn = torch.nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for imgs, poses in train_loader:
            imgs, poses = imgs.to(device), poses.to(device)
            optimizer.zero_grad()
            pred_poses = model(imgs)
            loss = loss_fn(pred_poses, poses)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, poses in val_loader:
                imgs, poses = imgs.to(device), poses.to(device)
                pred_poses = model(imgs)
                val_loss += loss_fn(pred_poses, poses).item()
                
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), base_dir / "weights" / "best_model.pt")
            
        scheduler.step(val_loss)

if __name__ == "__main__":
    train()