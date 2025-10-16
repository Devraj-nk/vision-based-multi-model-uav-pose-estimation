import torch
import torchvision

class SingleCamMultiDronePoseNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Single CNN backbone
        backbone = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        backbone.fc = torch.nn.Identity()
        self.backbone = backbone
            
        # MLP head for 4 drone positions
        self.pose_head = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 12)  # 4 drones x (x,y,z) position
        )
        
    def forward(self, x):
        # x: batch of images [B, C, H, W]
        features = self.backbone(x)  # [B, 512]
        poses = self.pose_head(features)  # [B, 12]
        return poses.view(-1, 4, 3)  # reshape to [B, 4, 3] for 4 drones