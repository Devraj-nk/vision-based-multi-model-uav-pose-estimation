import torch
import torchvision

class SingleCamMultiDronePoseNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Single CNN backbone
        backbone = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        backbone.fc = torch.nn.Identity()
        self.backbone = backbone
            
        # MLP head for 4 drone poses (position + orientation)
        self.pose_head = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 28)  # 4 drones x (x,y,z, qx,qy,qz,qw)
        )
        
    def forward(self, x):
        # x: batch of images [B, C, H, W]
        features = self.backbone(x)  # [B, 512]
        poses = self.pose_head(features)  # [B, 28]
        
        # Split into position and orientation
        B = poses.shape[0]
        poses = poses.view(B, 4, 7)  # [B, 4, 7] for 4 drones (xyz + quaternion)
        
        # Normalize quaternions
        pos = poses[..., :3]  # [B, 4, 3]
        quat = poses[..., 3:]  # [B, 4, 4]
        quat = torch.nn.functional.normalize(quat, p=2, dim=-1)  # ensure unit quaternions
        
        return torch.cat([pos, quat], dim=-1)  # [B, 4, 7]