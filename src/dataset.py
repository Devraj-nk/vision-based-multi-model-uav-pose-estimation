import torch, os, cv2, pandas as pd, numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class SingleCamMultiDronePoseDataset(Dataset):
    def __init__(self, base_dir, cam_id=25, transform=None):
        """
        Args:
            base_dir: root directory containing S0*_D*_A folders
            cam_id: camera id to use [1,2,...]
            transform: optional transforms on crops
        """
        self.base_dir = Path(base_dir)
        self.transform = transform
        self.cam_id = cam_id
        
        # Load camera parameters and poses for all drones
        cam_param_pattern = f"*/dron1_cam_{cam_id}.csv"
        cam_param_files = list(self.base_dir.glob(cam_param_pattern))
        print(f"Searching for camera parameter files with pattern: {cam_param_pattern}")
        print(f"Found files: {cam_param_files}")
        if not cam_param_files:
            raise FileNotFoundError(f"No camera parameter file found for pattern: {cam_param_pattern}")
        self.cam_params = pd.read_csv(cam_param_files[0], sep=';')
        
        # Load poses for all 4 drones
        self.poses = {}
        for drone_id in range(1, 5):
            pose_pattern = f"*/dron{drone_id}_pos_{cam_id}.csv"
            pose_files = list(self.base_dir.glob(pose_pattern))
            if not pose_files:
                raise FileNotFoundError(f"No pose file found for pattern: {pose_pattern}")
            self.poses[drone_id] = pd.read_csv(pose_files[0], sep=';')
            
        # Load detections for the camera
        det_pattern = f"*/dl_data/cam{cam_id}*.csv"
        det_files = list(self.base_dir.glob(det_pattern))
        print(f"Searching for detection files with pattern: {det_pattern}")
        print(f"Found files: {det_files}")
        if not det_files:
            print("Warning: No detection file found. Proceeding without detections.")
            self.detections = None
        else:
            self.detections = self._load_detections(det_files[0])
        
        # Get common frames across all drones
        self.frame_ids = sorted(set.intersection(*[
            set(self.poses[d].index) for d in range(1, 5)
        ]))
        print(f"First 10 frame indices from pose files: {self.frame_ids[:10]}")

        # List available image files
        frames_dir = self.base_dir / "data/frames"
        print(f"Looking for images in: {frames_dir}")
        available_images = sorted([p.name for p in frames_dir.glob("frame_*.jpg")])
        print(f"First 10 available image files: {available_images[:10]}")
        if not available_images:
            print("No images found in the expected folder. Check your image extraction or path.")
        
        # Filter out frames without images
        valid_frame_ids = []
        missing_frames = []
        for frame_id in self.frame_ids:
            frame_path = frames_dir / f"frame_{frame_id:06d}.jpg"
            if frame_path.exists():
                valid_frame_ids.append(frame_id)
            else:
                missing_frames.append(str(frame_path))
        if missing_frames:
            print(f"Warning: Missing image files for frames: {missing_frames[:10]}{'...' if len(missing_frames)>10 else ''}")
        self.frame_ids = valid_frame_ids
        print(f"Total valid frames with images and pose: {len(self.frame_ids)}")
        if len(self.frame_ids) == 0:
            raise RuntimeError("No valid frames found. Check your image and pose files.")
        
        print("Folders in base_dir:")
        for p in self.base_dir.iterdir():
            if p.is_dir():
                print(f"  {p}")
        print("Files in data/frames folder:")
        if frames_dir.exists():
            for p in frames_dir.iterdir():
                print(f"  {p}")
        else:
            print(f"  Folder {frames_dir} does not exist.")
        
    def _load_detections(self, det_file):
        """Parse detection CSV into frame_id -> list of detections dict"""
        dets = {}
        with open(det_file) as f:
            for line in f:
                if not line.strip(): continue
                vals = [float(x) for x in line.strip().split(',') if x]
                if len(vals) < 8: continue
                
                frame_id = int(vals[0])
                dets[frame_id] = []
                
                # Parse groups of 7 values into detection dicts
                for i in range(1, len(vals), 7):
                    if i+6 >= len(vals): break
                    dets[frame_id].append({
                        'x': vals[i], 'y': vals[i+1],
                        'w': vals[i+2], 'h': vals[i+3],
                        'conf': vals[i+6]
                    })
        return dets

    def __len__(self):
        return len(self.frame_ids)
        
    def __getitem__(self, idx):
        frame_id = self.frame_ids[idx]
        frames_dir = self.base_dir / "data/frames"
        frame_path = frames_dir / f"frame_{frame_id:06d}.jpg"
        
        # Load frame
        img = cv2.imread(str(frame_path))
        if img is None:
            raise ValueError(f"Could not load {frame_path}")
            
        # Resize full frame
        img = cv2.resize(img, (224, 224))
        if self.transform:
            img = self.transform(img)
        
        img_tensor = torch.tensor(img).permute(2,0,1).float()/255
        
        # Get poses for all drones
        poses = []
        for drone_id in range(1, 5):
            pose = self.poses[drone_id].loc[frame_id][['pos_x[m]','pos_y[m]','pos_z[m]']].values
            poses.append(torch.tensor(pose, dtype=torch.float32))
            
        # If detections are not loaded, just use the full frame as before
        return img_tensor, torch.stack(poses)