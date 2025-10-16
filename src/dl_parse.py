import csv
from pathlib import Path

def load_detection_file(path):
    """Return dict: frame_id -> list of detections (dicts with 7 values and score).
       Adjust keys to your detection format.
    """
    out = {}
    p = Path(path)
    with p.open() as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: 
                continue
            vals = [c for c in row if c != ""]
            if not vals:
                continue
            frame_id = int(float(vals[0]))
            rest = [float(x) for x in vals[1:]]
            dets = []
            for i in range(0, len(rest), 7):
                block = rest[i:i+7]
                if len(block) < 7: break
                dets.append({
                    "x": block[0], "y": block[1], "w": block[2], "h": block[3],
                    "x2": block[4], "y2": block[5], "score": block[6]
                })
            out[frame_id] = dets
    return out