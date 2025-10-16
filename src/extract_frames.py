import cv2, os

def extract_frames(video_path, output_dir, fps=25, max_frames=None):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return 0

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[INFO] Opened video: {video_path} | orig_fps={orig_fps:.2f} | total_frames={total_frames}")

    # compute how many source frames to skip to get approx `fps` output
    if fps <= 0 or orig_fps <= 0:
        step = 1
    else:
        step = max(1, int(round(orig_fps / fps)))

    saved = 0
    src_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if src_idx % step == 0:
                out_path = os.path.join(output_dir, f"frame_{saved:06d}.jpg")
                if not cv2.imwrite(out_path, frame):
                    print(f"[WARN] Failed to write frame to {out_path}")
                else:
                    saved += 1
                    if max_frames and saved >= max_frames:
                        break
            src_idx += 1
    finally:
        cap.release()

    print(f"[INFO] Extracted {saved} frames to {output_dir}")
    return saved

if __name__ == "__main__":
    video = r"d:\Pose Estimation of multi agent UAV\S02_D4_A\cam1i_s.avi"   # verify this path exists
    out = r"d:\Pose Estimation of multi agent UAV\data\frames"
    extract_frames(video, out, fps=25)  # Try with fps=25 or adjust as needed