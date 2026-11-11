import os
import sys
import glob
import cv2
import torch
import numpy as np

PIPELINE_DIR = "/mnt/d/aipipeline"
SUBMODULE_DIR = os.path.join(PIPELINE_DIR, "animegan2-pytorch")

if SUBMODULE_DIR not in sys.path:
    sys.path.insert(0, SUBMODULE_DIR)

print("[*] Importing official Generator from animegan2-pytorch repository...")
from model import Generator

weight_path = os.path.join(SUBMODULE_DIR, "weights/face_paint_512_v2.pt")
print(f"[*] Loading model weights from {weight_path}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Generator().to(device)
net.load_state_dict(torch.load(weight_path, map_location=device))
net.eval()

raw_frames_dir = os.path.join(PIPELINE_DIR, "pipeline_output/stage3_raw_frames")
anime_frames_dir = os.path.join(PIPELINE_DIR, "pipeline_output/stage5_anime_frames")
os.makedirs(anime_frames_dir, exist_ok=True)

frame_paths = sorted(glob.glob(os.path.join(raw_frames_dir, "*.jpg")))
print(f"[*] Processing {len(frame_paths)} frames at 1080p with caching enabled...")

prev_raw_resized = None
prev_out_img = None
cache_hits = 0

for idx, f_path in enumerate(frame_paths):
    img = cv2.imread(f_path)
    if img is None:
        continue
    
    h, w, _ = img.shape
    new_h = (h // 32) * 32
    new_w = (w // 32) * 32
    img_resized = cv2.resize(img, (new_w, new_h))

    out_name = os.path.basename(f_path)
    out_path = os.path.join(anime_frames_dir, out_name)

    if prev_raw_resized is not None:
        diff = np.mean(np.abs(img_resized.astype(np.float32) - prev_raw_resized.astype(np.float32)))
        if diff < 1.0:
            cv2.imwrite(out_path, prev_out_img)
            cache_hits += 1
            if (idx + 1) % 500 == 0:
                print(f"Processed {idx + 1}/{len(frame_paths)} frames... (Cache hits: {cache_hits})")
            continue

    img_t = img_resized.astype(np.float32) / 127.5 - 1.0
    img_t = torch.from_numpy(img_t).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        out = net(img_t)
    
    out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out = np.clip((out + 1.0) * 127.5, 0, 255).astype(np.uint8)
    
    if out.shape[0] != h or out.shape[1] != w:
        out = cv2.resize(out, (w, h))

    cv2.imwrite(out_path, out)

    prev_raw_resized = img_resized
    prev_out_img = out
    
    if (idx + 1) % 500 == 0:
        print(f"Processed {idx + 1}/{len(frame_paths)} frames... (Cache hits: {cache_hits})")

print(f"[*] Inference completed! Total cache hits (frames skipped/copied): {cache_hits}/{len(frame_paths)}")
