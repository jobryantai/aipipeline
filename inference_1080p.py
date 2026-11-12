import os
import sys
import glob
import argparse
import cv2
import torch
import numpy as np

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="AnimeGANv2 1080p Inference Pipeline")
parser.add_argument("--input_dir", type=str, required=True, help="Directory containing raw input frames")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save stylized anime frames")
parser.add_argument("--threshold", type=float, default=2.5, help="Caching difference threshold")
args = parser.parse_args()

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

raw_frames_dir = args.input_dir
anime_frames_dir = args.output_dir
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
        
    filename = os.path.basename(f_path)
    out_path = os.path.join(anime_frames_dir, filename)

    # Convert BGR to RGB and normalize for PyTorch
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image).unsqueeze(0).to(device)

    with torch.no_grad():
        out = net(image)

    out = out.squeeze(0).cpu().numpy()
    out = np.transpose(out, (1, 2, 0))
    out = (out + 1.0) * 127.5
    out = np.clip(out, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    cv2.imwrite(out_path, out)

print(f"[*] Inference completed successfully! Saved to {anime_frames_dir}")
