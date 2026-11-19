import os
import sys
import glob
import time
import argparse
import cv2
import torch
import numpy as np
from collections import deque

parser = argparse.ArgumentParser(description="AnimeGANv2 Inference Pipeline with FPS-based ETA")
parser.add_argument("--input_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--threshold", type=float, default=2.5)
parser.add_argument("--fp16", action="store_true")
args = parser.parse_args()

PIPELINE_DIR = "/mnt/c/aipipeline"
SUBMODULE_DIR = os.path.join(PIPELINE_DIR, "animegan2-pytorch")
if SUBMODULE_DIR not in sys.path:
    sys.path.insert(0, SUBMODULE_DIR)

try:
    from model import Generator
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "animegan2-pytorch")))
    from model import Generator

weight_path = os.path.join(SUBMODULE_DIR, "weights/hayao.pt")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Generator().to(device)
checkpoint = torch.load(weight_path, map_location=device)
if isinstance(checkpoint, dict):
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint

# Strip "module." prefix if saved from DataParallel training
cleaned_state_dict = {}
for k, v in state_dict.items():
    name = k[7:] if k.startswith("module.") else k
    cleaned_state_dict[name] = v

net.load_state_dict(cleaned_state_dict)

if args.fp16:
    net = net.half()
net.eval()

os.makedirs(args.output_dir, exist_ok=True)
frame_paths = sorted([
    os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
    if f.lower().endswith(".jpg")
])
total_frames = len(frame_paths)

existing_files = set(os.listdir(args.output_dir))
already_done = len(existing_files)
print(f"[*] Total frames: {total_frames} | Already completed: {already_done}", flush=True)

processed_count = 0
recent_times = deque(maxlen=15)

for idx, f_path in enumerate(frame_paths):
    filename = os.path.basename(f_path)
    
    if filename in existing_files:
        continue

    current_num = idx + 1
    out_path = os.path.join(args.output_dir, filename)
    img = cv2.imread(f_path)
    if img is None:
        continue

    frame_start = time.time()

    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image).unsqueeze(0).to(device)

    if args.fp16:
        image = image.half()

    with torch.no_grad():
        out = net(image)

    out = out.squeeze(0).cpu().numpy()
    out = np.transpose(out, (1, 2, 0))
    out = (out + 1.0) * 127.5
    out = np.clip(out, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    cv2.imwrite(out_path, out)
    processed_count += 1

    frame_duration = time.time() - frame_start
    recent_times.append(frame_duration)

    # Calculate actual current FPS and average seconds per frame
    avg_sec_per_frame = sum(recent_times) / len(recent_times)
    current_fps = 1.0 / avg_sec_per_frame if avg_sec_per_frame > 0 else 0.0

    frames_left = total_frames - current_num
    eta_seconds = frames_left * avg_sec_per_frame
    
    hours, rem = divmod(int(eta_seconds), 3600)
    minutes, seconds = divmod(rem, 60)
    eta_str = f"{hours}h {minutes}m {seconds}s" if hours > 0 else f"{minutes}m {seconds}s"
    
    percent = (current_num / total_frames) * 100
    print(f"[{current_num}/{total_frames} ({percent:.1f}%)] Styling: {filename} | Speed: {current_fps:.2f} fps ({avg_sec_per_frame:.2f}s/fr) | ETA: {eta_str}", flush=True)

print(f"[*] Done! Newly processed: {processed_count} | Total in output: {len(os.listdir(args.output_dir))}", flush=True)
