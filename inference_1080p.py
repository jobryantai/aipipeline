#!/usr/bin/env python3
"""
AnimeGANv2 Inference with Batching
Optimized for high-end GPUs (RTX 3090 / 4090 / 5090)
"""

import os
import sys
import time
import argparse
import cv2
import torch
import numpy as np
from collections import deque

parser = argparse.ArgumentParser(description="AnimeGANv2 Inference Pipeline with Batching + FPS-based ETA")
parser.add_argument("--input_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--threshold", type=float, default=2.5)          # kept for compatibility
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--batch_size", type=int, default=8, help="Frames per batch (4-16 recommended)")
args = parser.parse_args()

PIPELINE_DIR = "/mnt/c/aipipeline"
SUBMODULE_DIR = os.path.join(PIPELINE_DIR, "animegan2-pytorch")
if SUBMODULE_DIR not in sys.path:
    sys.path.insert(0, SUBMODULE_DIR)

try:
    from model import Generator
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "animegan2-pytorch")))
    from model import Generator

# -------------------- Model Loading --------------------
weight_path = os.path.join(SUBMODULE_DIR, "weights/hayao.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Using device: {device}")

net = Generator().to(device)
checkpoint = torch.load(weight_path, map_location=device, weights_only=False)

if isinstance(checkpoint, dict):
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint

# Strip "module." prefix if present
cleaned_state_dict = {}
for k, v in state_dict.items():
    name = k[7:] if k.startswith("module.") else k
    cleaned_state_dict[name] = v

net.load_state_dict(cleaned_state_dict)

if args.fp16:
    net = net.half()
net.eval()

# -------------------- Prepare frames --------------------
os.makedirs(args.output_dir, exist_ok=True)

frame_paths = sorted([
    os.path.join(args.input_dir, f)
    for f in os.listdir(args.input_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])
total_frames = len(frame_paths)

existing_files = set(os.listdir(args.output_dir))
already_done = len(existing_files)
print(f"[*] Total frames: {total_frames} | Already completed: {already_done}")
print(f"[*] Batch size: {args.batch_size} | FP16: {args.fp16}")

# -------------------- Helpers --------------------
def preprocess_batch(imgs, device, fp16=False):
    """imgs: list of BGR uint8 numpy arrays → batched tensor"""
    batch = []
    for img in imgs:
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        image = np.transpose(image, (2, 0, 1))  # HWC → CHW
        batch.append(image)
    batch = np.stack(batch, axis=0)  # BCHW
    tensor = torch.from_numpy(batch).to(device, non_blocking=True)
    if fp16:
        tensor = tensor.half()
    return tensor

def postprocess(tensor):
    """Single tensor (CHW) → BGR uint8"""
    out = tensor.float().cpu().numpy()
    out = np.transpose(out, (1, 2, 0))
    out = (out + 1.0) * 127.5
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

# -------------------- Main loop --------------------
processed_count = 0
recent_times = deque(maxlen=20)
batch_size = max(1, args.batch_size)

# Collect frames that still need processing
todo = []
for idx, f_path in enumerate(frame_paths):
    filename = os.path.basename(f_path)
    if filename not in existing_files:
        todo.append((idx, f_path, filename))

print(f"[*] Frames remaining to process: {len(todo)}")

i = 0
while i < len(todo):
    batch_items = todo[i : i + batch_size]
    batch_imgs = []
    batch_meta = []  # (original_idx, filename, out_path)

    for orig_idx, f_path, filename in batch_items:
        img = cv2.imread(f_path)
        if img is None:
            continue
        batch_imgs.append(img)
        out_path = os.path.join(args.output_dir, filename)
        batch_meta.append((orig_idx, filename, out_path))

    if not batch_imgs:
        i += batch_size
        continue

    frame_start = time.time()

    # Inference
    with torch.no_grad():
        input_tensor = preprocess_batch(batch_imgs, device, fp16=args.fp16)
        output_tensor = net(input_tensor)

    # Write results
    for j, (orig_idx, filename, out_path) in enumerate(batch_meta):
        out_img = postprocess(output_tensor[j])
        cv2.imwrite(out_path, out_img)
        processed_count += 1

    # Timing & progress
    batch_duration = time.time() - frame_start
    frames_in_batch = len(batch_meta)
    recent_times.append(batch_duration / frames_in_batch)

    avg_sec_per_frame = sum(recent_times) / len(recent_times)
    current_fps = 1.0 / avg_sec_per_frame if avg_sec_per_frame > 0 else 0.0

    current_num = batch_meta[-1][0] + 1
    frames_left = total_frames - current_num
    eta_seconds = frames_left * avg_sec_per_frame

    hours, rem = divmod(int(eta_seconds), 3600)
    minutes, seconds = divmod(rem, 60)
    eta_str = f"{hours}h {minutes}m {seconds}s" if hours > 0 else f"{minutes}m {seconds}s"

    percent = (current_num / total_frames) * 100
    print(
        f"[{current_num}/{total_frames} ({percent:.1f}%)] "
        f"Batch {frames_in_batch} frames | "
        f"Speed: {current_fps:.2f} fps ({avg_sec_per_frame:.2f}s/fr) | "
        f"ETA: {eta_str}",
        flush=True,
    )

    i += batch_size

print(f"[*] Done! Newly processed: {processed_count} | Total in output: {len(os.listdir(args.output_dir))}")