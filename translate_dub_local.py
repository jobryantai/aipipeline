#!/usr/bin/env python3
"""
Local Offline Translation & Voice-Cloning Dubbing Pipeline (RTX 3090)
- Utterance-level voice matching (each line uses its own original audio slice as reference).
- Auto-agrees to Coqui CPML terms.
- Explicitly loads NVIDIA CUDA libraries into global scope via ctypes.
- Patches PyAV for av>=19 compatibility.
"""

import os
import sys

# --- AUTO-AGREE TO COQUI TERMS (NON-COMMERCIAL CPML) ---
os.environ["COQUI_TOS_AGREED"] = "1"
# ------------------------------------------------------

# --- BULLETPROOF NVIDIA CUDA LIBRARIES LOADER ---
import glob
import ctypes
print("[*] Loading NVIDIA CUDA shared libraries into global scope...")
lib_patterns = [
    "/mnt/c/aipipeline/animegan2-pytorch/venv/lib/python3.*/site-packages/nvidia/*/lib/*.so*",
    "/usr/local/lib/python3*/dist-packages/nvidia/*/lib/*.so*"
]
for pattern in lib_patterns:
    for lib_path in glob.glob(pattern):
        if any(name in lib_path for name in ["libcublas.so", "libcudnn.so", "libcusparse.so", "libcurand.so", "libcusolver.so"]):
            try:
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            except Exception:
                pass
# -----------------------------------------------

# --- COMPATIBILITY PATCH FOR PyAV >= 19 ---
import av
_original_av_open = av.open
def _patched_av_open(*args, **kwargs):
    kwargs.pop("metadata_errors", None)
    return _original_av_open(*args, **kwargs)
av.open = _patched_av_open
# ------------------------------------------

import argparse
import subprocess
import torch
from faster_whisper import WhisperModel
from TTS.api import TTS
import torchaudio

def extract_audio(video_path, audio_out_path):
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1", audio_out_path]
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Local Offline Translation & Utterance-Level Voice Cloning")
    parser.add_argument("--input_video", required=True, help="Path to input video chunk")
    parser.add_argument("--output_audio", required=True, help="Path to output dubbed English audio wav")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Running local pipeline on device: {device}")

    temp_dir = "pipeline_output/temp_dub"
    os.makedirs(temp_dir, exist_ok=True)
    
    full_audio = os.path.join(temp_dir, "full_audio.wav")

    print("[*] Step 1: Extracting audio track from video...")
    extract_audio(args.input_video, full_audio)

    print("[*] Step 2: Loading faster-whisper (large-v3) for Cantonese -> English translation...")
    whisper_model = WhisperModel("large-v3", device=device, compute_type="float16")
    segments, info = whisper_model.transcribe(full_audio, task="translate", beam_size=5)
    
    trans_segments = list(segments)
    print(f"[+] Translated {len(trans_segments)} dialogue segments.")

    print("[*] Step 3: Loading Coqui XTTS-v2 for local offline voice cloning...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    print("[*] Step 4: Loading full audio tensor for utterance-level voice cloning...")
    audio_tensor, sr = torchaudio.load(full_audio)

    probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", args.input_video]
    total_duration = float(subprocess.check_output(probe_cmd).decode().strip())

    final_audio_tensor = torch.zeros(1, int(total_duration * 22050))

    print("[*] Step 5: Synthesizing English voice-cloned audio segments with utterance-level matching...")
    for i, seg in enumerate(trans_segments):
        text = seg.text.strip()
        if not text:
            continue
        
        # 1. Extract the exact original audio slice for this segment as the voice reference
        orig_start_sample = int(seg.start * sr)
        orig_end_sample = int(seg.end * sr)
        
        # Ensure minimum duration (at least 2 seconds) for stable XTTS-v2 voice embedding
        min_samples = sr * 2
        if (orig_end_sample - orig_start_sample) < min_samples:
            orig_end_sample = min(audio_tensor.shape[1], orig_start_sample + min_samples)
        
        ref_segment_tensor = audio_tensor[:, orig_start_sample:orig_end_sample]
        
        # Save temporary utterance reference wav for this specific line
        utterance_ref_path = os.path.join(temp_dir, f"ref_utterance_{i:04d}.wav")
        torchaudio.save(utterance_ref_path, ref_segment_tensor, sr)

        seg_wav_path = os.path.join(temp_dir, f"seg_{i:04d}.wav")
        try:
            # 2. Synthesize using THIS segment's unique voice signature
            tts.tts_to_file(
                text=text,
                speaker_wav=utterance_ref_path,
                language="en",
                file_path=seg_wav_path
            )
            
            wav_tensor, _ = torchaudio.load(seg_wav_path)
            start_sample = int(seg.start * 22050)
            end_sample = start_sample + wav_tensor.shape[1]
            
            if end_sample > final_audio_tensor.shape[1]:
                padding = end_sample - final_audio_tensor.shape[1]
                final_audio_tensor = torch.cat([final_audio_tensor, torch.zeros(1, padding)], dim=1)
                
            final_audio_tensor[0, start_sample:end_sample] = wav_tensor[0, :end_sample - start_sample]
            print(f"    [{seg.start:.2f}s -> {seg.end:.2f}s] Cloned & Synthesized: '{text}'")
        except Exception as e:
            print(f"    [!] Error synthesizing segment {i}: {e}")

    print("[*] Step 6: Exporting final dubbed English audio track...")
    torchaudio.save(args.output_audio, final_audio_tensor, 22050)
    print(f"[+] Dubbed offline audio successfully updated: {args.output_audio}")

if __name__ == "__main__":
    main()
