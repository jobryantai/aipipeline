with open("translate_dub_local.py", "r", encoding="utf-8") as f:
    content = f.read()

# Replace the torchaudio shim with robust handlers for soundfile + ffmpeg m4a export + non-silent reference checks
robust_shim = '''import argparse
import subprocess
import torch
import torchaudio
import soundfile as _sf
import os

def _ta_load(path, *args, **kwargs):
    data, sr = _sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr

def _ta_save(path, src, sample_rate, *args, **kwargs):
    arr = src.detach().cpu().numpy()
    if arr.ndim == 1:
        arr = arr[None, :]
    
    path_str = str(path)
    if path_str.endswith(".m4a") or path_str.endswith(".mp3"):
        # Soundfile cannot write m4a/mp3 directly; write to temp wav first, then convert with ffmpeg
        temp_wav = path_str + ".tmp.wav"
        _sf.write(temp_wav, arr.T, sample_rate)
        subprocess.run(["ffmpeg", "-y", "-i", temp_wav, "-c:a", "aac", "-b:a", "192-k", path_str], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
    else:
        _sf.write(path_str, arr.T, sample_rate)

torchaudio.load = _ta_load
torchaudio.save = _ta_save
'''

# Replace old shim if present, or insert at top
if "def _ta_load" in content:
    import re
    content = re.sub(r'def _ta_load.*?\ntorchaudio\.save = _ta_save\n', robust_shim, content, flags=re.DOTALL)
else:
    marker = "import argparse\nimport subprocess\nimport torch\n"
    content = content.replace(marker, marker + "\n" + robust_shim, 1)

# Also ensure reference slice is never completely silent (fallback to first 2 seconds if slice is silent)
old_slice_logic = '''        ref_segment_tensor = audio_tensor[:, orig_start_sample:orig_end_sample]
        
        # Save temporary utterance reference wav for this specific line
        utterance_ref_path = os.path.join(temp_dir, f"ref_utterance_{i:04d}.wav")
        torchaudio.save(utterance_ref_path, ref_segment_tensor, sr)'''

new_slice_logic = '''        ref_segment_tensor = audio_tensor[:, orig_start_sample:orig_end_sample]
        if ref_segment_tensor.numel() == 0 or ref_segment_tensor.abs().max() < 1e-5:
            # Fallback to beginning of audio if slice is empty or silent
            ref_segment_tensor = audio_tensor[:, :int(sr * 2)]
        
        # Save temporary utterance reference wav for this specific line
        utterance_ref_path = os.path.join(temp_dir, f"ref_utterance_{i:04d}.wav")
        torchaudio.save(utterance_ref_path, ref_segment_tensor, sr)'''

content = content.replace(old_slice_logic, new_slice_logic)

with open("translate_dub_local.py", "w", encoding="utf-8") as f:
    f.write(content)

print("[+] translate_dub_local.py definitively patched.")
