with open("translate_dub_local.py", "r", encoding="utf-8") as f:
    content = f.read()

shim = '''import argparse
import subprocess
import torch
import torchaudio
import soundfile as _sf

# --- torchaudio>=2.9 removed built-in I/O (needs torchcodec); route through soundfile ---
def _ta_load(path, *args, **kwargs):
    data, sr = _sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr

def _ta_save(path, src, sample_rate, *args, **kwargs):
    arr = src.detach().cpu().numpy()
    if arr.ndim == 1:
        arr = arr[None, :]
    _sf.write(str(path), arr.T, sample_rate)

torchaudio.load = _ta_load
torchaudio.save = _ta_save
# ---------------------------------------------------------------------------------------
'''

marker = "import argparse\nimport subprocess\nimport torch\n"
if "_ta_load" in content:
    print("[*] Already patched.")
elif marker in content:
    content = content.replace(marker, shim, 1)
    with open("translate_dub_local.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("[+] Patched torchaudio load/save to use soundfile.")
else:
    print("[!] Marker not found; check the import block in translate_dub_local.py")
