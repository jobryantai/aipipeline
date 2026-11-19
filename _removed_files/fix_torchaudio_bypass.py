with open("translate_dub_local.py", "r", encoding="utf-8") as f:
    content = f.read()

# Replace torchaudio.load call with direct soundfile read + torch tensor conversion
old_code = 'audio_tensor, sr = torchaudio.load(full_audio, backend="soundfile")'
new_code = '''import soundfile as sf
    data, sr = sf.read(full_audio)
    # Ensure tensor shape is [channel, time] matching torchaudio standard
    if data.ndim == 1:
        audio_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    else:
        audio_tensor = torch.tensor(data, dtype=torch.float32).T'''

content = content.replace(old_code, new_code)

with open("translate_dub_local.py", "w", encoding="utf-8") as f:
    f.write(content)

print("[+] Successfully patched translate_dub_local.py to use soundfile directly.")
