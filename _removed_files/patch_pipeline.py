import re

file_path = "translate_dub_local.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace torchaudio.load(full_audio) with explicit soundfile backend
old_pattern = r"audio_tensor,\s*sr\s*=\s*torchaudio\.load\s*\(\s*full_audio\s*\)"
new_replacement = "audio_tensor, sr = torchaudio.load(full_audio, backend=\"soundfile\")"

if re.search(old_pattern, content):
    content = re.sub(old_pattern, new_replacement, content)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("[+] Successfully patched translate_dub_local.py with backend='soundfile'")
else:
    print("[!] Pattern not found exactly. Checking alternative patterns...")
    # Fallback replacement if variable name differs slightly
    content = content.replace("torchaudio.load(full_audio)", "torchaudio.load(full_audio, backend=\"soundfile\")")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("[+] Applied fallback patch to translate_dub_local.py")

