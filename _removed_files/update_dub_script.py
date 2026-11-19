with open("translate_dub_local.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix Whisper log message for Japanese source video
content = content.replace(
    'Loading faster-whisper (large-v3) for Cantonese -> English translation',
    'Loading faster-whisper (large-v3) with auto-language detection for Japanese -> English translation'
)

# Ensure torchaudio.load uses soundfile backend explicitly
content = content.replace(
    'audio_tensor, sr = torchaudio.load(full_audio)',
    'audio_tensor, sr = torchaudio.load(full_audio, backend="soundfile")'
)

with open("translate_dub_local.py", "w", encoding="utf-8") as f:
    f.write(content)

print("[+] translate_dub_local.py successfully updated programmatically.")
