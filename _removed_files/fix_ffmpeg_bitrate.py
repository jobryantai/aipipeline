with open("translate_dub_local.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix the bitrate argument from "-b:a", "192-k" to "-b:a", "192k"
content = content.replace('"-b:a", "192-k"', '"-b:a", "192k"')

with open("translate_dub_local.py", "w", encoding="utf-8") as f:
    f.write(content)

print("[+] FFmpeg bitrate argument corrected.")
