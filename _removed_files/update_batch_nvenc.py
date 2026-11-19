with open("batch_pipeline.sh", "r", encoding="utf-8") as f:
    content = f.read()

# Replace the ffmpeg software encoding flags with hardware-accelerated NVENC flags for RTX 5090
old_ffmpeg_mux = '''        ffmpeg -y -framerate "$INPUT_FPS" \\
          -i "$ANIME_FRAMES/%04d.jpg" \\
          -i "$DUBBED_AUDIO" \\
          -i "$chunk" \\
          -filter_complex "[2:a]volume=0.15[orig_bg];[1:a][orig_bg]amix=inputs=2:duration=first[aout]" \\
          -vf "scale=-1:2160,pad=3840:2160:(ow-iw)/2:0" \\
          -c:v libx264 -pix_fmt yuv420p \\
          -c:a aac -map 0:v:0 -map "[aout]" \\
          -shortest \\
          -fps_mode cfr \\
          "$CHUNK_PROCESSED_MP4"'''

new_ffmpeg_mux = '''        ffmpeg -y -framerate "$INPUT_FPS" \\
          -i "$ANIME_FRAMES/%04d.jpg" \\
          -i "$DUBBED_AUDIO" \\
          -i "$chunk" \\
          -filter_complex "[2:a]volume=0.15[orig_bg];[1:a][orig_bg]amix=inputs=2:duration=first[aout]" \\
          -vf "scale_cuda=-1:2160,pad=3840:2160:(ow-iw)/2:0:color=black,hwupload_cuda" \\
          -c:v h264_nvenc -preset p7 -cq 19 -pix_fmt yuv420p \\
          -c:a aac -map 0:v:0 -map "[aout]" \\
          -shortest \\
          -fps_mode cfr \\
          "$CHUNK_PROCESSED_MP4"'''

if old_ffmpeg_mux in content:
    content = content.replace(old_ffmpeg_mux, new_ffmpeg_mux)
    print("[+] Successfully updated batch_pipeline.sh to use RTX 5090 (h264_nvenc).")
else:
    print("[!] Exact match not found. Trying flexible replacement...")
    content = content.replace("-c:v libx264", "-c:v h264_nvenc -preset p7 -cq 19")

with open("batch_pipeline.sh", "w", encoding="utf-8") as f:
    f.write(content)

