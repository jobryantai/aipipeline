with open("batch_pipeline.sh", "r", encoding="utf-8") as f:
    content = f.read()

# Replace the problematic CUDA scale filter with a clean format conversion and hardware upload
old_filter = '-vf "scale_cuda=-1:2160,pad=3840:2160:(ow-iw)/2:0:color=black,hwupload_cuda"'
new_filter = '-vf "scale=-1:2160,pad=3840:2160:(ow-iw)/2:0:color=black,format=yuv420p"'

if old_filter in content:
    content = content.replace(old_filter, new_filter)
else:
    # Flexible fallback replace
    content = content.replace('scale_cuda=-1:2160', 'scale=-1:2160')

with open("batch_pipeline.sh", "w", encoding="utf-8") as f:
    f.write(content)

print("[+] batch_pipeline.sh filter graph corrected for NVENC.")
