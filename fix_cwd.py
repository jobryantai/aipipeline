with open("batch_pipeline.sh", "r") as f:
    code = f.read()

old_snippet = """        # 2b. Run AnimeGANv2 inference on this chunk
        echo "[*] Running AnimeGANv2 inference on $chunk_name..."
        /mnt/c/aipipeline/animegan2-pytorch/venv/bin/python -u /mnt/c/aipipeline/inference_1080p.py \\
            --input_dir "$RAW_FRAMES" \\
            --output_dir "$ANIME_FRAMES" \\
            --threshold 2.5 \\
            --fp16"""

new_snippet = """        # 2b. Run AnimeGANv2 inference on this chunk
        echo "[*] Running AnimeGANv2 inference on $chunk_name..."
        cd /mnt/c/aipipeline/animegan2-pytorch
        /mnt/c/aipipeline/animegan2-pytorch/venv/bin/python -u /mnt/c/aipipeline/inference_1080p.py \\
            --input_dir "$RAW_FRAMES" \\
            --output_dir "$ANIME_FRAMES" \\
            --threshold 2.5 \\
            --fp16
        cd /mnt/c/aipipeline"""

if old_snippet in code:
    code = code.replace(old_snippet, new_snippet)
    with open("batch_pipeline.sh", "w") as f:
        f.write(code)
    print("[+] Successfully updated batch_pipeline.sh with correct working directory!")
else:
    print("[!] Could not match exact inference block. Please check batch_pipeline.sh")
