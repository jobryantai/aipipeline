with open("batch_pipeline.sh", "r") as f:
    code = f.read()

old_block = """        # 2b. Run AnimeGANv2 inference on this chunk
        \"$VENV_PYTHON\" -u /mnt/c/aipipeline/inference_1080p.py \\
            --input_dir \"$RAW_FRAMES\" \\
            --output_dir \"$ANIME_FRAMES\" \\
            --threshold 2.5 \\
            --fp16"""

new_block = """        # 2b. Run AnimeGANv2 inference on this chunk
        echo "[*] Running AnimeGANv2 inference on $chunk_name..."
        /mnt/c/aipipeline/animegan2-pytorch/venv/bin/python -u /mnt/c/aipipeline/inference_1080p.py \\
            --input_dir "$RAW_FRAMES" \\
            --output_dir "$ANIME_FRAMES" \\
            --threshold 2.5 \\
            --fp16"""

if old_block in code:
    code = code.replace(old_block, new_block)
    with open("batch_pipeline.sh", "w") as f:
        f.write(code)
    print("[+] Successfully updated inference invocation in batch_pipeline.sh")
else:
    print("[!] Block not matched exactly, please check lines.")
