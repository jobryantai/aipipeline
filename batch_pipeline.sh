#!/bin/bash
set -e

export INPUT_DIR="/mnt/d/aipipeline/input_videos"
export OUTPUT_DIR="/mnt/d/aipipeline/pipeline_output"
VENV_PYTHON="/mnt/d/aipipeline/animegan2-pytorch/venv/bin/python"
CHUNK_DURATION=300 # 5 minutes in seconds

# ==============================================================================
# MASTER CONFIGURATION: Set your preferred anime style model here!
# Options:
#   - "/mnt/d/aipipeline/animegan2-pytorch/weights/hayao.pt" (Studio Ghibli style)
#   - "/mnt/d/aipipeline/animegan2-pytorch/weights/paprika.pt" (Paprika style)
# ==============================================================================
MODEL_PATH="/mnt/d/aipipeline/animegan2-pytorch/weights/hayao.pt"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "[!] Error: Virtual environment python not found at $VENV_PYTHON" >&2
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "[!] Warning: Selected model not found at $MODEL_PATH. Falling back to paprika.pt"
    MODEL_PATH="/mnt/d/aipipeline/animegan2-pytorch/weights/paprika.pt"
fi

mkdir -p "$OUTPUT_DIR"
shopt -s nullglob

# Support multiple video formats including .rmvb, .mkv, .avi, .mp4, .mov
for video in "$INPUT_DIR"/*.{mp4,rmvb,mkv,avi,mov}; do
    [ -e "$video" ] || continue

    filename=$(basename "$video")
    basename_no_ext="${filename%.*}"

    echo "==================================================="
    echo "[*] Processing Video: $basename_no_ext"
    echo "[*] Selected Anime Style Model: $(basename "$MODEL_PATH")"
    echo "==================================================="

    VID_OUTPUT_DIR="$OUTPUT_DIR/$basename_no_ext"
    CHUNKS_DIR="$VID_OUTPUT_DIR/chunks"
    mkdir -p "$CHUNKS_DIR"

    INPUT_FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 "$video")

    # Step 1: Split video into 5-minute chunks (with resume check)
    EXISTING_CHUNKS=$(ls -1 "$CHUNKS_DIR"/chunk_*.mp4 2>/dev/null | wc -l)
    if [ "$EXISTING_CHUNKS" -gt 0 ]; then
        echo "[*] Found existing chunks in $CHUNKS_DIR. Skipping video splitting."
    else
        echo "[*] Splitting video into 5-minute chunks..."
        ffmpeg -y -i "$video" -c:v libx264 -c:a aac -map 0 -f segment -segment_time $CHUNK_DURATION -reset_timestamps 1 "$CHUNKS_DIR/chunk_%03d.mp4"
    fi

    # Step 2: Process, Dub, and Mux each chunk immediately
    CHUNK_LIST_FILE="$CHUNKS_DIR/processed_chunks.txt"
    > "$CHUNK_LIST_FILE"

    for chunk in "$CHUNKS_DIR"/chunk_*.mp4; do
        [ -e "$chunk" ] || continue
        chunk_name=$(basename "$chunk" .mp4)
        
        CHUNK_OUT_DIR="$CHUNKS_DIR/$chunk_name"
        RAW_FRAMES="$CHUNK_OUT_DIR/stage3_raw_frames"
        ANIME_FRAMES="$CHUNK_OUT_DIR/stage5_anime_frames"
        DUBBED_AUDIO="$CHUNK_OUT_DIR/${chunk_name}_english_dub.m4a"
        CHUNK_PROCESSED_MP4="$CHUNK_OUT_DIR/${chunk_name}_anime.mp4"

        # TRUE RESUME CHECK: Skip entirely if this chunk is already fully rendered
        if [ -f "$CHUNK_PROCESSED_MP4" ] && [ -s "$CHUNK_PROCESSED_MP4" ]; then
            echo "[*] Chunk $chunk_name already fully processed. Skipping."
            echo "file '$chunk_name/$(basename "$CHUNK_PROCESSED_MP4")'" >> "$CHUNK_LIST_FILE"
            continue
        fi

        echo "---------------------------------------------------"
        echo "[*] Processing Chunk: $chunk_name"
        echo "---------------------------------------------------"

        mkdir -p "$RAW_FRAMES" "$ANIME_FRAMES"

        # 2a. Extract raw frames with SAR 16:9 widescreen correction applied
        EXISTING_RAW=$(ls -1 "$RAW_FRAMES" 2>/dev/null | wc -l)
        if [ "$EXISTING_RAW" -gt 0 ]; then
            echo "[*] Found existing raw frames for $chunk_name. Skipping extraction."
        else
            echo "[*] Extracting widescreen-corrected raw frames..."
            ffmpeg -y -i "$chunk" -vf "scale=iw*sar:ih" -q:v 2 "$RAW_FRAMES/%04d.jpg"
        fi

        # 2b. Run official AnimeGANv2 inference using hayao:v2 preset
        "$VENV_PYTHON" -u /mnt/d/aipipeline/pytorch-animeGAN/inference.py \
            --weight hayao:v2 \
            --src "$RAW_FRAMES" \
            --out "$ANIME_FRAMES" \

        # 2c. Generate multi-speaker English dubbed audio with RVC
        if [ ! -f "$DUBBED_AUDIO" ]; then
            echo "[*] Generating multi-speaker English dubbed audio with RVC for $chunk_name..."
            "$VENV_PYTHON" -u /mnt/d/aipipeline/dub_chunk.py \
                --input_video "$chunk" \
                --output_audio "$DUBBED_AUDIO"
        else
            echo "[*] Found existing dubbed audio for $chunk_name. Skipping dubbing."
        fi

        # 2d. Mux into 16:9 4K widescreen container with dubbed audio
        echo "[*] Muxing frames into 16:9 4K widescreen container..."
        ffmpeg -y -framerate "$INPUT_FPS" \
          -i "$ANIME_FRAMES/%04d.jpg" \
          -i "$DUBBED_AUDIO" \
          -vf "scale=-1:2160,pad=3840:2160:(ow-iw)/2:0" \
          -c:v libx264 -pix_fmt yuv420p \
          -c:a aac -map 0:v:0 -map 1:a:0 \
          -shortest \
          -fps_mode cfr \
          "$CHUNK_PROCESSED_MP4"

        echo "[+] Completed 16:9 4K widescreen anime chunk: $CHUNK_PROCESSED_MP4"
        echo "file '$chunk_name/$(basename "$CHUNK_PROCESSED_MP4")'" >> "$CHUNK_LIST_FILE"
    done

    # Step 3: Merge all completed chunks into final production video
    FINAL_MERGED_VIDEO="$OUTPUT_DIR/${basename_no_ext}_english_dubbed_anime.mp4"
    echo "[*] Merging all completed chunks into final production video..."

    ffmpeg -y -f concat -safe 0 -i "$CHUNK_LIST_FILE" -c copy "$FINAL_MERGED_VIDEO"

    echo "[+] Successfully completed and merged: $FINAL_MERGED_VIDEO"
done
