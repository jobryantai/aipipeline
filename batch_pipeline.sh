#!/bin/bash
set -e

export INPUT_DIR="/mnt/d/aipipeline/input_videos"
export OUTPUT_DIR="/mnt/d/aipipeline/pipeline_output"
VENV_PYTHON="/mnt/d/aipipeline/animegan2-pytorch/venv/bin/python"
CHUNK_DURATION=300 # 5 minutes in seconds

if [ ! -f "$VENV_PYTHON" ]; then
    echo "[!] Error: Virtual environment python not found at $VENV_PYTHON" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ------------------------------------------------------------------
# GOAL #2: RMVB Ingestion & Automatic Container Normalization
# ------------------------------------------------------------------
echo "[*] Scanning for input videos (.mp4, .rmvb, etc.)..."
for raw_video in "$INPUT_DIR"/*; do
    [ -e "$raw_video" ] || continue
    
    filename=$(basename "$raw_video")
    ext="${filename##*.}"
    basename_no_ext="${filename%.*}"
    
    video="$raw_video"
    
    # If file is .rmvb, transcode/normalize to .mp4 container first
    if [ "${ext,,}" = "rmvb" ]; then
        normalized_mp4="$INPUT_DIR/${basename_no_ext}.mp4"
        if [ ! -f "$normalized_mp4" ]; then
            echo "[*] Detected .rmvb file: $filename. Normalizing to .mp4 container..."
            ffmpeg -y -i "$raw_video" -c:v libx264 -crf 20 -c:a aac -b:a 192k "$normalized_mp4"
            echo "[+] Successfully normalized .rmvb to: $normalized_mp4"
        else
            echo "[*] Normalized MP4 already exists for $filename. Using $normalized_mp4"
        fi
        video="$normalized_mp4"
        basename_no_ext="${basename_no_ext}"
    elif [ "${ext,,}" != "mp4" ]; then
        continue
    fi

    echo "==================================================="
    echo "[*] Processing Video: $basename_no_ext"
    echo "==================================================="

    VID_OUTPUT_DIR="$OUTPUT_DIR/$basename_no_ext"
    CHUNKS_DIR="$VID_OUTPUT_DIR/chunks"
    mkdir -p "$CHUNKS_DIR"

    INPUT_FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 "$video")

    # ------------------------------------------------------------------
    # GOAL #3: Robust Auto-Splitting Check
    # ------------------------------------------------------------------
    if [ ! "$(ls -A "$CHUNKS_DIR"/chunk_*.mp4 2>/dev/null)" ]; then
        echo "[*] No chunks found in $CHUNKS_DIR. Automatically splitting video into chunks..."
        ffmpeg -y -i "$video" -c copy -map 0 -f segment -segment_time $CHUNK_DURATION -reset_timestamps 1 "$CHUNKS_DIR/chunk_%03d.mp4"
    else
        echo "[*] Existing chunks detected in $CHUNKS_DIR. Skipping auto-splitting."
    fi

    # Step 2: Process, Dub, and Mux each chunk immediately
    CHUNK_LIST_FILE="$CHUNKS_DIR/processed_chunks.txt"
    > "$CHUNK_LIST_FILE"

    for chunk in "$CHUNKS_DIR"/chunk_*.mp4; do
        [ -e "$chunk" ] || continue
        chunk_name=$(basename "$chunk" .mp4)
        echo "---------------------------------------------------"
        echo "[*] Processing Chunk: $chunk_name"
        echo "---------------------------------------------------"

        CHUNK_OUT_DIR="$CHUNKS_DIR/$chunk_name"
        RAW_FRAMES="$CHUNK_OUT_DIR/stage3_raw_frames"
        ANIME_FRAMES="$CHUNK_OUT_DIR/stage5_anime_frames"
        DUBBED_AUDIO="$CHUNK_OUT_DIR/${chunk_name}_english_dub.m4a"
        CHUNK_PROCESSED_MP4="$CHUNK_OUT_DIR/${chunk_name}_anime.mp4"

        mkdir -p "$RAW_FRAMES" "$ANIME_FRAMES"

        # ------------------------------------------------------------------
        # GOAL #1 & #3: Dynamic Frame Generation with PAR Correction
        # ------------------------------------------------------------------
        EXISTING_RAW=$(ls -1 "$RAW_FRAMES"/*.jpg 2>/dev/null | wc -l)
        if [ "$EXISTING_RAW" -gt 0 ]; then
            echo "[*] Found $EXISTING_RAW existing raw frames for $chunk_name. Skipping extraction."
        else
            echo "[*] Extracting raw frames dynamically with PAR correction (scale=iw*sar:ih,setsar=1)..."
            ffmpeg -y -i "$chunk" -vf "scale=iw*sar:ih,setsar=1" -q:v 2 "$RAW_FRAMES/%04d.jpg"
        fi

        # 2b. Run AnimeGANv2 inference on this chunk
        "$VENV_PYTHON" -u /mnt/d/aipipeline/inference_1080p.py \
            --input_dir "$RAW_FRAMES" \
            --output_dir "$ANIME_FRAMES" \
            --threshold 2.5

        # 2c. Generate translated multi-speaker English dubbed audio for this chunk
        if [ ! -f "$DUBBED_AUDIO" ]; then
            echo "[*] Generating translated multi-speaker English audio for $chunk_name..."
            "$VENV_PYTHON" -u /mnt/d/aipipeline/dub_chunk.py \
                --input_video "$chunk" \
                --output_audio "$DUBBED_AUDIO"
        else
            echo "[*] Found existing dubbed audio for $chunk_name. Skipping dubbing."
        fi

        # 2d. Mux with 16:9 4K widescreen padding and pitch-matched English audio
        echo "[*] Muxing frames into 16:9 4K widescreen container with English audio..."
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
