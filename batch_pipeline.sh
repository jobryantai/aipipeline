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

for video in "$INPUT_DIR"/*.mp4; do
    [ -e "$video" ] || continue

    filename=$(basename "$video")
    basename_no_ext="${filename%.*}"

    echo "==================================================="
    echo "[*] Processing Video: $basename_no_ext"
    echo "==================================================="

    VID_OUTPUT_DIR="$OUTPUT_DIR/$basename_no_ext"
    CHUNKS_DIR="$VID_OUTPUT_DIR/chunks"
    mkdir -p "$CHUNKS_DIR"

    INPUT_FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 "$video")

    # Step 1: Split video into 5-minute chunks
    echo "[*] Checking/Splitting video into 5-minute chunks..."
    ffmpeg -y -i "$video" -c copy -map 0 -f segment -segment_time $CHUNK_DURATION -reset_timestamps 1 "$CHUNKS_DIR/chunk_%03d.mp4"

    # Step 2: Process, Dub, and Mux each chunk immediately
    CHUNK_LIST_FILE="$VID_OUTPUT_DIR/processed_chunks.txt"
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

        # 2a. Extract raw frames for this chunk if needed
        EXISTING_RAW=$(ls -1 "$RAW_FRAMES" 2>/dev/null | wc -l)
        if [ "$EXISTING_RAW" -gt 0 ]; then
            echo "[*] Found existing raw frames for $chunk_name. Skipping extraction."
        else
            ffmpeg -y -i "$chunk" -q:v 2 "$RAW_FRAMES/%04d.jpg"
        fi

        # 2b. Run AnimeGANv2 inference on this chunk
        "$VENV_PYTHON" -u /mnt/d/aipipeline/inference_1080p.py \
            --input_dir "$RAW_FRAMES" \
            --output_dir "$ANIME_FRAMES" \
            --threshold 2.5 \
            --fp16

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
    
    (cd "$CHUNKS_DIR" && ffmpeg -y -f concat -safe 0 -i "$CHUNK_LIST_FILE" -c copy "$FINAL_MERGED_VIDEO")

    echo "[+] Successfully completed and merged: $FINAL_MERGED_VIDEO"
done
