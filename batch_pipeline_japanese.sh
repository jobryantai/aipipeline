#!/bin/bash
set -e

export INPUT_DIR="/mnt/c/aipipeline/input_videos"
export OUTPUT_DIR="/mnt/c/aipipeline/pipeline_output"
VENV_PYTHON="/mnt/c/aipipeline/animegan2-pytorch/venv/bin/python"
CHUNK_DURATION=300 # 5 minutes in seconds

# Match files case-insensitively; unmatched globs expand to nothing instead of a literal string
shopt -s nullglob nocaseglob

if [ ! -f "$VENV_PYTHON" ]; then
    echo "[!] Error: Virtual environment python not found at $VENV_PYTHON" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

VIDEOS=("$INPUT_DIR"/*.{mp4,mpg,avi,mkv,mov})
if [ "${#VIDEOS[@]}" -eq 0 ]; then
    echo "[!] No videos (mp4, mpg, avi, mkv, mov) found in $INPUT_DIR" >&2
    exit 1
fi

for video in "${VIDEOS[@]}"; do
    filename=$(basename "$video")
    basename_no_ext="${filename%.*}"
    ext_lower=$(echo "${filename##*.}" | tr '[:upper:]' '[:lower:]')

    echo "==================================================="
    echo "[*] Processing Video: $basename_no_ext"
    echo "==================================================="

    VID_OUTPUT_DIR="$OUTPUT_DIR/$basename_no_ext"
    CHUNKS_DIR="$VID_OUTPUT_DIR/chunks"
    mkdir -p "$CHUNKS_DIR"

    INPUT_FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=noprint_wrappers=1:nokey=1 "$video")

    # Step 1: Split video into 5-minute MP4 chunks
    echo "[*] Checking/Splitting video into 5-minute chunks..."
    if [ "$ext_lower" = "mp4" ]; then
        # MP4 can be stream-copied directly
        ffmpeg -y -i "$video" -c copy -map 0 -f segment -segment_time $CHUNK_DURATION -reset_timestamps 1 "$CHUNKS_DIR/chunk_%03d.mp4"
    else
        # AVI/MKV/MOV/MPG codecs may not fit in MP4 -> re-encode to H.264/AAC while splitting
        ffmpeg -y -i "$video" \
          -map "0:v:0" -map "0:a:0?" \
          -c:v h264_nvenc -preset p5 -cq 19 -pix_fmt yuv420p \
          -c:a aac -b:a 192k \
          -force_key_frames "expr:gte(t,n_forced*$CHUNK_DURATION)" \
          -f segment -segment_time $CHUNK_DURATION -reset_timestamps 1 \
          "$CHUNKS_DIR/chunk_%03d.mp4"
    fi

    # Step 2: Process, Dub, and Mux each chunk immediately
    CHUNK_LIST_FILE="$CHUNKS_DIR/processed_chunks_ja.txt"
    > "$CHUNK_LIST_FILE"

    for chunk in "$CHUNKS_DIR"/chunk_*.mp4; do
        chunk_name=$(basename "$chunk" .mp4)
        echo "---------------------------------------------------"
        echo "[*] Processing Chunk: $chunk_name"
        echo "---------------------------------------------------"

        CHUNK_OUT_DIR="$CHUNKS_DIR/$chunk_name"
        RAW_FRAMES="$CHUNK_OUT_DIR/stage3_raw_frames"
        ANIME_FRAMES="$CHUNK_OUT_DIR/stage5_anime_frames"
        DUBBED_AUDIO="$CHUNK_OUT_DIR/${chunk_name}_japanese_dub.m4a"
        CHUNK_PROCESSED_MP4="$CHUNK_OUT_DIR/${chunk_name}_anime_ja.mp4"

        mkdir -p "$RAW_FRAMES" "$ANIME_FRAMES"

        # 2a. Extract raw frames for this chunk if needed
        EXISTING_RAW=$(ls -1 "$RAW_FRAMES" 2>/dev/null | wc -l)
        if [ "$EXISTING_RAW" -gt 0 ]; then
            echo "[*] Found existing raw frames for $chunk_name. Skipping extraction."
        else
            ffmpeg -y -i "$chunk" -q:v 2 "$RAW_FRAMES/%04d.jpg"
        fi

        # 2b. Run AnimeGANv2 inference on this chunk
        echo "[*] Running AnimeGANv2 inference on $chunk_name..."
        cd /mnt/c/aipipeline/animegan2-pytorch
        /mnt/c/aipipeline/animegan2-pytorch/venv/bin/python -u /mnt/c/aipipeline/inference_1080p.py \
            --input_dir "$RAW_FRAMES" \
            --output_dir "$ANIME_FRAMES" \
            --threshold 2.5 \
            --fp16
        cd /mnt/c/aipipeline

        # 2c. Translate and generate Japanese dubbed audio
        if [ ! -f "$DUBBED_AUDIO" ]; then
            echo "[*] Generating translated multi-speaker Japanese audio for $chunk_name..."
            "$VENV_PYTHON" -u /mnt/c/aipipeline/translate_dub_japanese.py \
                --input_video "$chunk" \
                --output_audio "$DUBBED_AUDIO"
        else
            echo "[*] Found existing dubbed audio for $chunk_name. Skipping dubbing."
        fi

        # 2d. Mux with 16:9 4K widescreen padding, Japanese dub, and soft background original audio
        echo "[*] Muxing frames into 4K container with Japanese dub and background original audio..."
        ffmpeg -y -framerate "$INPUT_FPS" \
          -i "$ANIME_FRAMES/%04d.jpg" \
          -i "$DUBBED_AUDIO" \
          -i "$chunk" \
          -filter_complex "[2:a]volume=0.15[orig_bg];[1:a][orig_bg]amix=inputs=2:duration=first[aout]" \
          -vf "scale=-1:2160,pad=3840:2160:(ow-iw)/2:0:color=black,format=yuv420p" \
          -c:v h264_nvenc -preset p7 -cq 19 -pix_fmt yuv420p \
          -c:a aac -map 0:v:0 -map "[aout]" \
          -shortest \
          -fps_mode cfr \
          "$CHUNK_PROCESSED_MP4"

        echo "[+] Completed 16:9 4K widescreen anime chunk: $CHUNK_PROCESSED_MP4"
        echo "file '$chunk_name/$(basename "$CHUNK_PROCESSED_MP4")'" >> "$CHUNK_LIST_FILE"
    done

    # Step 3: Merge all completed chunks into final production video
    FINAL_MERGED_VIDEO="$OUTPUT_DIR/${basename_no_ext}_japanese_dubbed_anime.mp4"
    echo "[*] Merging all completed chunks into final production video..."

    ffmpeg -y -f concat -safe 0 -i "$CHUNK_LIST_FILE" -c copy "$FINAL_MERGED_VIDEO"

    echo "[+] Successfully completed and merged: $FINAL_MERGED_VIDEO"
done
