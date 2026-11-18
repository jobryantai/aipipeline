import subprocess
import os

def assemble_video(video_path, audio_path, output_path, orig_vol=0.15):
    print("[Final Stage] Muxing video with timestamp-synced English audio and background ducking...")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Original video not found: {video_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Synced English audio not found: {audio_path}")

    # Mixes original audio (15% volume) with the new timestamp-synced English audio (100% volume)
    filter_complex = f"[0:a]volume={orig_vol}[a0];[1:a]volume=1.0[a1];[a0][a1]amix=inputs=2:duration=first[aout]"

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-filter_complex", filter_complex,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-map", "0:v:0",
        "-map", "[aout]",
        output_path
    ]
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("Video assembly failed.")
        
    print(f"\nSuccess! Final synchronized and dubbed video saved to: {output_path}")

if __name__ == "__main__":
    assemble_video("my_video.mp4", "audio_en_synced.wav", "final_dubbed_video.mp4")
