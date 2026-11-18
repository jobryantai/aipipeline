import json
import os
import subprocess

def extract_speaker_samples(video_path, segments_json="segments_diarized.json", output_dir="speaker_refs"):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(segments_json, "r", encoding="utf-8") as f:
        segments = json.load(f)
        
    found_speakers = set()
    
    print("[Extract] Finding clean reference clips for each speaker...")
    for seg in segments:
        speaker = seg["speaker"]
        if speaker not in found_speakers:
            # Grab a clear 10-second window starting from their first appearance
            start = max(0, seg["start"])
            duration = 10.0  # XTTS performs best with 6 to 10 seconds of clean speech
            
            output_wav = os.path.join(output_dir, f"{speaker}.wav")
            
            # Use ffmpeg to slice out the reference audio sample
            cmd = [
                "ffmpeg", "-y", "-ss", str(start), "-i", video_path, 
                "-t", str(duration), "-vn", "-acodec", "pcm_s16le", 
                "-ar", "22050", "-ac", "1", output_wav
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            print(f"Extracted reference sample for {speaker} -> {output_wav}")
            found_speakers.add(speaker)
            
    print(f"Successfully extracted {len(found_speakers)} speaker reference voices into '{output_dir}/'")

if __name__ == "__main__":
    extract_speaker_samples("my_video.mp4")
