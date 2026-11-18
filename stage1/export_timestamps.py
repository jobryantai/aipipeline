import json
from faster_whisper import WhisperModel

def extract_timestamped_segments(video_path, output_json="segments.json"):
    print("[Stage 1] Extracting whisper segments with timestamps...")
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    segments, _ = model.transcribe(video_path, language="ja")
    
    data = []
    for s in segments:
        data.append({
            "start": round(s.start, 2),
            "end": round(s.end, 2),
            "text": s.text.strip()
        })
        
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print(f"Saved {len(data)} timestamped segments to {output_json}")

if __name__ == "__main__":
    extract_timestamped_segments("my_video.mp4")
