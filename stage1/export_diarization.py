import json
from faster_whisper import WhisperModel

def export_smart_segments(video_path, output_json="segments_diarized.json"):
    print("[Stage 1] Running Whisper transcription...")
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    segments, _ = model.transcribe(video_path, language="ja")
    
    diarized_segments = []
    current_speaker = 0
    last_end = 0.0
    
    for s in segments:
        start = round(s.start, 2)
        end = round(s.end, 2)
        text = s.text.strip()
        
        if not text:
            continue
            
        # If there's a long pause (> 1.5 seconds), assume a change in speaker or turn
        if start - last_end > 1.5:
            current_speaker = (current_speaker + 1) % 4  # Rotate between 4 speakers (SPEAKER_00 to SPEAKER_03)
            
        diarized_segments.append({
            "start": start,
            "end": end,
            "speaker": f"SPEAKER_0{current_speaker}",
            "text": text
        })
        last_end = end
        
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(diarized_segments, f, ensure_ascii=False, indent=2)
        
    print(f"Saved {len(diarized_segments)} smart-turn segments to {output_json}")

if __name__ == "__main__":
    export_smart_segments("my_video.mp4")
