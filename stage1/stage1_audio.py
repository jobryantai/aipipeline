import os
import glob
import ffmpeg
from faster_whisper import WhisperModel

def find_video_file():
    # Automatically look for any .mp4 file in the current directory
    mp4_files = glob.glob("*.mp4")
    if not mp4_files:
        raise FileNotFoundError("No .mp4 files found in this folder! Please copy a video into D:\\aipipeline\\stage1")
    
    # Pick the first matching video file
    video_path = mp4_files[0]
    print(f" -> Auto-detected video file: {video_path}")
    return video_path

def extract_and_transcribe(output_text_path):
    video_path = find_video_file()
    audio_wav = "temp_extracted_audio.wav"
    
    print("[1/2] Extracting audio stream via FFmpeg...")
    ffmpeg.input(video_path).output(audio_wav, ac=1, ar='16000').run(overwrite_output=True, quiet=True)
    
    print("[2/2] Running Faster-Whisper (Japanese)...")
    # Uses CUDA compute type for local GPU processing
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    segments, _ = model.transcribe(audio_wav, language="ja", beam_size=5)
    
    transcript = " ".join([segment.text for segment in segments])
    
    with open(output_text_path, "w", encoding="utf-8") as f:
        f.write(transcript)
        
    if os.path.exists(audio_wav):
        os.remove(audio_wav)
        
    print(f"Stage 1 Complete! Transcript saved to: {output_text_path}")
    print(f"Transcript text: {transcript}")

if __name__ == "__main__":
    extract_and_transcribe("transcript_ja.txt")
