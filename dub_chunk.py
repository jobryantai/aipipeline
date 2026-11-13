import os
import sys
import asyncio
import subprocess
import argparse
import numpy as np
import librosa
from faster_whisper import WhisperModel
import edge_tts
from pydub import AudioSegment
import torch

parser = argparse.ArgumentParser(description="Smart Pitch-Matched Multi-Speaker Dubbing Pipeline")
parser.add_argument("--input_video", type=str, required=True, help="Path to source chunk mp4")
parser.add_argument("--output_audio", type=str, required=True, help="Path to output dubbed audio")
args = parser.parse_args()

work_dir = os.path.dirname(args.output_audio)
os.makedirs(work_dir, exist_ok=True)
raw_wav = os.path.join(work_dir, "temp_source.wav")

# 1. Extract 16kHz mono WAV from chunk
subprocess.run([
    "ffmpeg", "-y", "-i", args.input_video,
    "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", raw_wav
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

source_audio = AudioSegment.from_wav(raw_wav)
total_duration_ms = len(source_audio)
y_audio, sr = librosa.load(raw_wav, sr=16000)

# 2. Transcribe & Translate to English via Whisper
print("[*] Translating speech to English via Whisper...", flush=True)
model = WhisperModel("medium", device="cpu", compute_type="int8")
segments, info = model.transcribe(y_audio, task="translate", beam_size=5, vad_filter=True)
whisper_segments = list(segments)

# 3. Perform Speaker Diarization using Pyannote on RTX 3090
print("[*] Performing speaker diarization...", flush=True)
from pyannote.audio import Pipeline
speaker_intervals = []
try:
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.environ.get("HF_TOKEN", None)
    )
    if torch.cuda.is_available():
        diarization_pipeline.to(torch.device("cuda"))
    
    diarization = diarization_pipeline(raw_wav)
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_intervals.append((turn.start, turn.end, speaker))
except Exception as e:
    print(f"[!] Diarization warning ({e}). Using single speaker fallback.", flush=True)
    speaker_intervals = [(0.0, total_duration_ms / 1000.0, "SPEAKER_00")]

def assign_speaker(start, end):
    for s_start, s_end, speaker in speaker_intervals:
        if max(start, s_start) < min(end, s_end):
            return speaker
    return "SPEAKER_00"

# Determine pitch (F0) for each speaker to assign gender-matched voices
speaker_voices = {}
unique_speakers = set(s[2] for s in speaker_intervals)

for spk in unique_speakers:
    # Collect audio samples belonging to this speaker
    spk_samples = []
    for s_start, s_end, s_name in speaker_intervals:
        if s_name == spk:
            start_sample = int(s_start * sr)
            end_sample = int(s_end * sr)
            spk_samples.extend(y_audio[start_sample:end_sample])
    
    if len(spk_samples) > sr: # If at least 1 second of audio
        f0, voiced_flag, voiced_probs = librosa.pyin(np.array(spk_samples), fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'))
        valid_f0 = f0[voiced_flag & ~np.isnan(f0)]
        if len(valid_f0) > 0:
            avg_pitch = np.median(valid_f0)
            # Typically, median F0 < 165 Hz is male, >= 165 Hz is female
            if avg_pitch < 165:
                speaker_voices[spk] = "en-US-GuyNeural" # Male voice
                print(f"[*] Speaker {spk} detected as Male (Pitch: {avg_pitch:.1f}Hz) -> Assigned GuyNeural", flush=True)
            else:
                speaker_voices[spk] = "en-US-AriaNeural" # Female voice
                print(f"[*] Speaker {spk} detected as Female (Pitch: {avg_pitch:.1f}Hz) -> Assigned AriaNeural", flush=True)
            continue
            
    # Default fallback mapping if pitch analysis is inconclusive
    speaker_voices[spk] = "en-US-AriaNeural"

def get_voice_for_speaker(speaker_id):
    return speaker_voices.get(speaker_id, "en-US-AriaNeural")

# 4. Generate timestamp-aligned English TTS with pitch-matched voices
async def synthesize_segments():
    final_track = AudioSegment.silent(duration=total_duration_ms)
    
    for idx, seg in enumerate(whisper_segments):
        text = seg.text.strip()
        if not text:
            continue
            
        start_ms = int(seg.start * 1000)
        end_ms = int(seg.end * 1000)
        slot_duration_ms = max(end_ms - start_ms, 500)
        
        assigned_speaker = assign_speaker(seg.start, seg.end)
        voice = get_voice_for_speaker(assigned_speaker)
        
        seg_mp3 = os.path.join(work_dir, f"seg_{idx:04d}.mp3")
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(seg_mp3)
        
        if os.path.exists(seg_mp3) and os.path.getsize(seg_mp3) > 0:
            tts_clip = AudioSegment.from_mp3(seg_mp3)
            
            # Speed up slightly if TTS clip exceeds dialogue slot
            if len(tts_clip) > slot_duration_ms:
                speed_ratio = min(len(tts_clip) / slot_duration_ms, 1.35)
                sped_up_mp3 = os.path.join(work_dir, f"seg_{idx:04d}_fast.mp3")
                subprocess.run([
                    "ffmpeg", "-y", "-i", seg_mp3,
                    "-filter:a", f"atempo={speed_ratio:.2f}", sped_up_mp3
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.exists(sped_up_mp3):
                    tts_clip = AudioSegment.from_mp3(sped_up_mp3)
                    os.remove(sped_up_mp3)
            
            final_track = final_track.overlay(tts_clip, position=start_ms)
            os.remove(seg_mp3)
            print(f"    [{assigned_speaker} -> {voice}] ({seg.start:.1f}s): {text}", flush=True)

    final_track.export(args.output_audio, format="ipod")

asyncio.run(synthesize_segments())

if os.path.exists(raw_wav):
    os.remove(raw_wav)

print(f"[+] Pitch-matched dubbed audio saved to: {args.output_audio}", flush=True)
