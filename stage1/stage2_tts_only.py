import os
import re
import torch
import soundfile as sf
import numpy as np
from TTS.api import TTS

os.environ["COQUI_TOS_AGREED"] = "1"

def split_text_into_safe_chunks(text, max_chars=180):
    """Split text by sentences and clauses into chunks safe for XTTS limits."""
    # Split by periods, question marks, commas, and conjunctions
    raw_sentences = re.split(r'(?<=[.!?])\s+|,\s+', text)
    safe_chunks = []
    
    for sentence in raw_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # If single sentence is still too long, hard slice it
        while len(sentence) > max_chars:
            cut_idx = sentence.rfind(' ', 0, max_chars)
            if cut_idx == -1:
                cut_idx = max_chars
            safe_chunks.append(sentence[:cut_idx].strip())
            sentence = sentence[cut_idx:].strip()
            
        if sentence:
            safe_chunks.append(sentence)
            
    return safe_chunks

def generate_tts(en_text_path, output_audio_path):
    print("[TTS] Loading XTTS-v2 and generating English audio...")
    
    if not os.path.exists(en_text_path):
        raise FileNotFoundError(f"Transcript not found: {en_text_path}")
        
    with open(en_text_path, "r", encoding="utf-8") as f:
        en_text = f.read().strip()
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    
    try:
        speakers = list(tts.synthesizer.tts_model.speaker_manager.speakers.keys())
        speaker_name = speakers[0] if speakers else "Claribel Dervla"
    except Exception:
        speaker_name = "Claribel Dervla"
        
    print(f"Using built-in speaker profile: {speaker_name}")
    
    chunks = split_text_into_safe_chunks(en_text, max_chars=180)
    print(f"Total safe chunks to process: {len(chunks)}")
    
    combined_audio = []
    sample_rate = 24000  # XTTS native sample rate
    
    for i, chunk in enumerate(chunks):
        print(f"Synthesizing chunk {i+1}/{len(chunks)}: '{chunk[:40]}...'")
        # Generate raw waveform array in memory for each chunk
        wav = tts.tts(
            text=chunk,
            speaker=speaker_name,
            language="en"
        )
        combined_audio.extend(wav)
        
    # Save the combined audio array to file
    sf.write(output_audio_path, np.array(combined_audio), sample_rate)
    print(f"\nTTS complete! Combined English audio saved to {output_audio_path}")

if __name__ == "__main__":
    generate_tts("transcript_en.txt", "audio_en.wav")
