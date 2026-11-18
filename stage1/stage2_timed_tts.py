import os
import json
import torch
import numpy as np
import soundfile as sf
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from TTS.api import TTS

print("[Init] Starting multi-speaker timestamp-aware translation and TTS pipeline...")

os.environ["COQUI_TOS_AGREED"] = "1"
SAMPLE_RATE = 24000

# Diverse pool of built-in XTTS-v2 voices ( mix of male and female profiles )
SPEAKER_POOL = [
    "Claribel Dervla",
    "Andrew Chipper",
    "Daisy Studious",
    "Bily Hatcher",
    "Alison Carol",
    "Gracie Wise",
    "Matthew Hall"
]

def load_translator():
    print("[Translation] Loading NLLB-200...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="jpn_Jpan")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to("cuda")
    return tokenizer, model

def translate_line(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
        max_length=100,
        repetition_penalty=1.2
    )
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

def build_synced_audio(segments_json, output_wav="audio_en_synced.wav"):
    if not os.path.exists(segments_json):
        raise FileNotFoundError(f"Segments file not found: {segments_json}.")

    with open(segments_json, "r", encoding="utf-8") as f:
        segments = json.load(f)
        
    tokenizer, trans_model = load_translator()
    
    print("[TTS] Loading XTTS-v2...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    
    total_duration = segments[-1]["end"] + 5.0
    master_audio = np.zeros(int(total_duration * SAMPLE_RATE), dtype=np.float32)
    
    print(f"[Processing] Synthesizing {len(segments)} timed segments with multi-speaker rotation...")
    for idx, seg in enumerate(segments):
        ja_text = seg["text"]
        start_time = seg["start"]
        
        if not ja_text:
            continue
            
        en_text = translate_line(ja_text, tokenizer, trans_model)
        
        # Assign a rotating speaker profile based on segment index (or hash of text length for variation)
        assigned_speaker = SPEAKER_POOL[idx % len(SPEAKER_POOL)]
        
        try:
            wav = tts.tts(text=en_text, speaker=assigned_speaker, language="en")
            wav_arr = np.array(wav, dtype=np.float32)
            
            start_sample = int(start_time * SAMPLE_RATE)
            end_sample = start_sample + len(wav_arr)
            
            if end_sample > len(master_audio):
                master_audio = np.pad(master_audio, (0, end_sample - len(master_audio)))
                
            master_audio[start_sample:end_sample] += wav_arr
            print(f"[{idx+1}/{len(segments)}] [{assigned_speaker}] at {start_time:.1f}s: '{en_text[:30]}...'")
        except Exception as e:
            print(f"Skipping segment {idx+1} due to error: {e}")
            
    sf.write(output_wav, master_audio, SAMPLE_RATE)
    print(f"\nMulti-speaker synced master English audio saved to {output_wav}")

if __name__ == "__main__":
    build_synced_audio("segments.json")
