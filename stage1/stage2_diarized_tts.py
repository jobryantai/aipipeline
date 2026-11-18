import os
import json
import torch
import numpy as np
import soundfile as sf
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from TTS.api import TTS

print("[Init] Starting true voice-cloned multi-speaker TTS pipeline...")

os.environ["COQUI_TOS_AGREED"] = "1"
SAMPLE_RATE = 24000
REF_DIR = "speaker_refs"

def load_translator():
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

def build_cloned_audio(segments_json, output_wav="audio_en_synced.wav"):
    if not os.path.exists(segments_json):
        raise FileNotFoundError(f"Diarized segments file not found: {segments_json}.")

    with open(segments_json, "r", encoding="utf-8") as f:
        segments = json.load(f)
        
    tokenizer, trans_model = load_translator()
    
    print("[TTS] Loading XTTS-v2 for zero-shot voice cloning...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    
    total_duration = segments[-1]["end"] + 5.0
    master_audio = np.zeros(int(total_duration * SAMPLE_RATE), dtype=np.float32)
    
    print(f"[Processing] Synthesizing {len(segments)} segments using original cloned voices...")
    for idx, seg in enumerate(segments):
        ja_text = seg["text"]
        start_time = seg["start"]
        speaker = seg.get("speaker", "SPEAKER_00")
        
        if not ja_text:
            continue
            
        en_text = translate_line(ja_text, tokenizer, trans_model)
        
        # Locate the extracted reference audio for this specific speaker
        speaker_wav_path = os.path.join(REF_DIR, f"{speaker}.wav")
        if not os.path.exists(speaker_wav_path):
            # Fallback to SPEAKER_00 if a reference file is missing
            speaker_wav_path = os.path.join(REF_DIR, "SPEAKER_00.wav")
        
        try:
            # speaker_wav clones the target voice; language specifies the output language (English)
            wav = tts.tts(text=en_text, speaker_wav=speaker_wav_path, language="en")
            wav_arr = np.array(wav, dtype=np.float32)
            
            start_sample = int(start_time * SAMPLE_RATE)
            end_sample = start_sample + len(wav_arr)
            
            if end_sample > len(master_audio):
                master_audio = np.pad(master_audio, (0, end_sample - len(master_audio)))
                
            master_audio[start_sample:end_sample] += wav_arr
            print(f"[{idx+1}/{len(segments)}] [{speaker} (Cloned Voice)] at {start_time:.1f}s")
        except Exception as e:
            print(f"Skipping segment {idx+1} due to error: {e}")
            
    sf.write(output_wav, master_audio, SAMPLE_RATE)
    print(f"\nVoice-cloned master audio saved to {output_wav}")

if __name__ == "__main__":
    build_cloned_audio("segments_diarized.json")
