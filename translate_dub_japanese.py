#!/usr/bin/env python3
"""
Local Offline Translation & Voice-Cloning Dubbing Pipeline -> JAPANESE (RTX 3090)

Flow per chunk:
  1. Extract audio
  2. faster-whisper large-v3 TRANSCRIBES the speech in its original language (auto-detected)
  3. NLLB-200 translates each segment's text into Japanese
  4. Coqui XTTS-v2 speaks the Japanese text, cloning the original speaker's voice per utterance

Extra requirements in the venv (beyond translate_dub_local.py):
    pip install transformers sentencepiece
    pip install "TTS[ja]"        # or: pip install cutlet fugashi unidic-lite   (XTTS Japanese text processing)
"""

import os
import sys

# --- AUTO-AGREE TO COQUI TERMS (NON-COMMERCIAL CPML) ---
os.environ["COQUI_TOS_AGREED"] = "1"
# ------------------------------------------------------

# --- NVIDIA CUDA LIBRARIES LOADER ---
import glob
import ctypes
print("[*] Loading NVIDIA CUDA shared libraries into global scope...")
lib_patterns = [
    "/mnt/c/aipipeline/animegan2-pytorch/venv/lib/python3.*/site-packages/nvidia/*/lib/*.so*",
    "/usr/local/lib/python3*/dist-packages/nvidia/*/lib/*.so*"
]
for pattern in lib_patterns:
    for lib_path in glob.glob(pattern):
        if "/nvidia/cu13/" not in lib_path and any(name in lib_path for name in ["libcublas.so", "libcudnn.so", "libcusparse.so", "libcurand.so", "libcusolver.so"]):
            try:
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            except Exception:
                pass
# -----------------------------------

# --- COMPATIBILITY PATCH FOR PyAV >= 19 ---
import av
_original_av_open = av.open
def _patched_av_open(*args, **kwargs):
    kwargs.pop("metadata_errors", None)
    return _original_av_open(*args, **kwargs)
av.open = _patched_av_open
# ------------------------------------------

import argparse
import subprocess
import torch
import torchaudio
import soundfile as _sf

# --- torchaudio>=2.9 removed built-in I/O (needs torchcodec); route through soundfile ---
def _ta_load(path, *args, **kwargs):
    data, sr = _sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr

def _ta_save(path, src, sample_rate, *args, **kwargs):
    arr = src.detach().cpu().numpy()
    if arr.ndim == 1:
        arr = arr[None, :]

    path_str = str(path)
    if path_str.endswith(".m4a") or path_str.endswith(".mp3"):
        # soundfile can't write m4a/mp3: write temp wav, then convert with ffmpeg
        temp_wav = path_str + ".tmp.wav"
        _sf.write(temp_wav, arr.T, sample_rate)
        subprocess.run(["ffmpeg", "-y", "-i", temp_wav, "-c:a", "aac", "-b:a", "192k", path_str],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
    else:
        _sf.write(path_str, arr.T, sample_rate)

torchaudio.load = _ta_load
torchaudio.save = _ta_save
# ---------------------------------------------------------------------------------------

from faster_whisper import WhisperModel
from TTS.api import TTS

# Whisper language code -> NLLB-200 language code
NLLB_CODES = {
    "en": "eng_Latn", "zh": "zho_Hans", "yue": "yue_Hant", "ko": "kor_Hang",
    "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn", "it": "ita_Latn",
    "pt": "por_Latn", "ru": "rus_Cyrl", "nl": "nld_Latn", "pl": "pol_Latn",
    "tr": "tur_Latn", "ar": "arb_Arab", "hi": "hin_Deva", "id": "ind_Latn",
    "vi": "vie_Latn", "th": "tha_Thai",
}
NLLB_MODEL = "facebook/nllb-200-distilled-600M"
TARGET_NLLB = "jpn_Jpan"
TARGET_XTTS = "ja"


def extract_audio(video_path, audio_out_path):
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1", audio_out_path]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Local Offline Translation & Voice Cloning -> Japanese")
    parser.add_argument("--input_video", required=True, help="Path to input video chunk")
    parser.add_argument("--output_audio", required=True, help="Path to output dubbed Japanese audio")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Running local pipeline on device: {device}")

    temp_dir = "pipeline_output/temp_dub_ja"
    os.makedirs(temp_dir, exist_ok=True)
    full_audio = os.path.join(temp_dir, "full_audio.wav")

    print("[*] Step 1: Extracting audio track from video...")
    extract_audio(args.input_video, full_audio)

    print("[*] Step 2: Loading faster-whisper (large-v3) to transcribe the original speech (language auto-detected)...")
    whisper_model = WhisperModel("large-v3", device=device, compute_type="float16")
    segments, info = whisper_model.transcribe(full_audio, task="transcribe", beam_size=5)
    trans_segments = list(segments)
    src_lang = info.language
    print(f"[+] Detected source language: {src_lang} (p={info.language_probability:.2f}); {len(trans_segments)} segments.")

    # Free GPU memory held by Whisper before loading the other models
    del whisper_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # Step 3: translate segment text -> Japanese
    texts = [s.text.strip() for s in trans_segments]
    if src_lang == "ja":
        print("[!] Source audio is already Japanese; skipping translation (voices will be re-cloned from the original).")
        ja_texts = texts
    else:
        if src_lang not in NLLB_CODES:
            print(f"[!] No NLLB mapping for source language '{src_lang}'. Add it to NLLB_CODES in this script.", file=sys.stderr)
            sys.exit(1)
        print(f"[*] Step 3: Translating {src_lang} -> Japanese with {NLLB_MODEL}...")
        from transformers import pipeline
        translator = pipeline(
            "translation", model=NLLB_MODEL,
            src_lang=NLLB_CODES[src_lang], tgt_lang=TARGET_NLLB,
            device=0 if device == "cuda" else -1,
        )
        ja_texts = []
        for t in texts:
            if not t:
                ja_texts.append("")
                continue
            ja_texts.append(translator(t, max_length=256)[0]["translation_text"])
        del translator
        if device == "cuda":
            torch.cuda.empty_cache()

    print("[*] Step 4: Loading Coqui XTTS-v2 for local offline voice cloning...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    print("[*] Step 5: Loading full audio tensor for utterance-level voice cloning...")
    import soundfile as sf
    data, sr = sf.read(full_audio)
    if data.ndim == 1:
        audio_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    else:
        audio_tensor = torch.tensor(data, dtype=torch.float32).T

    probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", args.input_video]
    total_duration = float(subprocess.check_output(probe_cmd).decode().strip())
    final_audio_tensor = torch.zeros(1, int(total_duration * 22050))

    print("[*] Step 6: Synthesizing Japanese voice-cloned audio segments...")
    for i, (seg, text) in enumerate(zip(trans_segments, ja_texts)):
        if not text:
            continue

        # Original audio slice of this utterance is the voice reference
        orig_start_sample = int(seg.start * sr)
        orig_end_sample = int(seg.end * sr)

        # At least 2 seconds for a stable XTTS voice embedding
        min_samples = sr * 2
        if (orig_end_sample - orig_start_sample) < min_samples:
            orig_end_sample = min(audio_tensor.shape[1], orig_start_sample + min_samples)

        ref_segment_tensor = audio_tensor[:, orig_start_sample:orig_end_sample]
        if ref_segment_tensor.numel() == 0 or ref_segment_tensor.abs().max() < 1e-5:
            ref_segment_tensor = audio_tensor[:, :int(sr * 2)]

        utterance_ref_path = os.path.join(temp_dir, f"ref_utterance_{i:04d}.wav")
        torchaudio.save(utterance_ref_path, ref_segment_tensor, sr)

        seg_wav_path = os.path.join(temp_dir, f"seg_{i:04d}.wav")
        try:
            tts.tts_to_file(
                text=text,
                speaker_wav=utterance_ref_path,
                language=TARGET_XTTS,
                file_path=seg_wav_path
            )

            wav_tensor, _wsr = torchaudio.load(seg_wav_path)
            if _wsr != 22050:
                wav_tensor = torchaudio.functional.resample(wav_tensor, _wsr, 22050)
            start_sample = int(seg.start * 22050)
            end_sample = start_sample + wav_tensor.shape[1]

            if end_sample > final_audio_tensor.shape[1]:
                padding = end_sample - final_audio_tensor.shape[1]
                final_audio_tensor = torch.cat([final_audio_tensor, torch.zeros(1, padding)], dim=1)

            final_audio_tensor[0, start_sample:end_sample] = wav_tensor[0, :end_sample - start_sample]
            print(f"    [{seg.start:.2f}s -> {seg.end:.2f}s] '{texts[i]}'  =>  '{text}'")
        except Exception as e:
            print(f"    [!] Error synthesizing segment {i}: {e}")

    print("[*] Step 7: Exporting final dubbed Japanese audio track...")
    torchaudio.save(args.output_audio, final_audio_tensor, 22050)
    print(f"[+] Dubbed Japanese audio saved: {args.output_audio}")


if __name__ == "__main__":
    main()
