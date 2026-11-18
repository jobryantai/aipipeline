def generate_tts(en_text_path, output_audio_path):
    print("[2/2] Generating English voice via XTTS-v2...")
    with open(en_text_path, "r", encoding="utf-8") as f:
        en_text = f.read().strip()
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize XTTS-v2 model locally on your RTX 3070
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    
    # Safely retrieve a default built-in speaker from the inner model
    try:
        speakers = list(tts.synthesizer.tts_model.speaker_manager.speakers.keys())
        speaker_name = speakers[0] if speakers else "Clarissa Zarate"
    except Exception:
        speaker_name = "Clarissa Zarate"
        
    print(f"Using built-in speaker profile: {speaker_name}")
    
    # Generate English speech
    tts.tts_to_file(
        text=en_text,
        file_path=output_audio_path,
        speaker=speaker_name,
        language="en"
    )
    print(f"TTS complete! English audio saved to {output_audio_path}")
