#!/usr/bin/env python3
import os
import argparse
from stage1._version import get_version_info
from stage4_stylized_avatar import run_stylized_avatar_pipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Master Pipeline Orchestrator for AI Video Dubbing & Avatars")
    parser.add_argument("--input_video", type=str, required=True, help="Path to input source video (e.g., sample_clip.mp4)")
    parser.add_argument("--output_dir", type=str, default="pipeline_output", help="Root directory for outputs")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"==================================================")
    print(f" Starting {get_version_info()} Execution")
    print(f" Target Input Video: {args.input_video}")
    print(f"==================================================")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- Stage 1 & 2: Diarization & Voice Extraction ---
    print("\n[Orchestrator] Executing Stage 1 & 2: Diarization & Audio Extraction...")
    # TODO: Call your stage 1 & 2 modules here when ready
    
    # --- Stage 3: TTS & Final Dubbed Video Generation ---
    print("\n[Orchestrator] Executing Stage 3: TTS & Generating final_dubbed_video.mp4...")
    # TODO: Call your stage 3 module here to produce stage1/final_dubbed_video.mp4
    
    # --- Stage 4: Consistent Stylized Avatar Animation ---
    print("\n[Orchestrator] Executing Stage 4: Consistent Stylized Avatar Animation...")
    segments_json = "segments_diarized.json" 
    speaker_refs = "speaker_refs/"
    stage4_out = os.path.join(args.output_dir, "stage4_final")
    
    if os.path.exists(segments_json):
        run_stylized_avatar_pipeline(segments_json, speaker_refs, stage4_out)
    else:
        print(f"[Orchestrator Warning] Diarization segments json not found. Skipping Stage 4.")

    print(f"\n==================================================")
    print(f" Pipeline Execution Complete! Check '{args.output_dir}'")
    print(f"==================================================")

if __name__ == "__main__":
    main()
