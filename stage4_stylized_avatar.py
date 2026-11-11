# stage4_stylized_avatar.py
import os
import json
import torch
from types import SimpleNamespace
from stage1._version import get_version_info

from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline

def run_stylized_avatar_pipeline(segments_json_path, speaker_refs_dir, output_dir):
    print(f"=== Running {get_version_info()} [Stage 4: RTX 3090 LivePortrait Avatar Animation] ===")
    
    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA GPU (RTX 3090) with CUDA support is required.")
    
    print(f"Detected Hardware Acceleration on: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB)")

    with open(segments_json_path, 'r') as f:
        segments = json.load(f)
        
    os.makedirs(output_dir, exist_ok=True)
    
    inference_cfg = InferenceConfig()
    inference_cfg.flag_use_half_precision = True  # FP16 for RTX 3090
    inference_cfg.device_id = 0
    
    crop_cfg = CropConfig()
    pipeline = LivePortraitPipeline(inference_cfg=inference_cfg, crop_cfg=crop_cfg)
    print("LivePortrait Pipeline initialized successfully.")

    speaker_avatars = {}
    if os.path.exists(speaker_refs_dir):
        for file in os.listdir(speaker_refs_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                speaker_id = os.path.splitext(file)[0]
                speaker_avatars[speaker_id] = os.path.join(speaker_refs_dir, file)

    for idx, seg in enumerate(segments):
        speaker = seg.get('speaker', 'speaker_0')
        driving_video = seg.get('video_segment')
        source_avatar = speaker_avatars.get(speaker, os.path.join(speaker_refs_dir, "speaker_0.jpg"))
        
        output_segment_path = os.path.join(output_dir, f"out_seg_{idx}_{speaker}.mp4")
        
        print(f"Processing segment {idx}: Speaker [{speaker}] <- Driving [{driving_video}]")
        render_avatar_segment(pipeline, source_avatar, driving_video, output_segment_path)

    print("Stage 4 Consistent Avatar Animation Complete.")

def render_avatar_segment(pipeline, source_image, driving_video, output_path):
    if not os.path.exists(source_image):
        print(f"[Error] Missing source image: {source_image}")
        return
    if not os.path.exists(driving_video):
        print(f"[Error] Missing driving video: {driving_video}")
        return

    try:
        # Wrap parameters in a SimpleNamespace so attribute access works correctly
        args = SimpleNamespace(
            source=source_image,
            driving=driving_video,
            output_dir=os.path.dirname(output_path),
            flag_crop_driving_video=True,
            flag_stitching=True,
            flag_relative_motion=True,
            flag_pasteback=True,
        )
        
        print(f"Executing LivePortrait inference on RTX 3090 (FP16)...")
        pipeline.execute(args)
        print(f"Successfully generated animated avatar output at: {output_path}")
    except Exception as e:
        print(f"[Error] LivePortrait generation failed: {e}")

if __name__ == "__main__":
    run_stylized_avatar_pipeline(
        segments_json_path="segments_diarized.json", 
        speaker_refs_dir="speaker_refs/", 
        output_dir="pipeline_output/stage4_final"
    )
