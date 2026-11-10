# stage4_stylized_avatar.py
import os
import json
import torch
from stage1._version import get_version_info

# Import LivePortrait pipeline components (Assuming LivePortrait repo is cloned in your workspace or PYTHONPATH)
try:
    from src.config.inference_config import InferenceConfig
    from src.config.crop_config import CropConfig
    from src.live_portrait_pipeline import LivePortraitPipeline
    LIVE_PORTRAIT_AVAILABLE = True
except ImportError:
    LIVE_PORTRAIT_AVAILABLE = False

def run_stylized_avatar_pipeline(segments_json_path, speaker_refs_dir, output_dir):
    print(f"=== Running {get_version_info()} [Stage 4: RTX 3090 LivePortrait Avatar Animation] ===")
    
    if not torch.cuda.is_available():
        raise EnvironmentRefused("An NVIDIA GPU (RTX 3090) with CUDA support is required for LivePortrait inference.")
    
    # Print GPU specs for verification
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Detected Hardware Acceleration on: {gpu_name} (VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB)")

    # 1. Load diarized segments
    with open(segments_json_path, 'r') as f:
        segments = json.load(f)
        
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Initialize LivePortrait pipeline if available
    pipeline = None
    if LIVE_PORTRAIT_AVAILABLE:
        inference_cfg = InferenceConfig()
        inference_cfg.flag_use_half_precision = True  # FP16 optimization for RTX 3090 speedup
        inference_cfg.device_id = 0
        
        crop_cfg = CropConfig()
        pipeline = LivePortraitPipeline(inference_cfg=inference_cfg, crop_cfg=crop_cfg)
        print("LivePortrait Pipeline initialized successfully.")
    else:
        print("WARNING: LivePortrait package not found in path. Running in mock/stub mode.")

    # 3. Map unique speakers to their consistent stylized reference avatars
    speaker_avatars = {}
    if os.path.exists(speaker_refs_dir):
        for speaker_id in os.listdir(speaker_refs_dir):
            ref_path = os.path.join(speaker_refs_dir, speaker_id)
            if os.path.isfile(ref_path) and ref_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                speaker_avatars[speaker_id] = ref_path

    # 4. Process each segment with identity consistency
    for idx, seg in enumerate(segments):
        speaker = seg.get('speaker', 'speaker_0')
        driving_video = seg.get('video_segment') # Path to segment video file
        
        # Fallback to default reference if specific speaker image is missing
        source_avatar = speaker_avatars.get(speaker, os.path.join(speaker_refs_dir, "default_ref.jpg"))
        
        output_segment_path = os.path.join(output_dir, f"out_seg_{idx}_{speaker}.mp4")
        
        print(f"Processing segment {idx}: Speaker [{speaker}] <- Driving [{driving_video}]")
        render_avatar_segment(pipeline, source_avatar, driving_video, output_segment_path)

    print("Stage 4 Consistent Avatar Animation Complete.")

def render_avatar_segment(pipeline, source_image, driving_video, output_path):
    """
    Executes LivePortrait driving reenactment for a single segment video clip
    using PyTorch and RTX 3090 half-precision acceleration.
    """
    if pipeline is None or not os.path.exists(source_image) or not os.path.exists(driving_video):
        print(f"[Stub] Rendering skipped or files missing for: {output_path}")
        return

    # Construct execution arguments matching LivePortrait pipeline specs
    class Args:
        source = source_image
        driving = driving_video
        output_dir = os.path.dirname(output_path)
        flag_crop_driving_video = True
        flag_stitching = True
        flag_relative_motion = True
        flag_pasteback = True

    try:
        args = Args()
        # Execute LivePortrait inference frame-by-frame mapping
        pipeline.execute(args)
        print(f"Successfully rendered: {output_path}")
    except Exception as e:
        print(f"Error during LivePortrait inference on RTX 3090: {e}")

if __name__ == "__main__":
    run_stylized_avatar_pipeline(
        segments_json_path="segments_diarized.json", 
        speaker_refs_dir="speaker_refs/", 
        output_dir="stage4_output/"
    )
