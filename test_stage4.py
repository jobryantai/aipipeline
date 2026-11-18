import os
import torch
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline

print("Initializing LivePortrait Pipeline on RTX 3090...")
inference_cfg = InferenceConfig()
inference_cfg.flag_use_half_precision = True
inference_cfg.device_id = 0

crop_cfg = CropConfig()
pipeline = LivePortraitPipeline(inference_cfg=inference_cfg, crop_cfg=crop_cfg)
print("LivePortrait Pipeline is active and ready for generation!")
