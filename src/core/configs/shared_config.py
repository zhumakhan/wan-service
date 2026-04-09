# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from easydict import EasyDict

#------------------------ Wan shared config ------------------------#
wan_shared_cfg = EasyDict()

# t5
wan_shared_cfg.t5_model = 'umt5_xxl'
wan_shared_cfg.t5_dtype = torch.bfloat16
wan_shared_cfg.text_len = 512

# transformer
wan_shared_cfg.param_dtype = torch.bfloat16

# inference
wan_shared_cfg.num_train_timesteps = 1000
wan_shared_cfg.sample_fps = 16
wan_shared_cfg.sample_neg_prompt = 'Watermarks, screenshot, black borders, overexposed, high saturated, blurred details, style, artwork, painting, picture, worst quality, low quality, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, three legs, plastic face, artificial-looking face, doll-like face, overly smooth skin, ai generated image, synthetic look, artificial skin, oversaturated skin, unrealistic skin texture, journal, text'
