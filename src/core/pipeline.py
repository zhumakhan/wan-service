from functools import partial
import logging
import os

import torch
import torch.distributed as dist

from src.schemas import Payload
from src.core.text2video import WanT2V
from src.core.configs import WAN_CONFIGS
from src.exceptions import CriticalError
from src.core.distributed.fsdp import shard_model
from src.core.modules.t5 import T5EncoderModel
from src.core.modules.vae import WanVAE

from higgsmeter import track_state, AppState, state_tracker


class Pipeline:
    @track_state(AppState.WARMUP, operation='pipeline_initialization')
    def __init__(self, rank, world_size, device_id):
        self.rank = rank
        self.world_size = world_size
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}")

        if self.rank == 0:
            logging.info(
                f"Pipeline initializing on rank {self.rank} with world size {self.world_size} and device {self.device_id}"
            )
        
        config=WAN_CONFIGS["t2v-14B"]
        checkpoint_dir="/app/models/Wan2.1-T2V-14B"
        t5_fsdp = self.world_size > 1
        shard_fn = partial(shard_model, device_id=device_id)
        
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae = WanVAE(
            self.rank,
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        self.wan_t2v = WanT2V(
            config=WAN_CONFIGS["t2v-14B"],
            checkpoint_dir="/app/models/Wan2.1-T2V-14B",
            ckpt='/app/models/wan_bf16.safetensors',
            device_id=device_id,
            rank=rank,
            t5_fsdp=self.world_size > 1,
            dit_fsdp=False,
            use_usp=self.world_size > 1,
            t5_cpu=False,
        )
        self.wan_t2v.text_encoder = self.text_encoder
        self.wan_t2v.vae = self.vae
        
        self._warmup()
    
    
    def wan(self, data: Payload) -> torch.Tensor:    
        if self.rank == 0:
            logging.info(f"Generating...")

        try:
            kwargs = data.generation_params.to_kwargs()
            prompts = [gen.prompt for gen in data.generations]
            if None not in prompts:
                kwargs['input_prompt'] = prompts
            
            with state_tracker.scope(AppState.GPU, operation="inference"):
                images = self.wan_t2v.generate(
                    **kwargs,
                )
            return images
        except Exception as e:
            raise CriticalError(str(e))

    
    def __call__(self, data: Payload) -> torch.Tensor:
        return self.wan(data)
    
          
    def _warmup(self) -> None:
        if self.rank == 0:
            logging.info("Performing warmup...")
        
        shapes = [
            # [1152, 2048],
            # [2048, 1152],
            # [2048, 1536],
            # [1536, 2048],
            # [2048, 2048],
            # [1344, 2016],
            # [2016, 1344],
            [960, 1200],
            # [1536, 1536],
            # [1536, 1152],
            [1696, 960],
            # [1152, 1536],
            # [1120, 1680],
            # [1220, 980],
        ]
        batch_sizes = [4]
        ranks = [0]
        lora_paths = [
            None,
        ]
        
        for rank, lora_path in zip(ranks, lora_paths):
            
            from itertools import product
            
            for batch_size, shape in product(batch_sizes, shapes):
                _ = self.wan_t2v.generate(
                    "A humanoid robot dancing",
                    size=shape,
                    frame_num=1,
                    batch_size=batch_size,
                    shift=5.0,
                    sample_solver="unipc",
                    sampling_steps=1,
                    guide_scale=5.0,
                    seed=42,
                    offload_model=False,
                    teacache_thresh=0.0,
                )

        if self.rank == 0:
            logging.info("Warmup inference completed.")

        if dist.is_initialized():
            dist.barrier()
