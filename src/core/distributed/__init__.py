import os
import logging
import traceback
from typing import Any, Callable

import torch
import torch.distributed as dist


def setup_distributed_environment():
    """Initialize distributed environment and return rank, world_size, and device."""
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        torch.cuda.current_device()
        torch.tensor([0.0]).cuda()
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            device_id=torch.device(f"cuda:{local_rank}")
        )
        
        try:
            from xfuser.core.distributed import (initialize_model_parallel,
                                               init_distributed_environment)
            
            if rank == 0:
                logging.info(f"Initializing distributed environment with rank={rank}, world_size={world_size}")
            init_distributed_environment(rank=rank, world_size=world_size)
            
            if rank == 0:
                logging.info(f"Initializing model parallel with ulysses_size=8, ring_size=1, world_size={world_size}")
            
            initialize_model_parallel(
                sequence_parallel_degree=world_size,
                ring_degree=1,
                ulysses_degree=world_size,
            )
        except Exception as e:
            if rank == 0:
                logging.error(f"Error initializing model parallel: {e}")
                logging.error(f"Traceback: {traceback.format_exc()}")
            raise e
    return rank, world_size, local_rank
