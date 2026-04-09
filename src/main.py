import time
import traceback
import logging

import torch.distributed as dist

from src.settings import settings
from src.internal.api_server import ApiServer
from src.internal.message_processor import MessageProcessor
from src.core.distributed import setup_distributed_environment
from higgsmeter import enable_tracker

import torch
torch._dynamo.config.cache_size_limit = 64

import re
import sys
import logging
from datetime import UTC, datetime
from src.utils.hosts import get_public_ipv4_ubuntu, get_hostname


def setup_logging(rank):
    if rank == 0:
        ipv4 = get_public_ipv4_ubuntu()
        hostname = get_hostname()

        stream_name = f"{hostname}-{ipv4}-{datetime.now(UTC).isoformat()}"
        stream_name = re.sub(r"[^a-zA-Z0-9._/-]", "-", stream_name)

        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[
                logging.StreamHandler(stream=sys.stdout),
            ],
            force=True
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[
                logging.StreamHandler(stream=sys.stdout)
            ],
            force=True
        )


def main(rank, world_size, device):
    try:
        message_processor = MessageProcessor(
            rank=rank,
            world_size=world_size,
            device_id=device
        )
    except Exception as e:
        if rank == 0:
            logging.error(f"Healthcheck (warmup) failed: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")

        time.sleep(5)
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        
        raise e
    
    api_server = ApiServer(
        message_processor=message_processor,
        rank=rank,
        host="0.0.0.0",
        port=8001,
    )
    api_server.run()


if __name__ == '__main__':
    rank, world_size, device = setup_distributed_environment()
    setup_logging(rank)
    main(rank, world_size, device)
