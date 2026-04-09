import time
import logging
from datetime import datetime, UTC
import concurrent.futures

import torch.distributed as dist

from src.settings import settings
from src.schemas import Payload
from src.core.pipeline import Pipeline
from src.utils.hosts import get_hostname, get_public_ipv4_ubuntu
from src.exceptions import RetryLimitExceeded, CriticalError, JobAlreadyExists


class MessageProcessor:
    def __init__(self, rank: int, world_size: int, device_id: int):
        self.rank = rank
        self.world_size = world_size

        self.device_id = device_id

        if self.rank == 0:

            self.instance_name = get_hostname()
            self.gpu_type = 'h100'
            self.gpu_count = world_size
            self.gpu_provider = settings.gpu_provider


        self.pipeline = Pipeline(rank, world_size, device_id)


        if self.rank == 0:
            self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=16)
            open("/tmp/ready.flag", "a").close()

    def __call__(self, data: Payload, queue_url: str) -> float:
        try:
            logging.info(f'Received message: {data}')
            start_inference_time = time.perf_counter()
            images = self.pipeline(data)
            end_inference_time = time.perf_counter()
            logging.info(f'Done processing in {end_inference_time - start_inference_time} seconds')
            return end_inference_time - start_inference_time
        except CriticalError as e:
            raise e
        except Exception as e:
            raise e