import io
import base64
import time
import logging
from datetime import datetime, UTC
import concurrent.futures

import torch
import torch.distributed as dist
from PIL import Image

from src.settings import settings
from src.schemas import Payload
from src.core.pipeline import Pipeline
from src.utils.hosts import get_hostname, get_public_ipv4_ubuntu
from src.exceptions import RetryLimitExceeded, CriticalError, JobAlreadyExists


def _encode_images_to_base64(images) -> list[str]:
    if isinstance(images, torch.Tensor):
        images = [images]

    encoded: list[str] = []
    for img in images:
        t = img.detach().cpu().float()
        # WAN T2V outputs (C, T, H, W); plain images may be (C, H, W).
        if t.dim() == 4:
            t = t.permute(1, 2, 3, 0)  # (T, H, W, C)
        elif t.dim() == 3:
            t = t.permute(1, 2, 0).unsqueeze(0)  # (1, H, W, C)
        else:
            raise ValueError(f"Unexpected image tensor shape: {tuple(t.shape)}")

        t = ((t.clamp(-1, 1) + 1) * 127.5).byte().numpy()
        for frame in t:
            buf = io.BytesIO()
            Image.fromarray(frame).save(buf, format="PNG")
            encoded.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return encoded


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

    def __call__(self, data: Payload, queue_url: str) -> dict:
        try:
            logging.info(f'Received message: {data}')
            start_inference_time = time.perf_counter()
            images = self.pipeline(data)
            end_inference_time = time.perf_counter()
            latency = end_inference_time - start_inference_time
            logging.info(f'Done processing in {latency} seconds')

            encoded_images: list[str] = []
            if self.rank == 0:
                encoded_images = _encode_images_to_base64(images)

            return {"latency": latency, "images": encoded_images}
        except CriticalError as e:
            raise e
        except Exception as e:
            raise e