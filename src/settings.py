import os
from enum import Enum
from typing import List
from dataclasses import dataclass, field

from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
    gpu_provider: str = os.getenv('GPU_PROVIDER', 'UNKNOWN')


settings = Settings()
