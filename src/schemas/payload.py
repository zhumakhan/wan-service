from typing import List, Literal

from pydantic import BaseModel, model_validator

from src.schemas.generation_params import GenerationParams


class SingleJob(BaseModel):
    job_id: str
    result_object_key: str
    result_compressed_object_key: str
    prompt: str | None = None


class Payload(BaseModel):
    job_set_id: str
    s3_region: str
    sqs_region: str
    result_queue_url: str
    result_bucket_name: str
    generation_params: GenerationParams
    generations: List[SingleJob]

    @model_validator(mode='after')
    def validate(self):
        return self