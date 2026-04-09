from typing import Optional, List

from pydantic import BaseModel, model_validator

from src.exceptions import ValidationError


class Lora(BaseModel):
    path: str
    strength: float
    rank: int = 16

    @model_validator(mode='after')
    def validate(self):
        if self.strength < 0.0 or self.strength > 1.0:
            raise ValidationError("strength must be between 0.0 and 1.0")
        
        if self.rank not in [16, 32]:
            raise ValidationError("rank must be one of [16, 32]")

        if 'user_loras' in self.path:
            self.rank = 32
        
        return self


class GenerationParams(BaseModel):
    height: int
    width: int
    
    prompt: str = ''
    prompts: List[str] = []
    n_prompt: Optional[str] = ''
    
    sample_shift: float = 3.0
    sample_solver: str = 'unipc'
    sample_steps: int = 50
    sample_guide_scale: float = 4.0
    base_seed: int = 42
    
    batch_size: int = 1
    interactive_steps: Optional[List[int]] = []
    
    lora: Optional[str] = None
    lora_strength: float = 1.0
    lora_rank: int = 16

    loras: Optional[List[Lora]] = None
    
    ret_steps: bool = True

    use_apg: bool = False
    apg_momentum: float = 0.0
    apg_eta: float = 0.0
    apg_norm_threshold: float = 0.0

    @model_validator(mode='after')
    def validate(self):
        if not self.prompt and not self.prompts:
            raise ValidationError("prompt or prompts must be provided")
        
        if self.height % 16 != 0 or self.width % 16 != 0:
            raise ValidationError("height and width must be divisible by 16")

        if self.height > 2048 or self.width > 2048:
            raise ValidationError("height and width must be less than 2048")
        
        if self.batch_size is None or self.batch_size > 6 or self.batch_size < 1:
            raise ValidationError("batch_size must be between 1 and 4")
        
        if self.interactive_steps is None:
            self.interactive_steps = []

        for step in self.interactive_steps:
            if step is None or step < 0 or step > self.sample_steps:
                raise ValidationError("interactive_steps must be between 0 and sample_steps")

        if self.lora_strength and (self.lora_strength < 0.0 or self.lora_strength > 1.0):
            raise ValidationError("lora_strength must be between 0.0 and 1.0")

        if self.n_prompt is None:
            self.n_prompt = ''


        if self.use_apg:
            if self.apg_eta is None or self.apg_eta < 0.0 or self.apg_eta > 1.0:
                raise ValidationError("apg_eta must be a float between 0.0 and 1.0")

        return self

    def to_kwargs(self) -> dict:
        return {
            "input_prompt": self.prompts if self.prompts else [
                self.prompt for _ in range(self.batch_size)
            ],
            "size": (self.width, self.height),
            "frame_num": 1,
            "shift": self.sample_shift,
            "sample_solver": self.sample_solver,
            "sampling_steps": self.sample_steps,
            "guide_scale": self.sample_guide_scale,
            "seed": self.base_seed,
            "offload_model": False,
            "n_prompt": self.n_prompt,
            "batch_size": self.batch_size,
            "use_ret_steps": self.ret_steps,
            "use_apg": self.use_apg,
            "apg_eta": self.apg_eta,
            "apg_momentum": self.apg_momentum,
            "apg_norm_threshold": self.apg_norm_threshold,
        }
