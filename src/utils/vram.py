import torch
import logging


def has_sufficient_vram(device_id: int, required_gb: int = 100) -> bool:
    """
    Check if the specified GPU device has sufficient VRAM.
    
    Args:
        device_id (int): The GPU device ID to check
        required_gb (int): Required VRAM in gigabytes (default: 100GB)
    
    Returns:
        bool: True if device has sufficient VRAM, False otherwise
    """
    try:
        total_memory = torch.cuda.get_device_properties(device_id).total_memory
        total_memory_gb = total_memory / (1024 ** 3)
        return total_memory_gb >= required_gb
    except Exception as e:
        logging.error(f"Error checking VRAM for device {device_id}: {str(e)}")
        return False
