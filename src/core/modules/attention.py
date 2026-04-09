import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

__all__ = [
    'flash_attention',
    'attention',
]


def flash_attention(
    q,
    k,
    v
):
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1,2),
            k.transpose(1,2),
            v.transpose(1,2)
        ).transpose(1,2)

def attention(
    q,
    k,
    v
):
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1,2),
            k.transpose(1,2),
            v.transpose(1,2)
        ).transpose(1,2)