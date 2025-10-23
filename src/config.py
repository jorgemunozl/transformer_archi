import pydantic
import torch.nn as nn
import torch


class Config(pydantic.BaseModel):
    n_layer: int = 12
    block_size: int = 1024  # block size, which is the context
    vocab_size: int = 50257
    n_head: int = 12
    n_embd: int = 768
