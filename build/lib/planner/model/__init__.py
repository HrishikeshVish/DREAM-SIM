import torch

from .senevam import SeNeVAMLightningModule

# NOTE: By default, uses TensorFloat32 for matrix multiplication on devices
# with tensor cores available to accelerate the performance of the model.
torch.set_float32_matmul_precision("high")

__all__ = ["SeNeVAMLightningModule"]
