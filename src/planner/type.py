import os
from typing import Optional, Tuple, Union

import numpy
import torch

# Type aliases
Array = Union[numpy.ndarray, torch.Tensor]
DType = Union[numpy.dtype, torch.dtype]
PathLike = Union[os.PathLike, str]
OptArray = Optional[Array]

# Commonly used tuple of integers representing the size of a tensor
size_t_any = Tuple[int, ...]
size_t_2 = Tuple[int, int]
size_t_3 = Tuple[int, int, int]
