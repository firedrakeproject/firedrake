try:
    import torch
    del torch
except ImportError:
    raise ImportError("PyTorch is not installed and is required to use the FiredrakeTorchOperator.")

from firedrake.ml.pytorch.pytorch_custom_operator import torch_operator   # noqa: F401
