"""
This module contains the AdaptiveTransferManager used to perform
transfer operations on AdaptiveMeshHierarchies
"""
import warnings
from firedrake.mg.embedded import TransferManager

__all__ = ("AdaptiveTransferManager",)


def AdaptiveTransferManager(*args, **kwargs):
    """
    TransferManager for adaptively refined mesh hierarchies
    """
    warnings.warn(
        "The ``AdaptiveTransferManager`` class is deprecated and will be removed in a future release. "
        "Please use the ``TransferManager`` class instead.", FutureWarning
    )
    return TransferManager(*args, **kwargs)
