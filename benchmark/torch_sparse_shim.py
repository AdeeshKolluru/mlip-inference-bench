"""Minimal shim for torch_sparse.SparseTensor.

Only provides SparseTensor to satisfy imports during setup_imports().
The actual EquiformerV3 model doesn't use torch_sparse.
"""

import torch


class SparseTensor:
    """Minimal SparseTensor placeholder."""

    def __init__(self, row=None, col=None, value=None, sparse_sizes=None, **kwargs):
        self.row_indices = row
        self.col_indices = col
        self.values = value
        self.sparse_sizes = sparse_sizes
