"""Shim for torch_scatter using torch_geometric or native PyTorch ops.

Install this as a module at /usr/local/lib/python3.12/site-packages/torch_scatter/__init__.py
to satisfy imports from older fairchem forks that depend on torch_scatter.
"""

import torch


def scatter(src, index, dim=-1, out=None, dim_size=None, fill_value=0, reduce="sum"):
    """Replacement for torch_scatter.scatter using native PyTorch."""
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0

    size = list(src.size())
    size[dim] = dim_size

    if out is None:
        if reduce == "sum" or reduce == "add":
            out = src.new_zeros(size)
        elif reduce == "mean":
            out = src.new_zeros(size)
        elif reduce == "min":
            out = src.new_full(size, float("inf"))
        elif reduce == "max":
            out = src.new_full(size, float("-inf"))
        else:
            out = src.new_zeros(size)

    if reduce == "sum" or reduce == "add":
        return out.scatter_add_(dim, index.expand_as(src), src)
    elif reduce == "mean":
        count = src.new_zeros(size)
        ones = src.new_ones(src.size())
        count.scatter_add_(dim, index.expand_as(src), ones)
        out.scatter_add_(dim, index.expand_as(src), src)
        count = count.clamp(min=1)
        return out / count
    elif reduce == "min":
        return out.scatter_reduce_(dim, index.expand_as(src), src, reduce="amin")
    elif reduce == "max":
        return out.scatter_reduce_(dim, index.expand_as(src), src, reduce="amax")
    else:
        return out.scatter_add_(dim, index.expand_as(src), src)


def segment_coo(src, index, out=None, dim_size=None, reduce="sum"):
    """Replacement for torch_scatter.segment_coo."""
    return scatter(src, index, dim=0, out=out, dim_size=dim_size, reduce=reduce)


def segment_csr(src, indptr, out=None, reduce="sum"):
    """Replacement for torch_scatter.segment_csr using scatter."""
    # Convert CSR indptr to COO index
    sizes = indptr[1:] - indptr[:-1]
    index = torch.repeat_interleave(torch.arange(len(sizes), device=src.device), sizes)
    return scatter(src, index, dim=0, out=out, dim_size=len(sizes), reduce=reduce)
