"""Federated grokking: library code for grokking under federated averaging."""

import os

# cuBLAS picks its reduction algorithm from the workspace it is given, and with
# CUDA >= 10.2 that choice can differ between two launches of the same GEMM
# unless the workspace is pinned (PyTorch's reproducibility notes name this
# exact variable). Set here, at the package root, so it is in place before any
# CUDA context exists in the driver -- and inherited by the Ray client actors,
# which are forked from a process that has already imported this package.
# setdefault: an operator's explicit choice wins.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
