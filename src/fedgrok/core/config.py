"""Experiment configuration for Gromov (2023) 'Grokking modular arithmetic'."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class Config:
    # --- Modular arithmetic task ---
    p: int = 97                          # modulus (prime)
    task: str = "addition"               # one of: addition, subtraction, division,
                                         #   x2_plus_y2, x_plus_y_squared, multiplication,
                                         #   x2_y2_xy, x3_xy2_y

    # --- Data ---
    dataset: str = "modular"             # dataset family (see fedgrok.data.registry)
    alpha: float = 0.5                   # fraction of p^2 dataset used for training
    seed: int = 42                       # random seed for train/test split
    n_train: int = 1000                  # subset size for non-modular datasets (MNIST)
    n_test: int = 5000                   # held-out size for non-modular datasets
    group_n: int = 5                     # n for the symmetric-group S_n dataset (S5 = 120 elems)
    coset_subgroup: str = "s_nm1"        # subgroup for the coset FL partition: s_nm1 (S4) or a_n (A5)

    # --- Architecture ---
    model: str = "groknet"               # architecture family (see fedgrok.core.registry)
    hidden_width: int = 100              # N: width of the single hidden layer
                                         #   (= d_model for the transformer)
    n_layers: int = 3                    # hidden layers for the generic "mlp" model (Omnigrok)
    init_scale: float = 1.0              # init multiplier; >1 is the Omnigrok large-init trick
    n_heads: int = 4                     # transformer attention heads; hidden_width must
                                         #   be divisible by it (Nanda default 4)
    d_mlp: int = 512                     # transformer MLP width (Nanda default 512)
    activation: Literal[
        "quadratic", "relu", "gelu",
        "abs", "quartic"
    ] = "quadratic"

    # --- Training ---
    loss: Literal["mse", "ce"] = "mse"   # mse: one-hot targets (Gromov); ce: class indices
    optimizer: Literal["gd", "adamw"] = "gd"
    lr: float = 50.0                     # learning rate (50 for GD, 1e-4 for AdamW)
    weight_decay: float = 0.0            # explicit weight decay
    momentum: float = 0.0               # momentum for GD (0.0 = vanilla GD)
    batch_size: int = 0                  # 0 = full batch (Gromov default); >0 = minibatch SGD
    epochs: int = 10_000                 # number of epochs (= gradient steps when full-batch)
    log_every: int = 100                 # log metrics every N epochs

    # --- Output ---
    output_dir: str = "results/baselines/centralized"
    save_weights: bool = False           # save final model weights
    checkpoint_every: int = 0            # save checkpoints every N epochs (0 = disabled)

    # NOTE: `apply_adamw_defaults()` and its `_lr_set` / `_wd_set` / `_epochs_set`
    # companions used to live here. They were v1 CLI machinery: argparse set the
    # flags when the user passed --lr / --wd / --epochs, and the method then
    # filled AdamW defaults for whatever was left. Nothing on the v2 path ever
    # called it -- only the tests did -- so it was a safety net that had been
    # detached from the thing it was catching.
    #
    # It also could not simply be wired in. A manifest spec has no "was it set"
    # flag; "set" means "the key is in the dict", so calling the method after
    # build_config would have overwritten every explicitly chosen lr with 1e-4.
    # The check now lives in `fedgrok.manifest.build_config`, which can still see
    # the spec, and it rejects rather than guesses.
