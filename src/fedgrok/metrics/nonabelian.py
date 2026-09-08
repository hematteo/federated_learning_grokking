"""Progress measures for non-abelian group composition (S_n).

The Z_p DFT metrics in metrics.fourier are meaningless on a non-abelian group,
so S_n runs need their own measure. Following Stander et al. (2312.06581), we
measure how much of the model's function is explained by *coset structure*: the
network solves S_n composition via subgroup cosets, so a growing alignment
between its predictions and a coset-based reference is the analogue of the
Fourier circuit forming.

`coset_attribution` returns, for a subgroup H, the fraction of the model's
correct predictions that are consistent with the coset of the true answer —
i.e. how strongly the learned map respects the coset partition of the output.
It rises as the coset circuit forms, so it is a cheap online progress measure.
"""

import numpy as np
import torch


def nonabelian_applicable(cfg) -> bool:
    """True iff this run is over a non-abelian group dataset (S_n)."""
    return getattr(cfg, "dataset", "modular") == "s5"


def coset_attribution(model, x, y_true, cfg) -> dict:
    """Coset-structure progress measures for an S_n model.

    Returns:
        coset_accuracy    fraction of predictions whose *coset* matches the true
                          answer's coset (>= exact accuracy; rises earlier).
        coset_purity      fraction of predictions that are exactly correct among
                          those that at least land in the right coset — how much
                          of the coset-level signal has sharpened to the element.
    """
    from fedgrok.data.groups import coset_labels

    labels = coset_labels(getattr(cfg, "group_n", 5), cfg.coset_subgroup)
    labels_t = torch.as_tensor(labels, device=x.device)

    with torch.no_grad():
        preds = model(x).argmax(dim=1)

    pred_coset = labels_t[preds]
    true_coset = labels_t[y_true]

    in_right_coset = (pred_coset == true_coset)
    exact = (preds == y_true)

    coset_acc = float(in_right_coset.float().mean().item())
    n_in_coset = int(in_right_coset.sum().item())
    coset_purity = float(exact.sum().item()) / n_in_coset if n_in_coset > 0 else 0.0
    return {"coset_accuracy": coset_acc, "coset_purity": coset_purity}
