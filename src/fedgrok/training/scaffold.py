"""SCAFFOLD (Karimireddy et al., 1910.06378) for the grokking study.

SCAFFOLD is the gold-standard *variance-reduction* drift correction: it keeps a
server control variate c and a per-client control variate c_i, and corrects each
local gradient g -> g - c_i + c so that local steps track the global direction
instead of drifting toward the client's local optimum. It provably removes the
heterogeneity term from the convergence rate — so it is the load-bearing test of
"is client drift the mechanism behind the grokking delay?": if grokking still
fails under SCAFFOLD, drift is not the cause.

Adapted to this repo's GrokClient (which returns updated parameters, not deltas)
via the metrics channel rather than niid_bench's parameter-concatenation:

  - the server control variate c is shipped to clients as bytes in the fit
    config each round;
  - each client keeps c_i in a module-level dict (like _client_cache), applies
    the g - c_i + c correction during local steps, and returns its updated model
    (aggregated normally by FedAvg) plus the control-variate delta Δc_i as bytes;
  - ScaffoldStrategy (a thin FedAvg subclass) reads the Δc_i back and updates
    c <- c + (|S|/N) * mean(Δc_i), then the on_fit_config closure ships the new c.

Control-variate update is SCAFFOLD's Option II:
    c_i^+ = c_i - c + (x - y_i) / (eta * K)
where x is the round's global model, y_i the client's local model after K local
gradient steps, eta the local lr. Δc_i = c_i^+ - c_i = -c + (x - y_i)/(eta K).

Sanity property (tested): at round 1, c = c_i = 0, so the correction is zero and
the round reduces exactly to FedAvg; the control variates only bite from round 2.
"""

import numpy as np
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg


# Per-client control variates c_i, keyed like _client_cache. Lives in each Ray
# actor's process and persists across that client's rounds.
_client_cv = {}


def reset_client_cv():
    _client_cv.clear()


def _cv_bytes(cv_list):
    """Serialize a list of ndarrays to bytes (shapes are implied by the model)."""
    return b"".join(c.astype(np.float32).tobytes() for c in cv_list)


def _cv_from_bytes(buf, shapes):
    """Inverse of _cv_bytes given the parameter shapes."""
    out, offset = [], 0
    for shape in shapes:
        n = int(np.prod(shape))
        chunk = np.frombuffer(buf, dtype=np.float32, count=n, offset=offset * 4)
        out.append(chunk.reshape(shape).copy())
        offset += n
    return out


def zeros_like_params(ndarrays):
    return [np.zeros_like(a) for a in ndarrays]


def get_client_cv(key, template):
    """Return this client's c_i (zeros on first sight)."""
    if key not in _client_cv:
        _client_cv[key] = zeros_like_params(template)
    return _client_cv[key]


def set_client_cv(key, cv):
    _client_cv[key] = cv


def apply_correction(model, server_cv, client_cv):
    """Add (c - c_i) to each parameter's gradient in place (SCAFFOLD correction).

    Call after loss.backward(), before optimizer.step().

    ORDERING. The control variates are built from the weight vectors Flower
    exchanges, which come from `_model_to_ndarrays` and are therefore in
    `state_dict()` order; this walks `parameters()` order. The two agree only for
    a module with no buffers -- `state_dict()` yields parameters then buffers per
    module, `parameters()` yields parameters alone. Every model here (GrokNet,
    GrokFormer, MLP) is buffer-free, so they agree today. The moment one gains a
    mask, a running statistic or a positional cache they would silently
    misalign: the correction would land on the wrong tensors, SCAFFOLD would
    still run, and the "is drift the mechanism?" arm this exists to answer would
    return plausible numbers computed from garbage. Hence the check.
    """
    params = list(model.parameters())
    if len(params) != len(server_cv) or len(params) != len(client_cv):
        raise ValueError(
            f"SCAFFOLD control-variate misalignment: {len(params)} parameters "
            f"against {len(server_cv)} server / {len(client_cv)} client "
            f"variates. The variates follow state_dict() order and this follows "
            f"parameters() order; they diverge once the model has buffers."
        )
    for param, c, ci in zip(params, server_cv, client_cv):
        if param.grad is not None:
            if tuple(param.shape) != tuple(c.shape):
                raise ValueError(
                    f"SCAFFOLD control-variate shape mismatch: parameter "
                    f"{tuple(param.shape)} against variate {tuple(c.shape)}."
                )
            param.grad.add_(torch.from_numpy(c - ci).to(param.grad.device))


def client_cv_update(x_ndarrays, y_ndarrays, server_cv, client_cv, lr, n_steps):
    """New client control variate and the delta to send the server.

    c_i^+ = c_i - c + (x - y_i)/(lr * n_steps);  Δc_i = c_i^+ - c_i.
    """
    new_cv, delta = [], []
    scale = 1.0 / (lr * max(1, n_steps))
    for x, y, c, ci in zip(x_ndarrays, y_ndarrays, server_cv, client_cv):
        ci_new = ci - c + scale * (x - y)
        new_cv.append(ci_new)
        delta.append(ci_new - ci)
    return new_cv, delta


class ScaffoldStrategy(FedAvg):
    """FedAvg model aggregation + server control-variate maintenance.

    The model is aggregated exactly as FedAvg (clients return their corrected
    local models). The control variate is updated from the Δc_i clients report
    in their fit metrics: c <- c + (participating/total) * mean(Δc_i). The shared
    `server_cv_box[0]` is read by the on_fit_config closure to ship c to clients.
    """

    def __init__(self, *args, server_cv_box, num_total_clients, param_shapes, **kwargs):
        super().__init__(*args, **kwargs)
        self._cv_box = server_cv_box
        self._num_total = num_total_clients
        self._shapes = param_shapes

    def aggregate_fit(self, server_round, results, failures):
        # Standard FedAvg model aggregation first.
        aggregated_params, metrics = super().aggregate_fit(server_round, results, failures)

        # Then update the server control variate from the reported Δc_i.
        deltas = []
        for _client, fit_res in results:
            dc = fit_res.metrics.get("scaffold_dc")
            if dc is not None:
                deltas.append(_cv_from_bytes(dc, self._shapes))
        if deltas:
            mean_dc = [np.mean([d[j] for d in deltas], axis=0)
                       for j in range(len(self._shapes))]
            frac = len(deltas) / max(1, self._num_total)
            self._cv_box[0] = [c + frac * dc for c, dc in zip(self._cv_box[0], mean_dc)]

        return aggregated_params, metrics
