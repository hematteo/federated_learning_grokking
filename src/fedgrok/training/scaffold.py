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
  - each client receives its own c_i in the same fit config (`scaffold_ci`,
    absent on its first round, when c_i = 0), applies the g - c_i + c
    correction during local steps, and returns its updated model (aggregated
    normally by FedAvg) plus the control-variate delta Δc_i as bytes;
  - ScaffoldStrategy (a thin FedAvg subclass) keeps every c_i keyed by
    PARTITION id, applies the Δc_i, and updates c <- c + (|S|/N) * mean(Δc_i);
    the on_fit_config closure ships the new c and configure_fit attaches each
    client's c_i.

  c_i used to live in a module-level dict inside the Ray actor that ran the
  client. Flower's simulation does not pin a partition to an actor -- the pool
  hands each message to whichever actor is idle -- so from round 2 a partition
  was routinely served by an actor holding no c_i for it (zeros) or a stale
  one. That made the correction wrong and made every SCAFFOLD run
  non-reproducible. The 15 banked anchor SCAFFOLD runs (t3_algorithm_comparison)
  carry this; see RUNS_TODO. Server-side state is exact and deterministic.

Control-variate update is SCAFFOLD's Option II:
    c_i^+ = c_i - c + (x - y_i) / (eta * K)
where x is the round's global model, y_i the client's local model after K local
gradient steps, eta the local lr. Δc_i = c_i^+ - c_i = -c + (x - y_i)/(eta K).

Sanity property (tested): at round 1, c = c_i = 0, so the correction is zero and
the round reduces exactly to FedAvg; the control variates only bite from round 2.
"""

import numpy as np
import torch
from flwr.common import FitIns
from flwr.server.strategy import FedAvg


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


def client_cv_update_option1(grad_mean, client_cv):
    """Option I: c_i^+ = mean local gradient; Δc_i = c_i^+ - c_i.

    `grad_mean` is the average of the RAW gradients (before the SCAFFOLD
    correction is added) over the client's local steps this round. Under plain
    GD at momentum 0 this equals Option II's (x - y_i)/(lr K) exactly, because
    y_i = x - lr * sum(g_t - c_i + c) makes (x - y_i)/(lr K) = mean(g_t) - c_i + c
    and Option II's c_i - c + that = mean(g_t). Under AdamW the two differ and
    only this one is unbiased.
    """
    new_cv = [g.astype(np.float32, copy=True) for g in grad_mean]
    delta = [n - ci for n, ci in zip(new_cv, client_cv)]
    return new_cv, delta


class _SortedResultsMixin:
    """Aggregate client results in client-id order, not arrival order.

    Flower's aggregate_inplace sums the results list front to back, and the
    list is ordered by whichever Ray actor answered first. Ten float32 layers
    summed in different orders differ by ~1.5 ulp per round; on setups near a
    critical point that is enough to flip whether a seed memorises (RESULTS 23).
    Sorting by cid makes the sum a pure function of the client updates. Applied
    to every strategy through _build_strategy; FedConfig.aggregation_order
    = "arrival" bypasses it.
    """

    sort_results = True

    def aggregate_fit(self, server_round, results, failures):
        if self.sort_results:
            results = sorted(results, key=_result_key)
        return super().aggregate_fit(server_round, results, failures)


def _result_key(result):
    """Sort by the PARTITION the client trained, not by Flower's cid.

    In the simulation engine a cid is the node id, and node ids are drawn at
    random per run -- so an order by cid is fixed within a run and different
    between runs, which is exactly the run-to-run noise being removed. The
    client reports its partition id in the fit metrics; a result without one
    (a stub in a test) falls back to the cid, numerically where possible.
    """
    client, fit_res = result
    pid = fit_res.metrics.get("partition_id") if fit_res.metrics else None
    if pid is not None:
        return (0, int(pid), "")
    text = str(client.cid)
    return (1, int(text), "") if text.isdigit() else (2, 0, text)


class ScaffoldStrategy(_SortedResultsMixin, FedAvg):
    """FedAvg model aggregation + server-side control-variate state.

    The model is aggregated exactly as FedAvg (clients return their corrected
    local models). Every c_i lives here, keyed by partition id, and is shipped
    to its client in configure_fit; the server variate c is updated from the
    Δc_i clients report in their fit metrics, c <- c + (participating/total) *
    mean(Δc_i), and read by the on_fit_config closure through `server_cv_box[0]`.
    """

    def __init__(self, *args, server_cv_box, num_total_clients, param_shapes, **kwargs):
        super().__init__(*args, **kwargs)
        self._cv_box = server_cv_box
        self._num_total = num_total_clients
        self._shapes = param_shapes
        self._client_cv = {}          # partition id -> c_i
        self._cid_to_partition = {}   # Flower cid (node id) -> partition id

    def configure_fit(self, server_round, parameters, client_manager):
        # FedAvg hands every client the SAME FitIns object; copy the config
        # before attaching a per-client value.
        out = []
        for client, fit_ins in super().configure_fit(server_round, parameters, client_manager):
            pid = self._cid_to_partition.get(client.cid)
            if pid is not None and pid in self._client_cv:
                fit_ins = FitIns(fit_ins.parameters,
                                 {**fit_ins.config,
                                  "scaffold_ci": _cv_bytes(self._client_cv[pid])})
            out.append((client, fit_ins))
        return out

    def aggregate_fit(self, server_round, results, failures):
        if self.sort_results:
            results = sorted(results, key=_result_key)
        # Standard FedAvg model aggregation first.
        aggregated_params, metrics = super().aggregate_fit(server_round, results, failures)

        # Then apply each client's Δc_i to its stored c_i, and the mean to c.
        deltas = []
        for client, fit_res in results:
            dc = fit_res.metrics.get("scaffold_dc")
            pid = fit_res.metrics.get("partition_id")
            if dc is None or pid is None:
                continue
            pid = int(pid)
            self._cid_to_partition[client.cid] = pid
            delta = _cv_from_bytes(dc, self._shapes)
            prev = self._client_cv.get(pid) or zeros_like_params(delta)
            self._client_cv[pid] = [c + d for c, d in zip(prev, delta)]
            deltas.append(delta)
        if deltas:
            mean_dc = [np.mean([d[j] for d in deltas], axis=0)
                       for j in range(len(self._shapes))]
            frac = len(deltas) / max(1, self._num_total)
            self._cv_box[0] = [c + frac * dc for c, dc in zip(self._cv_box[0], mean_dc)]

        return aggregated_params, metrics
