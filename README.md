# Grokking under federated learning

Does **grokking** — a model memorising its training set, looking like a failure for
a long time, then abruptly generalising — survive when training is split across many
clients that average their weights?

Reproduces Gromov (2023) on modular arithmetic and extends it across six setups and
the FedAvg family, using Flower.

> **The framing that shapes everything here.** FedAvg at one local epoch with `n_k/n`
> weighting is an *algebraic identity* with centralized GD, proven in
> `tests/test_fedavg_identity.py` and observed in the wild. So "federation preserves
> grokking" is not a result — it is forced. The load-bearing axes are therefore local
> epochs `E`, client count `K`, and **how the data is partitioned**, not the
> architecture list.

## Status

Active work is on branch **`v2-multisetup`**. `main` is the frozen single-setup study
(tag `v1-single-setup`) and is 60+ commits behind; everything below describes v2.

- **`RUNS_TODO.md`** — what still needs running, and what was decided against. Start here.
- **`PROGRESS.md`** — what is built, how to run it, and the reasoning worth not re-deriving.
- **`RESULTS.md`** — every measured number, with the run data behind it.
- Ground truth is `results/data/runs_v2.csv`, not the prose. The docs lag the data.

## Install

```bash
python3.10 -m venv venv                 # pyproject requires >=3.10
venv/bin/pip install -e ".[dev]"
```

Versions are pinned in `pyproject.toml`; `torch` and `torchvision` are a pinned *pair*,
and `flwr>=1.27` is required for the `run_simulation` API. Runs assume CUDA — the
launcher pins one GPU per subprocess via `CUDA_VISIBLE_DEVICES`.

**Hardware.** This project ran on an 8× L4 box until 2026-08-17 and now runs on a
single **RTX 3080 Laptop (8 GB), 16 cores**. Multi-GPU pools in `scripts/run_*.sh`
are records of what was run, not runnable commands. Banked `wall_s` figures come
from the L4 and still transfer: the work is orchestration-bound, so per-run wall is
about the same here (measured, §Concurrency in `PROGRESS.md`).

## Running a sweep

Experiments are declared as **manifests** — JSONL files of run specs — rather than as
CLI invocations. A run's id is a content hash of its config, which is what makes
resume free and makes duplicate work across manifests impossible.

```bash
venv/bin/python scripts/build_manifests.py                    # (re)generate manifests/
venv/bin/python scripts/validate_manifest.py manifests/<name>.jsonl
venv/bin/python scripts/launch_sweep.py manifests/<name>.jsonl --gpus 0 --per-gpu 4
venv/bin/python scripts/collect_runs.py                       # -> results/data/runs_v2.csv
venv/bin/python scripts/summarize_runs.py results/data/runs_v2.csv --group setup,num_clients
```

Resume is automatic: re-running a manifest executes only what is missing. A long sweep
should be detached, or it dies with its shell:

```bash
setsid nohup venv/bin/python -u scripts/launch_sweep.py manifests/<name>.jsonl \
    --gpus 0 --per-gpu 4 > logs/sweeps/<name>.log 2>&1 < /dev/null &
```

Sweep logs go in `logs/sweeps/`, not `logs/` — the latter holds v1 experiment logs and
is harvested non-recursively into `results/data/runs.csv`.

**Every manifest builder in `scripts/build_manifests.py` carries its decision rule in
its docstring.** Read it before reading that sweep's results, not after.

## Two things that will bite you

**Budget as `t_memo(K) + delay`, never as a multiple of the centralized T_grok.**
Federation slows *memorisation* steeply with client count while leaving the *delay*
roughly flat. A centralized-anchored budget under-provisions exactly the high-`K`
cells you care about. Six boundaries in this project were manufactured by getting
this wrong — every headline failure it has reported turned out, on re-measurement, to
be a clock running out. `t_memo` is recorded next to `t_grok` for this reason.

**Wall-clock is ~99% orchestration, not compute.** 50,000 centralized gradient steps
take under a minute; the identical arithmetic federated across 5 clients takes ~22.
Cost scales with *client count*, not training length, so order manifests
longest-job-first and do not size the work in GPU-hours.

**VRAM is CUDA contexts, not tensors.** Every client is a separate Ray actor
*process*, and each pays a full CUDA context — ~226 MiB measured, before any model
or data exists. Setup B's transformer is 0.9 MB and a K=50 client's shard is
0.09 MB, so a K=50 run wants ~180 MB of actual memory and ~11 GB of context. VRAM
therefore scales with **how many clients run at once**, and the model size barely
enters. Two env vars control it, neither of which changes what is computed:

```bash
FEDGROK_GPU_CLIENT_CAP=8   # at most 8 clients hold a context at once
FEDGROK_CLIENT_CPU=1       # clients train on CPU; server still evaluates on GPU
```

FedAvg is synchronous, so running K clients in waves of N is the same computation —
verified bit-identical on accuracy, with losses agreeing to 4e-9. They are env vars
rather than config fields because run ids are content hashes of the spec: an added
field would re-id every banked run.

On the 8 GB card, `--per-gpu 4` is the measured working default up to K=20 — four
concurrent runs cost nothing per-run against the banked single-slot L4 rate. Past
K=20, cap the clients rather than the runs: 50 client processes on 16 cores contend
rather than compute, and capping to 8 is 2.9× faster *and* halves peak VRAM.

## Layout

```
src/fedgrok/
  core/        Config + FedConfig dataclasses, model/loss registry, guards
  data/        dataset registry — modular arithmetic, S_n composition, MNIST-1k;
               partitioning (iid, operand, target, dirichlet, label_block, coset)
  models/      GrokNet (quadratic MLP), Nanda transformer, generic ReLU MLP
  training/    centralized loop, Flower/Ray federated loop, SCAFFOLD, runner
  metrics/     Fourier/IPR, S_n isotypic decomposition, exact quadratic-circuit
               split, per-setup mechanistic probes
  analysis/    T_grok / t_memo detection, censored-survival statistics
  manifest.py  spec -> config, content-hash run ids, grid expansion
  run.py       single-run entry point, atomic result JSON

scripts/       build_manifests, validate_manifest, launch_sweep, collect_runs,
               summarize_runs, backfill_runs, harvest_logs
scripts/plotting/   result-row consumers; grok_curves.py builds a self-contained
                    HTML page of train/test curves with the delay band marked
manifests/     the declared experiments
results/data/  runs_v2.csv + runs.csv — the committed evidentiary base
tests/         the suite, ~9 min including Flower/Ray integration
```

`run_experiment.py` and `experiments/` are the **v1 orchestration surface**, kept
because v1's 870 runs are still cited. They predate the manifest system and are not
used by anything in `src/`, `scripts/` or `tests/`.

## Data release

Model checkpoints and per-client weights (40 GB, 664 runs) are on Hugging Face at
[FedGrok/fedgrok-checkpoints](https://huggingface.co/datasets/FedGrok/fedgrok-checkpoints),
gated with automatic approval. One uncompressed tar per campaign `group`, filed
under a folder named for the paper axis it supports (`num_clients/`, `local_epochs/`,
`participation/`, `heterogeneity/`, `mechanism/`); the `group` column of `runs_v2.csv`
is the join key, and the dataset's `MANIFEST.csv` maps each group to its path and figure. Runs with `checkpoint_every = 0`
(the strategy comparison, setup A's local-epochs ladder, most centralised
anchors) have histories here but no weights anywhere. `scripts/package_checkpoints.py`
builds a campaign's archives; `scripts/merge_checkpoint_release.py` extends the
published root files without touching existing archives.

## Statistics

Runs that do not grok within budget are **right-censored**, not dropped and not
recorded as infinity. Headline numbers are Kaplan–Meier medians with bootstrap 95%
CIs over seeds, alongside the fraction of seeds that grokked — which is the order
parameter, and the honest headline whenever a cell is partly censored.

The grok threshold is a **dataset property**, not a constant (95% modular, 90% MNIST,
85% S₅), and is stored per run: a `t_grok` only means something next to the bar it was
measured at.

## Tests

```bash
venv/bin/python -m pytest tests/ -q          # full suite, ~9 min
venv/bin/python -m pytest tests/ -q -k "not Fed and not fed and not integration"   # ~45 s
```

The exact passing count lives in `PROGRESS.md`; the slow half is Flower/Ray simulation.

## References

- Gromov, A. (2023). *Grokking modular arithmetic.* arXiv:2301.02679
- Nanda et al. (2023). *Progress measures for grokking via mechanistic interpretability.*
- Liu et al. (2023). *Omnigrok: grokking beyond algorithmic data.*
- Stander et al. (2023). *Grokking group multiplication with cosets.*
