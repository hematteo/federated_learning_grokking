"""Run manifests: a manifest is a list of run specs, one per training run.

A *spec* is a plain dict that fully determines one run: its `mode`
("centralized" or "federated"), the config fields for that mode, and optional
grouping tags (`tier`, `group`, `experiment`, `setting`, `algorithm`) that are
carried through to the results table but do not affect training.

Specs are the atomic unit the launcher schedules. This module builds them
(grid expansion) and turns them into the right dataclass; it has no I/O and no
training dependencies, so it is cheap to test.
"""

import dataclasses
import hashlib
import itertools
import json
import os

from fedgrok.core.config import Config
from fedgrok.core.fed_config import FedConfig


# Keys that tag/group a run for analysis but are not config fields. Tags are
# excluded from the content hash, so adding one to an existing spec is free --
# which is why `setup` (the setup's short name, e.g. "B" for transformer+modular)
# is a tag rather than a config field: it carries identity into the results table
# without re-hashing any banked run.
TAG_KEYS = {"id", "mode", "tier", "group", "experiment", "setting", "algorithm",
            "label", "manifest", "setup",
            # exp2's three-arm structure. `arm` names the condition
            # (cent_full / cent_reduced / fl) and `reduced_from_k` records which
            # K a reduced cell was derived from -- without it a floor run at
            # alpha=0.0026 is just an odd alpha with no trace of the comparison
            # it belongs to. Both are tags, so they cost no run ids: a floor cell
            # shared by two FL cells hashes once and executes once.
            "arm", "reduced_from_k"}


def config_class(mode: str):
    if mode == "centralized":
        return Config
    if mode == "federated":
        return FedConfig
    raise ValueError(f"Unknown mode: {mode!r} (expected 'centralized' or 'federated')")


def _config_fields(mode: str):
    return {f.name for f in dataclasses.fields(config_class(mode))}


def build_config(spec: dict):
    """Instantiate the Config/FedConfig for `spec`, ignoring tag keys.

    Config fields present in the spec are passed through; unknown non-tag keys
    raise, so a typo in a manifest fails loudly instead of being silently
    dropped.
    """
    mode = spec.get("mode")
    cls = config_class(mode)
    fields = _config_fields(mode)

    unknown = set(spec) - fields - TAG_KEYS
    if unknown:
        raise ValueError(
            f"Spec has keys that are neither {mode} config fields nor tags: "
            f"{sorted(unknown)}"
        )

    # An AdamW spec must state its own learning rate. Config's lr=50.0 default is
    # GD's -- Gromov's -- and it is catastrophic under AdamW, but nothing
    # downstream catches it: check_decay_stability returns early whenever
    # weight_decay is 0, so the run proceeds, diverges, and banks NaN accuracies
    # as an ordinary censored result. Every one of the 1,290 banked AdamW specs
    # sets lr, so this closes a trap rather than fixing damage.
    if spec.get("optimizer") == "adamw" and "lr" not in spec:
        raise ValueError(
            "An optimizer='adamw' spec must set `lr` explicitly. The default "
            f"lr={cls.lr} is GD's and diverges under AdamW; published AdamW "
            "grokking configs sit at 1e-4 (Omnigrok) to 1e-3 (Nanda). Nothing "
            "downstream rejects the mismatch, so it has to be caught here."
        )

    kwargs = {k: v for k, v in spec.items() if k in fields}
    return cls(**kwargs)


def run_id(spec: dict) -> str:
    """Stable short id for a spec.

    Uses an explicit `id` if present; otherwise hashes the config-relevant part
    (everything except tags and any pre-existing id), so the same run always
    maps to the same id regardless of tag noise or key order. hashlib is
    deterministic — safe for resume across sessions.
    """
    if spec.get("id"):
        return spec["id"]
    core = {k: v for k, v in spec.items() if k not in TAG_KEYS}
    canonical = json.dumps(core, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(canonical.encode()).hexdigest()[:12]
    mode = spec.get("mode", "run")
    task = spec.get("task", "")
    return f"{mode[:4]}_{task}_{digest}" if task else f"{mode[:4]}_{digest}"


def expand_grid(base: dict, axes: dict, tags: dict = None) -> list:
    """Cartesian product of `axes` over a `base` spec.

    base:  fields common to every run (mode, p, hidden_width, ...).
    axes:  {field: [values]} — one run per combination.
    tags:  optional constant tags added to every run (tier, group, ...).

    Each run gets a stable `id`. Example:
        expand_grid({"mode": "federated", "task": "addition", "p": 97},
                    {"seed": [42, 123], "local_epochs": [1, 5]})
    yields 4 specs.
    """
    tags = tags or {}
    keys = list(axes)
    specs = []
    for combo in itertools.product(*(axes[k] for k in keys)):
        spec = dict(base)
        spec.update(tags)
        spec.update(dict(zip(keys, combo)))
        spec["id"] = run_id(spec)
        specs.append(spec)
    return specs


def load_manifest(path: str) -> list:
    """Read a JSONL manifest (one spec per line). Blank lines / # comments skipped."""
    specs = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            spec = json.loads(line)
            spec.setdefault("id", run_id(spec))
            specs.append(spec)
    return specs


def orphaned_ids(specs: list, path: str) -> list:
    """Ids present in the manifest at `path` that `specs` would no longer produce.

    A run id is a content hash of the spec *as written* (see run_id), so adding a
    field to a spec changes its id even at the field's default value. Any banked
    result JSON keyed by an old id is then orphaned: the launcher no longer
    recognises it as done and re-runs completed work.

    Returns the ids that would disappear. Empty list means the rewrite is safe.
    """
    if not os.path.exists(path):
        return []
    existing = {s["id"] for s in load_manifest(path)}
    new = {s.get("id") or run_id(s) for s in specs}
    return sorted(existing - new)


def write_manifest(specs: list, path: str, force: bool = False):
    """Write specs as JSONL, filling in ids.

    Refuses to drop ids an existing manifest already claims unless `force`, so a
    schema change cannot silently orphan banked runs. See `orphaned_ids`.
    """
    if not force:
        lost = orphaned_ids(specs, path)
        if lost:
            raise ValueError(
                f"Rewriting {path} would orphan {len(lost)} run id(s) that the "
                f"existing manifest claims, e.g. {lost[:3]}. Any banked result "
                f"JSON under those ids would stop counting as done and be re-run. "
                f"Add new cells in a NEW manifest, or pass force=True if the "
                f"orphaned runs are genuinely unwanted."
            )
    with open(path, "w") as handle:
        for spec in specs:
            spec = dict(spec)
            spec.setdefault("id", run_id(spec))
            handle.write(json.dumps(spec, sort_keys=True) + "\n")
