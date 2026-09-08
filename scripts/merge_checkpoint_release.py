"""Extend the published checkpoint release with a new campaign, without touching it.

The dataset (FedGrok/fedgrok-checkpoints on Hugging Face) files one archive per
campaign `group` under a folder named for the paper axis it supports:

    num_clients/      local_epochs/     participation/
    heterogeneity/    mechanism/        legacy/

An archive, once published, is never renamed, moved or replaced: its basename is
the `group` column of `results/data/runs_v2.csv`, and that is the join key
readers rely on. What DOES change when a campaign lands is the four root files,
and each must be a strict superset of what is published:

    MANIFEST.csv   every published row kept, new rows appended
    SHA256SUMS     every published line kept, new lines appended
    runs_v2.csv    the current table (a superset by construction: ids are never dropped)
    README.md      regenerated from the merged manifest

`package_checkpoints.py` writes a MANIFEST.csv + SHA256SUMS for the groups it
packaged, at bare names; this joins them onto the published ones, assigns each
new archive its folder, and adds the columns a reader needs to go from an
archive to the paper (axis, figure, campaign, setups, what varies). Every group
must have an entry in AXIS below, so a new campaign fails here, loudly, until
someone says which axis and figure it belongs to.

    hf download FedGrok/fedgrok-checkpoints MANIFEST.csv SHA256SUMS \\
        --repo-type dataset --local-dir published/
    python scripts/merge_checkpoint_release.py --published published/ \\
        --new /tmp/fedgrok_sept --out root/
    # upload each new archive to the folder the merged MANIFEST's `path` names,
    # then the four root files:
    hf upload FedGrok/fedgrok-checkpoints root/ . --repo-type dataset \\
        --include README.md --include MANIFEST.csv --include SHA256SUMS --include runs_v2.csv
"""

import argparse
import collections
import csv
import os
import shutil
import sys

# group -> (folder, axis label, figure in the Sept 2026 draft, campaign key)
AXIS = {
    "aggregation": ("num_clients", "number of clients K", "Fig 1, Fig A1", "aug"),
    "aggregation_alpha2": ("num_clients", "number of clients K", "Fig 1", "aug"),
    "setup_k_ladder": ("num_clients", "number of clients K", "Fig 1, Fig A1", "aug"),
    "b_k20_wd_control": ("num_clients", "number of clients K", "text (B weight-decay control)", "aug"),
    "k_collapse_budget": ("num_clients", "number of clients K", "text (B K-collapse diagnosis)", "aug"),
    "local_epochs": ("local_epochs", "local epochs E", "Fig 2", "sep"),
    "e50_long": ("local_epochs", "local epochs E", "Fig 2, Fig 6", "sep"),
    "participation": ("participation", "participation f", "Fig 3", "aug"),
    "participation_setups": ("participation", "participation f", "Fig 3", "sep"),
    "dirichlet_band": ("heterogeneity", "heterogeneity (Dirichlet)", "Fig 4a-b", "aug"),
    "dirichlet_setups": ("heterogeneity", "heterogeneity (Dirichlet)", "Fig 4a-b", "sep"),
    "size_control": ("heterogeneity", "heterogeneity (Dirichlet)", "Fig 4c", "aug"),
    "partitions": ("heterogeneity", "heterogeneity (structured partitions)", "Fig 4d", "aug"),
    "boundary": ("mechanism", "mechanism", "Fig 7", "aug"),
    "d_internals": ("mechanism", "mechanism", "Fig 7, Fig A1", "aug"),
    "ungrouped": ("legacy", "legacy", "none", "aug"),
}
FOLDER = {  # folder -> (axis label, one-line description)
    "num_clients": (
        "number of clients K",
        "The K ladder for every setup, with the 87 centralised anchors (the K=1 end) "
        "inside `checkpoints_aggregation.tar`.",
    ),
    "local_epochs": (
        "local epochs E",
        "Local work per round, E in {10, 25, 50}, and the long E=50 runs behind the "
        "fixed point.",
    ),
    "participation": ("participation f", "Fraction of clients per round."),
    "heterogeneity": (
        "heterogeneity",
        "Dirichlet label skew, the client-size control, and the structured partitions.",
    ),
    "mechanism": (
        "mechanism",
        "The boundary cells and the setup-D internals the mechanism analysis reads.",
    ),
    "legacy": (
        "legacy",
        "11 partial runs from a superseded batch; not in the run table, keep out of "
        "analyses.",
    ),
}
CAMPAIGN = {"aug": "v2 campaigns to 26 Aug 2026", "sep": "2-4 Sep 2026 sweep"}
KNOBS = [
    "num_clients",
    "local_epochs",
    "fraction_train",
    "dirichlet_alpha",
    "partition",
    "strategy",
    "alpha",
    "weight_decay",
]
SHORT = {
    "num_clients": "K",
    "local_epochs": "E",
    "fraction_train": "f",
    "dirichlet_alpha": "alpha_dir",
    "partition": "partition",
    "strategy": "strategy",
    "alpha": "alpha",
    "weight_decay": "wd",
}
FIELDS = [
    "archive",
    "path",
    "group",
    "runs",
    "files",
    "bytes",
    "sha256",
    "axis",
    "figure",
    "campaign",
    "setups",
    "modes",
    "varies",
]


def read_manifest(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def read_sums(path):
    """{archive path: sha256} from a SHA256SUMS file."""
    out = {}
    with open(path) as fh:
        for line in fh:
            if line.strip():
                digest, name = line.split(maxsplit=1)
                out[name.strip()] = digest
    return out


def group_stats(rows_by_group, group):
    """(setups, modes, what varies) over the group's checkpointed runs."""
    rs = [
        r
        for r in rows_by_group.get(group, [])
        if r["checkpoint_every"] not in ("", "0", "0.0")
    ]
    if not rs:
        return "", "", ""
    setups = sorted({r["setup"] or "A" for r in rs})
    modes = sorted({r["mode"] for r in rs})
    varying = []
    for k in KNOBS:
        vals = sorted({r[k] for r in rs if r[k] != ""}, key=lambda v: (len(v), v))
        if len(vals) > 1:
            varying.append(f"{SHORT[k]} in {{{', '.join(vals)}}}")
    return "/".join(setups), "+".join(modes), "; ".join(varying)


def write_readme(out_dir, rows, n_table):
    total_files = sum(int(r["files"]) for r in rows)
    total_bytes = sum(int(r["bytes"]) for r in rows)
    total_runs = sum(int(r["runs"]) for r in rows)
    folders = [f for f in FOLDER if any(r["path"].startswith(f + "/") for r in rows)]

    def table(sel):
        lines = [
            "| archive | group | setups | what varies | runs | files | size | sha256 (first 16) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in sel:
            lines.append(
                f"| `{r['path']}` | {r['group']} | {r['setups']} | {r['varies'] or '-'} | "
                f"{r['runs']} | {int(r['files']):,} | {int(r['bytes']) / 1e9:.2f} GB | "
                f"`{r['sha256'][:16]}` |"
            )
        return lines

    L = [
        "---",
        "pretty_name: Federated grokking checkpoints",
        "size_categories:",
        "- n<1K",
        "tags:",
        "- grokking",
        "- federated-learning",
        "- modular-arithmetic",
        "- checkpoints",
        "- pytorch",
        "configs:",
        "- config_name: runs",
        "  data_files: runs_v2.csv",
        "- config_name: archives",
        "  data_files: MANIFEST.csv",
        "---",
        "",
        "# Federated grokking — model checkpoints and per-client weights",
        "",
        f"{total_files:,} PyTorch checkpoint files, {total_bytes / 1e9:.1f} GB, from {total_runs} runs of",
        "the v2 multi-setup campaign of a study on whether grokking survives federated",
        "averaging (FedAvg). One archive per campaign group, filed by the axis of the paper it",
        "supports; **no compression** (these are already-packed float tensors and gzip recovers ~8%).",
        "Access is gated with automatic approval: click *request access* once, then any",
        "authenticated download works.",
        "",
        "The study trains five setups — A: quadratic MLP on mod-97 addition (GD, wd 0);",
        "A': the same with AdamW; B: one-layer transformer on mod-113 addition (AdamW);",
        "C: transformer on S5 composition; D: quadratic MLP on S5; E: MLP on MNIST —",
        "centralised and under FedAvg, and measures when each memorises and when it grokks",
        "as the number of clients K, local epochs E, participation f and data heterogeneity vary.",
        "",
        "## Layout",
        "",
        "| path | what it is |",
        "|---|---|",
    ]
    for f in folders:
        sel = [r for r in rows if r["path"].startswith(f + "/")]
        L.append(
            f"| `{f}/` | {FOLDER[f][1].rstrip('.')} — "
            f"{len(sel)} archive{'s' if len(sel) != 1 else ''}, "
            f"{sum(int(r['runs']) for r in sel)} runs, "
            f"{sum(int(r['bytes']) for r in sel) / 1e9:.1f} GB |"
        )
    L += [
        "| `MANIFEST.csv` | one row per archive: path, group, run/file counts, bytes, sha256, axis, figure, campaign, setups, what varies |",
        "| `SHA256SUMS` | `sha256sum -c SHA256SUMS` from this directory verifies every archive |",
        f"| `runs_v2.csv` | the run table: {n_table:,} rows, one per run, every config and result field |",
        "",
        "Folder names are the run table's column names, not figure numbers, so they stay put",
        "when the paper is renumbered; the `figure` column of `MANIFEST.csv` carries that map.",
        "Before 8 Sep 2026 every archive sat at the repository root; those paths no longer resolve.",
        "",
        "## Which archive holds which runs",
        "",
        "A run is identified by `run_id`, a content hash of its full config. The join is:",
        "`runs_v2.csv` maps run id -> `group` (plus setup, K, E, f, alpha_dir, partition,",
        "weight decay, seed, t_memo, t_first_cross, ...); `MANIFEST.csv` maps group -> `path`.",
        "Runs with `checkpoint_every = 0` in the table have no checkpoints and are in no archive.",
        "Every run with `checkpoint_every > 0` is in exactly one archive, complete for its",
        "save schedule (verified file by file on 8 Sep 2026).",
        "",
    ]
    for f in folders:
        sel = [r for r in rows if r["path"].startswith(f + "/")]
        figs = []
        for r in sel:
            for fig in r["figure"].split(", "):
                if fig not in figs:
                    figs.append(fig)
        L += [f"### `{f}/` — {FOLDER[f][0]}", ""]
        if f == "legacy":
            L += [
                "`checkpoints_ungrouped.tar` holds 11 partial `aggregation_alpha2` runs (old",
                "`fede_addition_*` ids, 2-15 checkpoints each) that are not in the run table.",
                "They were superseded by the runs in `num_clients/checkpoints_aggregation_alpha2.tar`",
                "and are kept only so the published record is unchanged. Do not use them for analysis.",
                "",
            ]
        else:
            L += [f"{FOLDER[f][1]} Sept 2026 draft: {', '.join(figs)}.", ""]
        L += table(sel) + [""]
    L += [
        "## What is not here",
        "",
        "- **Strategy comparison** (FedAvg, FedAdam, FedYogi, SCAFFOLD, FedAvgM, FedProx; Fig 5 and",
        "  Fig A2 of the draft): those 90 runs were never checkpointed. Their histories are in the",
        "  source repository.",
        "- **Setup A on the local-epochs axis** (Fig 2, the anchor panel): the 12 runs at E in",
        "  {1, 10, 25, 50} were launched with checkpointing off. Histories only.",
        "- **Centralised anchors** outside the `aggregation` and `d_internals` groups, and every",
        "  hyper-parameter calibration group: histories only.",
        "- **Per-round training histories and run specs** for every run: committed in the source",
        "  repository under `results/runs/<run_id>/`, not duplicated here.",
        "",
        "## What is in an archive",
        "",
        "Paths inside each tar are `results/runs/<run_id>/checkpoints/*.pt`, holding:",
        "",
        "| prefix | what it is |",
        "|---|---|",
        "| `ckpt_*.pt` | global model `state_dict` at that round or epoch |",
        "| `client_*.pt` | per-client weight signature — the channel the mechanism analysis reads |",
        "| `spectrum_*.pt` | saved Fourier spectrum at that step |",
        "",
        "## Verify and extract",
        "",
        "```bash",
        "sha256sum -c SHA256SUMS                          # from this directory",
        "tar -xf heterogeneity/checkpoints_partitions.tar  # restores results/runs/<id>/checkpoints/",
        "```",
        "",
        "With `huggingface_hub`, one axis at a time:",
        "",
        "```python",
        "from huggingface_hub import snapshot_download",
        'snapshot_download("FedGrok/fedgrok-checkpoints", repo_type="dataset",',
        '                  allow_patterns=["heterogeneity/*", "*.csv", "SHA256SUMS"])',
        "```",
        "",
        "## Campaigns",
        "",
        "| campaign | archives | runs | size |",
        "|---|---|---|---|",
    ]
    for key in CAMPAIGN:
        sel = [r for r in rows if r["campaign"] == CAMPAIGN[key]]
        if sel:
            L.append(
                f"| {CAMPAIGN[key]} | {len(sel)} | {sum(int(r['runs']) for r in sel)} | "
                f"{sum(int(r['bytes']) for r in sel) / 1e9:.1f} GB |"
            )
    L.append("")
    with open(os.path.join(out_dir, "README.md"), "w") as fh:
        fh.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--published",
        required=True,
        help="directory holding the dataset's current MANIFEST.csv and SHA256SUMS",
    )
    ap.add_argument(
        "--new",
        required=True,
        help="package_checkpoints.py output for the new campaign",
    )
    ap.add_argument("--out", required=True, help="where to write the four root files")
    ap.add_argument("--table", default="results/data/runs_v2.csv")
    args = ap.parse_args()

    old = read_manifest(os.path.join(args.published, "MANIFEST.csv"))
    new = read_manifest(os.path.join(args.new, "MANIFEST.csv"))
    clash = {r["archive"] for r in old} & {r["archive"] for r in new}
    if clash:
        sys.exit(
            f"Refusing: {sorted(clash)} are already published. Archives are never "
            f"replaced; package under a new group or leave the published one."
        )
    unknown = {r["group"] for r in old + new} - set(AXIS)
    if unknown:
        sys.exit(
            f"No axis/figure entry for group(s) {sorted(unknown)}: add them to AXIS "
            f"in {os.path.basename(__file__)} first."
        )

    with open(args.table) as fh:
        table = list(csv.DictReader(fh))
    by_group = collections.defaultdict(list)
    for r in table:
        by_group[r["group"] or "ungrouped"].append(r)

    rows = []
    for r in old + new:
        folder, axis, fig, camp = AXIS[r["group"]]
        setups, modes, varies = group_stats(by_group, r["group"])
        path = r.get("path") or f"{folder}/{r['archive']}"
        rows.append(
            {
                k: r.get(k, "")
                for k in FIELDS
                if k
                not in (
                    "path",
                    "axis",
                    "figure",
                    "campaign",
                    "setups",
                    "modes",
                    "varies",
                )
            }
            | dict(
                path=path,
                axis=axis,
                figure=fig,
                campaign=CAMPAIGN[camp],
                setups=setups,
                modes=modes,
                varies=varies,
            )
        )
    rows.sort(key=lambda r: list(AXIS).index(r["group"]))

    # SHA256SUMS: published lines verbatim, new archives keyed by their folder path.
    sums = read_sums(os.path.join(args.published, "SHA256SUMS"))
    new_sums = read_sums(os.path.join(args.new, "SHA256SUMS"))
    for r in new:
        sums[f"{AXIS[r['group']][0]}/{r['archive']}"] = new_sums[r["archive"]]
    for r in rows:
        if sums.get(r["path"]) != r["sha256"]:
            sys.exit(f"sha256 for {r['path']} differs between MANIFEST and SHA256SUMS.")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "MANIFEST.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, "SHA256SUMS"), "w") as fh:
        for r in rows:
            fh.write(f"{r['sha256']}  {r['path']}\n")
    shutil.copy(args.table, os.path.join(args.out, "runs_v2.csv"))
    write_readme(args.out, rows, len(table))

    print(
        f"{args.out}: {len(rows)} archives ({len(old)} published + {len(new)} new), "
        f"table {len(table):,} rows"
    )


if __name__ == "__main__":
    main()
