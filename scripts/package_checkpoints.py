"""Package the gitignored checkpoints into uploadable archives, by campaign group.

The ~34 GB of `results/runs/*/checkpoints/*.pt` is the one part of this project
that git cannot hold: a sample gzips to 92% of its size, so the tree would stop
being clonable and buy nothing. This packages it for a data host instead.

Three choices here are deliberate and cost real time if reversed:

  * **No compression.** `tar`, not `tar.gz`. Torch checkpoints are already packed
    float tensors; the measured ratio is 92%, so gzip spends hours to save ~8%.
  * **Archive per campaign group, not per run.** 33,088 loose files means 33,088
    LFS pointers and per-file round trips. Twelve archives upload far faster, and
    a group is the unit someone actually wants ("give me the boundary runs").
  * **A manifest that ties archives back to configs.** `results/data/runs_v2.csv`
    is committed and already maps run id -> every config field, so an archive
    plus that CSV is self-describing. The manifest written here is the join.

Usage:
    venv/bin/python scripts/package_checkpoints.py --out /path/with/40GB/free
    venv/bin/python scripts/package_checkpoints.py --out DIR --groups boundary
    venv/bin/python scripts/package_checkpoints.py --out DIR --dry-run

The archives live at https://huggingface.co/datasets/FedGrok/fedgrok-checkpoints
(gated, auto-approved), one per campaign group, filed under a folder named for
the paper axis it supports (num_clients/, local_epochs/, participation/,
heterogeneity/, mechanism/, legacy/). The release is ADDITIVE: an archive, once
published, is never renamed, moved or replaced, because its basename is the
`group` column of `runs_v2.csv` and that is the join key readers rely on. To add
a campaign, package only its groups here, run `scripts/merge_checkpoint_release.py`
to assign each new archive its folder and extend the root files (README.md,
MANIFEST.csv, SHA256SUMS, runs_v2.csv) as strict supersets of the published ones,
upload the archives to the paths the merged manifest names, then the root files.
The README this script writes is a starting point for a fresh dataset, not a
replacement for the published one.

`--groups boundary` is the cheapest useful subset: 4.12 GB, and it is the cell
RESULTS.md 16.2 reads for the mechanism result, so it makes the paper's strongest
mechanistic claim reproducible on its own.
"""

import argparse
import collections
import csv
import glob
import hashlib
import json
import os
import subprocess
import sys

RUNS_ROOT = "results/runs"
CSV_PATH = "results/data/runs_v2.csv"


def run_groups():
    """{run_id: campaign group} from the committed summary CSV."""
    with open(CSV_PATH) as fh:
        return {r["id"]: (r["group"] or "ungrouped") for r in csv.DictReader(fh)}


def collect():
    """{group: [(run_id, [checkpoint paths], bytes)]} for every run with checkpoints."""
    groups = run_groups()
    out = collections.defaultdict(list)
    for run_id in sorted(os.listdir(RUNS_ROOT)):
        paths = sorted(glob.glob(os.path.join(RUNS_ROOT, run_id, "checkpoints", "*.pt")))
        if not paths:
            continue
        size = sum(os.path.getsize(p) for p in paths)
        out[groups.get(run_id, "ungrouped")].append((run_id, paths, size))
    return out


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_readme(out_dir, rows, total_bytes, total_files):
    """A README the archive can be published with, so it is not a pile of tarballs."""
    lines = [
        "# Federated grokking — model checkpoints and per-client weights",
        "",
        f"{total_files:,} PyTorch checkpoint files, {total_bytes / 1e9:.1f} GB, from the",
        "v2 multi-setup campaign of a study on whether grokking survives federated",
        "averaging. One archive per campaign group; **no compression** (these are",
        "already-packed float tensors and gzip recovers ~8%).",
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
        "## How to map a run id to its configuration",
        "",
        "`run_id` is a content hash of the run's full config, so it is stable and",
        "unique. `results/data/runs_v2.csv` in the source repository maps every id to",
        "all 66 of its config and result fields (setup, alpha, K, local epochs,",
        "partition, weight decay, t_memo, t_first_cross, ...). An archive plus that",
        "CSV is therefore self-describing — nothing here needs a separate key.",
        "",
        "## Archives",
        "",
        "| archive | group | runs | files | size | sha256 (first 16) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['archive']}` | {r['group']} | {r['runs']} | {r['files']:,} | "
            f"{r['bytes'] / 1e9:.2f} GB | `{r['sha256'][:16]}` |"
        )
    lines += [
        "",
        "Verify with `sha256sum -c SHA256SUMS`.",
        "",
        "## Extract",
        "",
        "```bash",
        "tar -xf <archive>.tar          # restores results/runs/<id>/checkpoints/",
        "```",
        "",
    ]
    with open(os.path.join(out_dir, "README.md"), "w") as fh:
        fh.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory for the archives")
    ap.add_argument("--groups", nargs="*", help="only these campaign groups (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="report the plan, write nothing")
    ap.add_argument("--no-hash", action="store_true",
                    help="skip sha256 (hashing 34 GB takes a few minutes)")
    args = ap.parse_args()

    found = collect()
    if args.groups:
        missing = set(args.groups) - set(found)
        if missing:
            raise SystemExit(f"No checkpoints for group(s): {sorted(missing)}\n"
                             f"Available: {sorted(found)}")
        found = {g: v for g, v in found.items() if g in args.groups}

    plan = sorted(found.items(), key=lambda kv: -sum(s for _, _, s in kv[1]))
    total_bytes = sum(s for v in found.values() for _, _, s in v)
    total_files = sum(len(p) for v in found.values() for _, p, _ in v)

    print(f"{len(plan)} archive(s), {total_files:,} files, {total_bytes / 1e9:.1f} GB\n")
    for group, runs in plan:
        size = sum(s for _, _, s in runs)
        print(f"  {group:<22} {size / 1e9:>6.2f} GB  {len(runs):>4} runs  "
              f"{sum(len(p) for _, p, _ in runs):>6} files")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    free = os.statvfs(args.out if os.path.isdir(args.out) else ".")
    free_bytes = free.f_bavail * free.f_frsize
    if free_bytes < total_bytes * 1.05:
        raise SystemExit(f"\nNeed ~{total_bytes / 1e9:.0f} GB free at {args.out}, "
                         f"have {free_bytes / 1e9:.0f} GB. Archives are uncompressed "
                         f"by design; pass --groups to do a subset.")

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for group, runs in plan:
        archive = os.path.join(args.out, f"checkpoints_{group}.tar")
        listfile = archive + ".filelist"
        with open(listfile, "w") as fh:
            for _, paths, _ in runs:
                fh.write("\n".join(paths) + "\n")
        print(f"\n  -> {os.path.basename(archive)} "
              f"({sum(s for _, _, s in runs) / 1e9:.2f} GB) ...", end="", flush=True)
        # -T reads the path list, so the argv length is bounded regardless of file count
        subprocess.run(["tar", "-cf", archive, "-T", listfile], check=True)
        os.remove(listfile)
        digest = "" if args.no_hash else sha256(archive)
        print(" done" + ("" if args.no_hash else f"  sha256 {digest[:16]}"))
        rows.append(dict(archive=os.path.basename(archive), group=group,
                         runs=len(runs), files=sum(len(p) for _, p, _ in runs),
                         bytes=sum(s for _, _, s in runs), sha256=digest))

    with open(os.path.join(args.out, "MANIFEST.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["archive", "group", "runs", "files",
                                           "bytes", "sha256"])
        w.writeheader()
        w.writerows(rows)
    if not args.no_hash:
        with open(os.path.join(args.out, "SHA256SUMS"), "w") as fh:
            for r in rows:
                fh.write(f"{r['sha256']}  {r['archive']}\n")
    write_readme(args.out, rows, total_bytes, total_files)

    print(f"\nWrote {len(rows)} archive(s) + MANIFEST.csv + README.md to {args.out}")
    print("\nNext — publish to FedGrok/fedgrok-checkpoints (additive; see the module docstring):")
    print("  1. Merge onto the published root files; this also names each archive's folder:")
    print("       hf download FedGrok/fedgrok-checkpoints MANIFEST.csv SHA256SUMS \\")
    print("           --repo-type dataset --local-dir <published>")
    print(f"       python scripts/merge_checkpoint_release.py --published <published> \\")
    print(f"           --new {args.out} --out <root>")
    print("  2. Upload each new archive to the `path` in <root>/MANIFEST.csv, e.g.")
    print(f"       hf upload FedGrok/fedgrok-checkpoints {args.out}/checkpoints_<group>.tar \\")
    print("           <folder>/checkpoints_<group>.tar --repo-type dataset")
    print("  3. Then the four root files:")
    print("       hf upload FedGrok/fedgrok-checkpoints <root> . --repo-type dataset \\")
    print("           --include README.md --include MANIFEST.csv --include SHA256SUMS \\")
    print("           --include runs_v2.csv")
    print("  Zenodo, if a citable DOI is wanted: one record per campaign (50 GB cap),")
    print("  with runs_v2.csv alongside so the run ids resolve without the repo.")


if __name__ == "__main__":
    main()
