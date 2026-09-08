"""Training loop: full-batch gradient descent or AdamW, MSE or CE loss."""

import json
import os
import time
import torch
from fedgrok.core.config import Config
from fedgrok.data.registry import build_dataset, dataset_dims
from fedgrok.core.registry import build_model, build_loss
from fedgrok.metrics.fourier import (
    weight_norms, gradient_norms, compute_ipr, compute_accuracy,
    fourier_spectrum, dft_applicable, weight_norms_applicable, weight_norm_report,
)
from fedgrok.metrics.probes import mechanistic_probe
from fedgrok.core.utils import get_device, make_optimizer


def train(cfg: Config):
    """Run the full training loop. Returns the history dict."""
    torch.manual_seed(cfg.seed)
    device = get_device()
    print(f"Using device: {device}")

    # Data (via the registry, so cfg.dataset selects modular / MNIST / ...)
    x_train, y_train, x_test, y_test = build_dataset(cfg)
    # Number of output classes: p for modular, but 10 for MNIST — take it from
    # the dataset dims so the MSE one-hot width is correct for every dataset.
    n_classes = dataset_dims(cfg)[1]

    # Move data to device, then prepare loss targets there (one-hot for MSE,
    # class indices for CE — build_loss owns that choice).
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    loss = build_loss(cfg)
    loss_fn = loss.loss_fn
    y_train_target = loss.prepare_target(y_train, n_classes)
    y_test_target = loss.prepare_target(y_test, n_classes)

    # Model (via the registry, so cfg.model selects the architecture)
    model = build_model(cfg).to(device)

    optimizer = make_optimizer(model, cfg)

    # History
    history = {
        "epoch": [],
        "train_loss": [], "test_loss": [],
        "train_acc": [], "test_acc": [],
        "weight_norm_layer1": [], "weight_norm_layer2": [],
        "grad_norm_layer1": [], "grad_norm_layer2": [],
        "ipr": [],
    }

    start = time.time()

    minibatch = cfg.batch_size and cfg.batch_size > 0
    n_train = x_train.shape[0]

    for epoch in range(cfg.epochs + 1):
        # ── Forward + backward ──────────────────────────────────────────
        model.train()
        if minibatch:
            # One epoch = one shuffled pass in minibatches (Omnigrok/MNIST).
            # randperm is only ever called here, so the full-batch path below
            # leaves the RNG stream untouched and stays bit-identical.
            perm = torch.randperm(n_train, device=device)
            for i in range(0, n_train, cfg.batch_size):
                idx = perm[i:i + cfg.batch_size]
                optimizer.zero_grad()
                loss_i = loss_fn(model(x_train[idx]), y_train_target[idx])
                loss_i.backward()
                optimizer.step()
            # Recompute the full-batch train state for logging.
            with torch.no_grad():
                out_train = model(x_train)
                train_loss_value = loss_fn(out_train, y_train_target).item()
        else:
            # Full-batch GD: one epoch = one step. Log before stepping so the
            # gradient norms reflect this step's gradient (step is at the bottom).
            out_train = model(x_train)
            train_loss_t = loss_fn(out_train, y_train_target)
            optimizer.zero_grad()
            train_loss_t.backward()
            train_loss_value = train_loss_t.item()

        # Log before step (so gradient norms are available in the full-batch path).
        # The FINAL epoch is always logged, even when it is not a multiple of
        # log_every: run._completion treats the last logged epoch as the horizon
        # the run reached, and its docstring already promises this loop records
        # it. Without the `or`, any config where epochs % log_every != 0 ends one
        # sample short and is raised as an IncompleteRun -- a correct run thrown
        # away and re-queued. No banked run trips it (all 610 centralized rows
        # divide evenly), but reduced_arm derives the floor arm's budget from the
        # FL arm's step count, which is under no obligation to be round.
        if epoch % cfg.log_every == 0 or epoch == cfg.epochs:
            # eval() for the held-out pass. No model in this repo has dropout or
            # batch norm -- GrokFormer deliberately has no LayerNorm either -- so
            # this changes nothing today. It is here because the FEDERATED loop's
            # evaluate_fn does call eval(), and an undocumented asymmetry between
            # the two loops is the kind that stays invisible until a model gains
            # a stochastic layer and only the centralized numbers move.
            #
            # `out_train` is still the training forward pass, computed above in
            # train mode. That is the cheap and conventional choice, and it is
            # the one remaining difference from the federated loop, which
            # recomputes train accuracy under eval().
            model.eval()
            with torch.no_grad():
                out_test = model(x_test)
                test_loss = loss_fn(out_test, y_test_target).item()
                train_acc = compute_accuracy(out_train, y_train)
                test_acc = compute_accuracy(out_test, y_test)

            # Two INDEPENDENT capabilities, deliberately not one gate:
            #   weight_norms_applicable -- the model has named W1/W2. Frobenius
            #     norms are basis-free, so they are valid on S5 too.
            #   dft_applicable -- W1/P *and* a cyclic group. IPR only.
            # These used to share the `fourier_applicable` gate, which silently
            # discarded real weight-norm data on every non-modular run.
            nan = float("nan")
            if weight_norms_applicable(model):
                wn = weight_norms(model)
                gn = gradient_norms(model)
                wn1, wn2 = wn["weight_norm_layer1"], wn["weight_norm_layer2"]
                gn1 = gn.get("grad_norm_layer1", 0.0)
                gn2 = gn.get("grad_norm_layer2", 0.0)
            else:
                wn1 = wn2 = gn1 = gn2 = nan
            ipr_val = compute_ipr(model)["ipr"] if dft_applicable(model, cfg) else nan

            # Architecture-agnostic norms: the Omnigrok order parameter, and the
            # only weight signal available on the transformer / MNIST MLP.
            report = weight_norm_report(model)
            probe = mechanistic_probe(cfg)(model, x_test, y_test, cfg)
            model.train()

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss_value)
            history["test_loss"].append(test_loss)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)
            history["weight_norm_layer1"].append(wn1)
            history["weight_norm_layer2"].append(wn2)
            history["grad_norm_layer1"].append(gn1)
            history["grad_norm_layer2"].append(gn2)
            history["ipr"].append(ipr_val)
            # Additive keys. setdefault keeps them in lockstep with the fixed
            # series above: epoch 0 is always logged, so every key is created on
            # the first pass and appended to on every pass thereafter.
            for key, value in {**report, **probe}.items():
                history.setdefault(key, []).append(value)

            if epoch % (cfg.log_every * 10) == 0:
                elapsed = time.time() - start
                print(
                    f"[{epoch:>6d}/{cfg.epochs}]  "
                    f"train_loss={train_loss_value:.6f}  test_loss={test_loss:.6f}  "
                    f"train_acc={train_acc:.1f}%  test_acc={test_acc:.1f}%  "
                    f"ipr={ipr_val:.4f}  ({elapsed:.1f}s)"
                )

        # Save checkpoint if requested. The Fourier spectrum is GrokNet-specific;
        # other models still checkpoint their weights, just without it.
        if cfg.checkpoint_every > 0 and epoch % cfg.checkpoint_every == 0 and epoch > 0:
            ckpt_dir = os.path.join(cfg.output_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)
            if dft_applicable(model, cfg):
                spec = fourier_spectrum(model)
                spec_path = os.path.join(ckpt_dir, f"spectrum_epoch{epoch}.pt")
                torch.save(spec["spectrum"], spec_path)

        # Full-batch takes its single step here (after logging its gradient);
        # the minibatch path already stepped inside its pass.
        if not minibatch:
            optimizer.step()

    # Save results
    os.makedirs(cfg.output_dir, exist_ok=True)
    # Setup goes in the name — see the note on the federated tag. An MNIST or S5
    # run would otherwise inherit Config's task/p defaults and be written as
    # "addition_..._p97_...".
    task_part = f"{cfg.task}_p{cfg.p}" if cfg.dataset == "modular" else (
        f"n{cfg.group_n}" if cfg.dataset == "s5" else f"n{cfg.n_train}")
    tag = (f"{cfg.dataset}_{cfg.model}_{cfg.loss}_{task_part}_{cfg.optimizer}"
           f"_N{cfg.hidden_width}_a{cfg.alpha}_s{cfg.seed}")
    history_path = os.path.join(cfg.output_dir, f"history_{tag}.json")
    with open(history_path, "w") as f:
        json.dump(history, f)
    print(f"\nHistory saved to {history_path}")

    if cfg.save_weights:
        weights_path = os.path.join(cfg.output_dir, f"weights_{tag}.pt")
        torch.save(model.state_dict(), weights_path)
        print(f"Weights saved to {weights_path}")

    return history, model
