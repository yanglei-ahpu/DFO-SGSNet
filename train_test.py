import argparse
import csv
import math
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from dataset import WidebandC4DataGenerator, WidebandC4Dataset
from loss import calibration_loss, count_loss, joint_loss, spectrum_loss
from model import PhysicsSpectrumNet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def profile_defaults(profile):
    if profile == "quick":
        return {
            "num_sensors": 5,
            "min_sources": 2,
            "max_sources": 6,
            "snapshots": 384,
            "num_subbands": 4,
            "train_samples": 800,
            "val_samples": 200,
            "test_samples": 200,
            "calibration_epochs": 5,
            "count_epochs": 18,
            "spectrum_epochs": 20,
            "joint_epochs": 8,
            "hidden_dim": 64,
            "batch_size": 16,
            "angle_min": -30.0,
            "angle_max": 30.0,
            "min_separation": 4.0,
            "snr_min": 24.0,
            "snr_max": 34.0,
            "coherence_max": 0.30,
            "grid_size": 1201,
        }

    if profile == "precision":
        return {
            "num_sensors": 8,
            "min_sources": 2,
            "max_sources": 12,
            "snapshots": 2048,
            "num_subbands": 10,
            "train_samples": 30000,
            "val_samples": 3000,
            "test_samples": 3000,
            "calibration_epochs": 6,
            "count_epochs": 60,
            "spectrum_epochs": 80,
            "joint_epochs": 25,
            "hidden_dim": 128,
            "batch_size": 16,
            "angle_min": -60.0,
            "angle_max": 60.0,
            "min_separation": 5.0,
            "snr_min": 28.0,
            "snr_max": 38.0,
            "coherence_max": 0.30,
            "grid_size": 2401,
        }

    return {
        "num_sensors": 8,
        "min_sources": 2,
        "max_sources": 10,
        "snapshots": 512,#512
        "num_subbands": 8,
        "train_samples": 16000,
        "val_samples": 2000,
        "test_samples": 2000,
        "calibration_epochs": 6,
        "count_epochs": 45,
        "spectrum_epochs": 60,
        "joint_epochs": 20,
        "hidden_dim": 96,
        "batch_size": 18,
        "angle_min": -60.0,
        "angle_max": 60.0,
        "min_separation": 5.0,
        "snr_min": 10.0,
        "snr_max": 15.0,
        "coherence_max": 0.35,
        "grid_size": 2401,
    }


def apply_profile(args):
    defaults = profile_defaults(args.profile)
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    if args.data_dir is None:
        args.data_dir = "./data_physics_spectrum_v6_{}".format(args.profile)
    if args.checkpoint_dir is None:
        args.checkpoint_dir = "./checkpoints_physics_spectrum_v6_{}".format(args.profile)
    if args.result_dir is None:
        args.result_dir = "./results_physics_spectrum_v6_{}".format(args.profile)
    if args.output_min_separation is None:
        args.output_min_separation = 0.60 * float(args.min_separation)
    return args


def make_generator(args, number, seed):
    return WidebandC4DataGenerator(
        num_samples=number,
        num_sensors=args.num_sensors,
        snapshots=args.snapshots,
        min_sources=args.min_sources,
        max_sources=args.max_sources,
        num_subbands=args.num_subbands,
        frequency_min=args.frequency_min,
        frequency_max=args.frequency_max,
        angle_min=args.angle_min,
        angle_max=args.angle_max,
        min_separation=args.min_separation,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        coherence_min=args.coherence_min,
        coherence_max=args.coherence_max,
        true_spacing_error=args.true_spacing_error,
        true_phase_slope_deg=args.true_phase_slope_deg,
        num_blocks=args.num_blocks,
        seed=seed,
    )


def generate_data(args):
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    configurations = [
        ("train.pt", args.train_samples, args.seed),
        ("val.pt", args.val_samples, args.seed + 1000),
        ("test.pt", args.test_samples, args.seed + 2000),
    ]

    for filename, number, seed in configurations:
        path = data_dir / filename
        if path.exists() and not args.regenerate:
            print("Using existing:", path)
            continue
        generator = make_generator(args, number, seed)
        generator.generate(
            str(path),
            batch_size=args.generation_batch_size,
            device=args.generation_device,
        )


def make_loaders(args):
    train_set = WidebandC4Dataset(str(Path(args.data_dir) / "train.pt"))
    val_set = WidebandC4Dataset(str(Path(args.data_dir) / "val.pt"))
    test_set = WidebandC4Dataset(str(Path(args.data_dir) / "test.pt"))

    common = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        **common
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        **common
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        **common
    )
    return train_set, train_loader, val_loader, test_loader


def move_batch(batch, device):
    return tuple(item.to(device, non_blocking=True) for item in batch)


def check_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError("Non-finite value detected in {}.".format(name))


def accumulate(total, losses):
    for key, value in losses.items():
        total[key] = total.get(key, 0.0) + float(value.detach().item())


def average(total, number):
    return {key: value / max(number, 1) for key, value in total.items()}


def run_epoch(model, loader, device, stage, args, optimizer=None, training=True):
    model.train(training)
    total = {}
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for batch in loader:
            (
                lags,
                log_variance,
                doa,
                source_number,
                true_spacing,
                true_phase,
            ) = move_batch(batch, device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            if stage == "calibration":
                output = model(lags, log_variance, mode="calibration")
                losses = calibration_loss(
                    output,
                    true_spacing,
                    true_phase,
                    model.max_spacing_error,
                    model.max_phase_slope,
                )
            elif stage == "count":
                output = model(lags, log_variance, mode="count")
                losses = count_loss(output, source_number, model.Kmin)
            elif stage == "spectrum":
                output = model(lags, log_variance, mode="full")
                losses = spectrum_loss(
                    output,
                    doa,
                    source_number,
                    fine_sigma=args.fine_sigma,
                    medium_sigma=args.medium_sigma,
                    coarse_sigma=args.coarse_sigma,
                )
            elif stage == "joint":
                output = model(lags, log_variance, mode="full")
                losses = joint_loss(
                    output,
                    doa,
                    source_number,
                    model.Kmin,
                    fine_sigma=args.fine_sigma,
                    medium_sigma=args.medium_sigma,
                    coarse_sigma=args.coarse_sigma,
                )
            else:
                raise ValueError("Unknown stage: {}".format(stage))

            check_finite("loss", losses["total"])

            if training:
                losses["total"].backward()
                parameters = [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ]
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad and parameter.grad is not None:
                        check_finite("gradient:" + name, parameter.grad)
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
                optimizer.step()
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        check_finite("parameter:" + name, parameter)

            accumulate(total, losses)

    return average(total, len(loader))


def peak_indices(spectrum, minimum_distance_bins, number_peaks):
    spectrum = np.asarray(spectrum, dtype=np.float64)
    local = []
    for index in range(1, spectrum.size - 1):
        if spectrum[index] >= spectrum[index - 1] and spectrum[index] > spectrum[index + 1]:
            local.append(index)

    order = sorted(local, key=lambda index: spectrum[index], reverse=True)
    if len(order) < number_peaks:
        order.extend(
            index
            for index in np.argsort(spectrum)[::-1]
            if index not in order
        )

    selected = []
    for index in order:
        if all(abs(index - previous) >= minimum_distance_bins for previous in selected):
            selected.append(index)
        if len(selected) >= number_peaks:
            break
    return selected


def extract_doa(spectrum, angle_grid, source_number, minimum_distance_deg):
    step = float(np.mean(np.diff(angle_grid)))
    distance_bins = max(1, int(round(minimum_distance_deg / step)))
    indices = peak_indices(spectrum, distance_bins, source_number)
    estimates = []

    for index in indices:
        if 0 < index < len(spectrum) - 1:
            left = math.log(max(float(spectrum[index - 1]), 1e-12))
            center = math.log(max(float(spectrum[index]), 1e-12))
            right = math.log(max(float(spectrum[index + 1]), 1e-12))
            denominator = left - 2.0 * center + right
            if abs(denominator) > 1e-12:
                offset = 0.5 * (left - right) / denominator
                offset = float(np.clip(offset, -0.5, 0.5))
            else:
                offset = 0.0
            estimates.append(float(angle_grid[index] + offset * step))
        else:
            estimates.append(float(angle_grid[index]))

    return np.sort(np.asarray(estimates, dtype=np.float64))


def match_errors(true_angles, estimated_angles):
    if len(true_angles) == 0 or len(estimated_angles) == 0:
        return np.empty(0, dtype=np.float64)
    cost = np.abs(true_angles[:, None] - estimated_angles[None, :])
    row, column = linear_sum_assignment(cost)
    return cost[row, column]


def empty_accumulator():
    return {
        "errors": [],
        "penalized_errors": [],
        "correct_k_errors": [],
        "under_errors": [],
        "samples": 0,
        "count_correct": 0,
        "all_within_1": 0,
        "under_samples": 0,
        "under_count_correct": 0,
        "by_k": {},
    }


def summarize_values(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    return (
        float(np.mean(np.abs(values))),
        float(np.sqrt(np.mean(np.square(values)))),
    )


def finalize_metrics(accumulator):
    overall_mae, overall_rmse = summarize_values(accumulator["errors"])
    penalized_mae, penalized_rmse = summarize_values(accumulator["penalized_errors"])
    correct_mae, correct_rmse = summarize_values(accumulator["correct_k_errors"])
    under_mae, under_rmse = summarize_values(accumulator["under_errors"])

    samples = max(accumulator["samples"], 1)
    correct_count = max(accumulator["count_correct"], 1)
    under_samples = max(accumulator["under_samples"], 1)

    metrics = {
        "samples": accumulator["samples"],
        "source_count_accuracy": accumulator["count_correct"] / samples,
        "matched_mae_deg": overall_mae,
        "matched_rmse_deg": overall_rmse,
        "penalized_mae_deg": penalized_mae,
        "penalized_rmse_deg": penalized_rmse,
        "correct_k_mae_deg": correct_mae,
        "correct_k_rmse_deg": correct_rmse,
        "all_sources_within_1deg_rate": accumulator["all_within_1"] / correct_count,
        "underdetermined_samples": accumulator["under_samples"],
        "underdetermined_count_accuracy": accumulator["under_count_correct"] / under_samples,
        "underdetermined_correct_k_mae_deg": under_mae,
        "underdetermined_correct_k_rmse_deg": under_rmse,
    }
    for k, values in sorted(accumulator["by_k"].items()):
        mae, rmse = summarize_values(values)
        metrics["K{}_mae_deg".format(k)] = mae
        metrics["K{}_rmse_deg".format(k)] = rmse
    return metrics


@torch.no_grad()
def evaluate(model, loader, device, args, spectrum_mode="refined", oracle_k=False, print_samples=False):
    model.eval()
    accumulator = empty_accumulator()
    last = None

    for batch in loader:
        (
            lags,
            log_variance,
            doa,
            source_number,
            true_spacing,
            true_phase,
        ) = move_batch(batch, device)
        output = model(lags, log_variance, mode="full")

        bank = output["music_bank"].cpu().numpy()
        physical = output["physical_spectrum"].cpu().numpy()
        refined = output["refined_spectrum"].cpu().numpy()
        predicted_k = output["predicted_k"].cpu().numpy()
        angle_grid = output["angle_grid"].cpu().numpy()

        for batch_index in range(lags.shape[0]):
            true_k = int(source_number[batch_index].item())
            estimate_k = true_k if oracle_k else int(predicted_k[batch_index])
            true_angles = doa[batch_index, :true_k].cpu().numpy().astype(np.float64)

            if oracle_k:
                spectrum = bank[batch_index, true_k - model.Kmin]
            elif spectrum_mode == "physical":
                spectrum = physical[batch_index]
            elif spectrum_mode == "refined":
                spectrum = refined[batch_index]
            else:
                raise ValueError("Unknown spectrum mode: {}".format(spectrum_mode))

            estimated_angles = extract_doa(
                spectrum,
                angle_grid,
                estimate_k,
                args.output_min_separation,
            )
            errors = match_errors(true_angles, estimated_angles)

            accumulator["errors"].extend(errors.tolist())
            accumulator["samples"] += 1
            accumulator["count_correct"] += int(true_k == estimate_k)

            penalty = 0.5 * (args.angle_max - args.angle_min)
            penalized = errors.tolist()
            penalized.extend([penalty] * abs(true_k - estimate_k))
            accumulator["penalized_errors"].extend(penalized)

            if true_k == estimate_k:
                accumulator["correct_k_errors"].extend(errors.tolist())
                accumulator["all_within_1"] += int(
                    len(errors) == true_k and np.all(errors <= 1.0)
                )

            if true_k > model.M:
                accumulator["under_samples"] += 1
                accumulator["under_count_correct"] += int(true_k == estimate_k)
                if true_k == estimate_k:
                    accumulator["under_errors"].extend(errors.tolist())

            accumulator["by_k"].setdefault(true_k, []).extend(errors.tolist())

            if print_samples:
                print(
                    "Sample {:05d} | True K={} | Pred K={} | True DOA={} | "
                    "Estimated DOA={} | Matched error={}".format(
                        accumulator["samples"],
                        true_k,
                        estimate_k,
                        np.round(true_angles, 4).tolist(),
                        np.round(estimated_angles, 4).tolist(),
                        np.round(errors, 4).tolist(),
                    )
                )

            last = (spectrum, angle_grid, true_angles, estimated_angles)

    return finalize_metrics(accumulator), last


def metric_score(metrics):
    penalized = metrics["penalized_mae_deg"]
    if not np.isfinite(penalized):
        return float("inf")
    return float(penalized)


def save_checkpoint(path, model, optimizer, epoch, score, inference_mode):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "score": score,
            "inference_mode": inference_mode,
        },
        path,
    )


def load_checkpoint(path, model, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def train_stage(
    model,
    train_loader,
    val_loader,
    device,
    stage,
    epochs,
    optimizer,
    checkpoint_path,
    args,
):
    best_score = float("inf")
    best_loss = float("inf")
    patience = 0
    history = []

    for epoch in range(epochs):
        train_losses = run_epoch(
            model,
            train_loader,
            device,
            stage,
            args,
            optimizer=optimizer,
            training=True,
        )
        val_losses = run_epoch(
            model,
            val_loader,
            device,
            stage,
            args,
            optimizer=None,
            training=False,
        )

        spacing, phase = model.calibration_values()
        row = {
            "stage": stage,
            "epoch": epoch + 1,
            "train_total": train_losses["total"],
            "val_total": val_losses["total"],
            "spacing_error": float(spacing.detach().cpu().item()),
            "phase_slope_deg": math.degrees(float(phase.detach().cpu().item())),
        }

        if stage == "calibration":
            score = val_losses["total"]
            inference_mode = "physical"
            message = (
                "CALIBRATION [{:03d}/{:03d}] Train={:.6f} Val={:.6f} "
                "Spacing={:.6f} Phase={:.4f}deg"
            ).format(
                epoch + 1,
                epochs,
                train_losses["total"],
                val_losses["total"],
                row["spacing_error"],
                row["phase_slope_deg"],
            )
        elif stage == "count":
            score = val_losses["total"]
            inference_mode = "physical"
            message = (
                "COUNT [{:03d}/{:03d}] Train={:.6f} Val={:.6f} "
                "CountAcc={:.3f}"
            ).format(
                epoch + 1,
                epochs,
                train_losses["total"],
                val_losses["total"],
                val_losses["count_accuracy"],
            )
        else:
            physical_metrics, _ = evaluate(
                model,
                val_loader,
                device,
                args,
                spectrum_mode="physical",
            )
            refined_metrics, _ = evaluate(
                model,
                val_loader,
                device,
                args,
                spectrum_mode="refined",
            )

            if metric_score(refined_metrics) <= metric_score(physical_metrics):
                selected_metrics = refined_metrics
                inference_mode = "refined"
            else:
                selected_metrics = physical_metrics
                inference_mode = "physical"

            score = metric_score(selected_metrics)
            row.update({
                "val_count_accuracy": selected_metrics["source_count_accuracy"],
                "val_matched_mae": selected_metrics["matched_mae_deg"],
                "val_penalized_mae": selected_metrics["penalized_mae_deg"],
                "val_correct_k_mae": selected_metrics["correct_k_mae_deg"],
                "val_physical_penalized_mae": physical_metrics["penalized_mae_deg"],
                "val_refined_penalized_mae": refined_metrics["penalized_mae_deg"],
                "inference_mode": inference_mode,
            })
            message = (
                "{} [{:03d}/{:03d}] Train={:.5f} Val={:.5f} "
                "CountAcc={:.3f} PhysicalPMAE={:.3f} RefinedPMAE={:.3f} "
                "Selected={} CorrectKMAE={:.3f}"
            ).format(
                stage.upper(),
                epoch + 1,
                epochs,
                train_losses["total"],
                val_losses["total"],
                selected_metrics["source_count_accuracy"],
                physical_metrics["penalized_mae_deg"],
                refined_metrics["penalized_mae_deg"],
                inference_mode,
                selected_metrics["correct_k_mae_deg"],
            )

        print(message)
        for key, value in val_losses.items():
            row["val_" + key] = value
        history.append(row)

        improved = score < best_score - 1e-8
        if improved:
            best_score = score
            best_loss = val_losses["total"]
            patience = 0
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch + 1,
                score,
                inference_mode,
            )
            print("Saved:", checkpoint_path)
        else:
            patience += 1
            if args.early_stopping > 0 and patience >= args.early_stopping:
                print("{} early stopping.".format(stage.upper()))
                break

    checkpoint = load_checkpoint(checkpoint_path, model, device)
    return history, checkpoint.get("inference_mode", "physical")


def save_history(path, history):
    if not history:
        return
    keys = sorted(set().union(*(row.keys() for row in history)))
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def write_metrics(path, metrics):
    lines = []
    for key, value in metrics.items():
        lines.append("{}: {}".format(key, value))
    text = "\n".join(lines)
    print("\n" + text)
    Path(path).write_text(text + "\n", encoding="utf-8")


def plot_last(path, last):
    if last is None:
        return
    spectrum, angle_grid, true_angles, estimated_angles = last
    spectrum_db = 10.0 * np.log10(
        np.maximum(spectrum / (np.max(spectrum) + 1e-12), 1e-12)
    )
    plt.figure(figsize=(11, 5))
    plt.plot(angle_grid, spectrum_db, linewidth=1.4)
    for angle in true_angles:
        plt.axvline(angle, linestyle="--", linewidth=1.0)
    for angle in estimated_angles:
        plt.axvline(angle, linestyle=":", linewidth=1.2)
    plt.xlabel("DOA (degree)")
    plt.ylabel("Normalized spectrum (dB)")
    plt.title("Last test sample")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=600, bbox_inches="tight")
    plt.close()


def main(args):
    args = apply_profile(args)
    set_seed(args.seed)
    generate_data(args)

    train_set, train_loader, val_loader, test_loader = make_loaders(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print(
        "Train={}, Val={}, Test={} | M={}, L={}, K={}-{} | Grid={} ({:.4f} deg)".format(
            len(train_loader.dataset),
            len(val_loader.dataset),
            len(test_loader.dataset),
            train_set.num_sensors,
            train_set.virtual_dim,
            train_set.min_sources,
            train_set.max_sources,
            args.grid_size,
            (args.angle_max - args.angle_min) / (args.grid_size - 1),
        )
    )
    print(
        "Forward policy: true K is never passed to model.forward(); final DOAs are "
        "peaks of an angle-preserving physical/refined spectrum."
    )

    angle_grid = torch.linspace(
        args.angle_min,
        args.angle_max,
        args.grid_size,
        dtype=torch.float32,
    )
    model = PhysicsSpectrumNet(
        num_sensors=train_set.num_sensors,
        frequencies=train_set.frequencies,
        reference_frequency=train_set.reference_frequency,
        angle_grid=angle_grid,
        min_sources=train_set.min_sources,
        max_sources=train_set.max_sources,
        snapshots=train_set.snapshots,
        hidden_dim=args.hidden_dim,
        max_spacing_error=args.max_spacing_error,
        max_phase_slope_deg=args.max_phase_slope_deg,
        residual_limit=args.residual_limit,
    ).to(device)

    checkpoint_dir = Path(args.checkpoint_dir)
    result_dir = Path(args.result_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    history = []

    model.set_trainable_stage("calibration")
    optimizer = torch.optim.Adam(
        [model.raw_spacing_error, model.raw_phase_slope],
        lr=args.calibration_lr,
    )
    stage_history, _ = train_stage(
        model,
        train_loader,
        val_loader,
        device,
        "calibration",
        args.calibration_epochs,
        optimizer,
        checkpoint_dir / "calibration_best.pth",
        args,
    )
    history.extend(stage_history)

    oracle_metrics, _ = evaluate(
        model,
        val_loader,
        device,
        args,
        spectrum_mode="physical",
        oracle_k=True,
    )
    print(
        "Oracle-K physical preflight: MAE={:.4f} deg, RMSE={:.4f} deg, "
        "within1={:.4f}".format(
            oracle_metrics["matched_mae_deg"],
            oracle_metrics["matched_rmse_deg"],
            oracle_metrics["all_sources_within_1deg_rate"],
        )
    )
    if args.require_oracle and (
        not np.isfinite(oracle_metrics["matched_mae_deg"])
        or oracle_metrics["matched_mae_deg"] > args.oracle_mae_limit
    ):
        raise RuntimeError(
            "Oracle-K physical MAE is above {:.3f} deg. The selected data "
            "conditions do not support the requested target; training was stopped.".format(
                args.oracle_mae_limit
            )
        )

    model.set_trainable_stage("count")
    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=args.count_lr,
        weight_decay=args.weight_decay,
    )
    stage_history, _ = train_stage(
        model,
        train_loader,
        val_loader,
        device,
        "count",
        args.count_epochs,
        optimizer,
        checkpoint_dir / "count_best.pth",
        args,
    )
    history.extend(stage_history)

    model.set_trainable_stage("spectrum")
    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=args.spectrum_lr,
        weight_decay=args.weight_decay,
    )
    stage_history, inference_mode = train_stage(
        model,
        train_loader,
        val_loader,
        device,
        "spectrum",
        args.spectrum_epochs,
        optimizer,
        checkpoint_dir / "spectrum_best.pth",
        args,
    )
    history.extend(stage_history)

    model.set_trainable_stage("joint")
    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=args.joint_lr,
        weight_decay=args.weight_decay,
    )
    stage_history, inference_mode = train_stage(
        model,
        train_loader,
        val_loader,
        device,
        "joint",
        args.joint_epochs,
        optimizer,
        checkpoint_dir / "joint_best.pth",
        args,
    )
    history.extend(stage_history)

    save_history(result_dir / "training_history.csv", history)

    spectrum_checkpoint = torch.load(
        checkpoint_dir / "spectrum_best.pth",
        map_location=device,
    )
    joint_checkpoint = torch.load(
        checkpoint_dir / "joint_best.pth",
        map_location=device,
    )
    if float(spectrum_checkpoint.get("score", float("inf"))) <= float(
        joint_checkpoint.get("score", float("inf"))
    ):
        selected_checkpoint_path = checkpoint_dir / "spectrum_best.pth"
    else:
        selected_checkpoint_path = checkpoint_dir / "joint_best.pth"

    checkpoint = load_checkpoint(selected_checkpoint_path, model, device)
    inference_mode = checkpoint.get("inference_mode", inference_mode)
    print("Selected checkpoint:", selected_checkpoint_path)
    print("Selected test inference mode:", inference_mode)

    metrics, last = evaluate(
        model,
        test_loader,
        device,
        args,
        spectrum_mode=inference_mode,
        oracle_k=False,
        print_samples=args.print_samples,
    )
    oracle_test_metrics, _ = evaluate(
        model,
        test_loader,
        device,
        args,
        spectrum_mode="physical",
        oracle_k=True,
        print_samples=False,
    )
    metrics["selected_inference_mode"] = inference_mode
    metrics["oracle_k_physical_mae_deg"] = oracle_test_metrics["matched_mae_deg"]
    metrics["oracle_k_physical_rmse_deg"] = oracle_test_metrics["matched_rmse_deg"]

    write_metrics(result_dir / "metrics.txt", metrics)
    plot_last(result_dir / "last_spectrum.png", last)

    if args.require_target:
        failures = []
        if not np.isfinite(metrics["matched_mae_deg"]) or metrics["matched_mae_deg"] >= args.target_mae:
            failures.append(
                "matched_mae_deg={:.4f} >= {:.4f}".format(
                    metrics["matched_mae_deg"], args.target_mae
                )
            )
        if metrics["source_count_accuracy"] < args.target_count_accuracy:
            failures.append(
                "source_count_accuracy={:.4f} < {:.4f}".format(
                    metrics["source_count_accuracy"], args.target_count_accuracy
                )
            )
        if failures:
            raise RuntimeError(
                "Target acceptance failed: " + "; ".join(failures)
            )
        print("Target acceptance passed.")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Unknown-K wideband fourth-order physics-spectrum DOA network."
        )
    )
    parser.add_argument(
        "--profile",
        choices=["quick", "target", "precision"],
        default="target",
    )
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--result_dir", default=None)
    parser.add_argument("--regenerate", action="store_true")

    for name in [
        "num_sensors",
        "min_sources",
        "max_sources",
        "snapshots",
        "num_subbands",
        "train_samples",
        "val_samples",
        "test_samples",
        "calibration_epochs",
        "count_epochs",
        "spectrum_epochs",
        "joint_epochs",
        "hidden_dim",
        "batch_size",
        "grid_size",
    ]:
        parser.add_argument("--" + name, type=int, default=None)

    for name in [
        "angle_min",
        "angle_max",
        "min_separation",
        "snr_min",
        "snr_max",
        "coherence_max",
    ]:
        parser.add_argument("--" + name, type=float, default=None)

    parser.add_argument("--frequency_min", type=float, default=700.0)
    parser.add_argument("--frequency_max", type=float, default=1300.0)
    parser.add_argument("--coherence_min", type=float, default=0.0)
    parser.add_argument("--true_spacing_error", type=float, default=0.012)
    parser.add_argument("--true_phase_slope_deg", type=float, default=1.2)
    parser.add_argument("--max_spacing_error", type=float, default=0.03)
    parser.add_argument("--max_phase_slope_deg", type=float, default=3.0)
    parser.add_argument("--num_blocks", type=int, default=4)

    parser.add_argument("--fine_sigma", type=float, default=0.12)
    parser.add_argument("--medium_sigma", type=float, default=0.32)
    parser.add_argument("--coarse_sigma", type=float, default=0.75)
    parser.add_argument("--residual_limit", type=float, default=5.0)
    parser.add_argument("--output_min_separation", type=float, default=None)

    parser.add_argument("--calibration_lr", type=float, default=5e-2)
    parser.add_argument("--count_lr", type=float, default=8e-4)
    parser.add_argument("--spectrum_lr", type=float, default=5e-4)
    parser.add_argument("--joint_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--early_stopping", type=int, default=12)

    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--generation_batch_size", type=int, default=8)
    parser.add_argument("--generation_device", default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print_samples", action="store_true")

    parser.add_argument("--require_oracle", action="store_true")
    parser.add_argument("--oracle_mae_limit", type=float, default=1.0)
    parser.add_argument("--require_target", action="store_true")
    parser.add_argument("--target_mae", type=float, default=1.0)
    parser.add_argument("--target_count_accuracy", type=float, default=0.90)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
