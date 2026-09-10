import torch
import torch.nn.functional as F


def build_target_spectrum(doa, source_number, angle_grid, sigma):
    target = torch.zeros(
        doa.shape[0],
        angle_grid.numel(),
        dtype=torch.float32,
        device=doa.device,
    )
    for batch_index in range(doa.shape[0]):
        k = int(source_number[batch_index].item())
        distance = angle_grid[None, :] - doa[batch_index, :k, None]
        peaks = torch.exp(-0.5 * (distance / float(sigma)) ** 2)
        target[batch_index] = peaks.amax(dim=0)
    return target


def calibration_loss(
    output,
    true_spacing,
    true_phase,
    max_spacing,
    max_phase,
):
    spacing_target = true_spacing.mean()
    phase_target = true_phase.mean()

    spacing = (
        (output["spacing_error"] - spacing_target)
        / max(float(max_spacing), 1e-6)
    ) ** 2
    phase = (
        (output["phase_slope"] - phase_target)
        / max(float(max_phase), 1e-6)
    ) ** 2
    total = spacing.mean() + phase.mean()
    return {
        "total": total,
        "calibration": total,
        "spacing": spacing.mean(),
        "phase": phase.mean(),
    }


def count_loss(output, source_number, min_sources):
    target = source_number - int(min_sources)
    cross_entropy = F.cross_entropy(
        output["count_logits"],
        target,
        label_smoothing=0.01,
    )

    candidate_k = torch.arange(
        output["count_probability"].shape[-1],
        device=source_number.device,
        dtype=torch.float32,
    ) + float(min_sources)
    expected = torch.sum(
        output["count_probability"] * candidate_k[None, :],
        dim=-1,
    )
    regression = F.smooth_l1_loss(
        expected,
        source_number.float(),
        beta=0.25,
    )
    total = cross_entropy + 0.20 * regression
    accuracy = (
        output["predicted_k"] == source_number
    ).float().mean()
    return {
        "total": total,
        "count": cross_entropy,
        "count_regression": regression,
        "count_accuracy": accuracy,
    }


def sample_spectrum_at_angles(spectrum, angles, angle_min, angle_max):
    normalized = 2.0 * (
        (angles - float(angle_min))
        / max(float(angle_max - angle_min), 1e-6)
    ) - 1.0
    input_tensor = spectrum[:, None, None, :]
    x_grid = normalized[:, None, :, None]
    y_grid = torch.zeros_like(x_grid)
    grid = torch.cat([x_grid, y_grid], dim=-1)
    sampled = F.grid_sample(
        input_tensor,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled[:, 0, 0, :]


def _weighted_bce(prediction, target):
    prediction = prediction.clamp(1e-5, 1.0 - 1e-5)
    positive_weight = 1.0 + 24.0 * target
    return -(
        positive_weight * target * torch.log(prediction)
        + (1.0 - target) * torch.log(1.0 - prediction)
    ).mean()


def _dice_loss(prediction, target):
    intersection = torch.sum(prediction * target, dim=-1)
    denominator = (
        torch.sum(prediction, dim=-1)
        + torch.sum(target, dim=-1)
        + 1e-6
    )
    return 1.0 - torch.mean(
        (2.0 * intersection + 1e-6) / denominator
    )


def local_softargmax_loss(
    spectrum,
    doa,
    source_number,
    angle_grid,
    window_radius=1.0,
    temperature=0.05,
):
    losses = []
    absolute_errors = []

    for batch_index in range(spectrum.shape[0]):
        k = int(source_number[batch_index].item())
        for source_index in range(k):
            true_angle = doa[batch_index, source_index]
            mask = torch.abs(angle_grid - true_angle) <= float(window_radius)
            local_angle = angle_grid[mask]
            local_spectrum = spectrum[batch_index, mask]

            if local_angle.numel() < 2:
                continue

            weight = torch.softmax(
                local_spectrum / max(float(temperature), 1e-4),
                dim=-1,
            )
            estimate = torch.sum(weight * local_angle)
            error = torch.abs(estimate - true_angle)
            losses.append(F.smooth_l1_loss(
                estimate,
                true_angle,
                beta=0.10,
            ))
            absolute_errors.append(error)

    if not losses:
        zero = spectrum.new_tensor(0.0)
        return zero, zero

    return torch.stack(losses).mean(), torch.stack(absolute_errors).mean()


def spectrum_loss(
    output,
    doa,
    source_number,
    fine_sigma=0.12,
    medium_sigma=0.32,
    coarse_sigma=0.75,
    exclusion_radius=0.65,
):
    spectrum = output["refined_spectrum"]
    physical = output["physical_spectrum"]
    residual = output["spectrum_residual"]
    angle_grid = output["angle_grid"]

    coarse_target = build_target_spectrum(
        doa,
        source_number,
        angle_grid,
        coarse_sigma,
    )
    medium_target = build_target_spectrum(
        doa,
        source_number,
        angle_grid,
        medium_sigma,
    )
    fine_target = build_target_spectrum(
        doa,
        source_number,
        angle_grid,
        fine_sigma,
    )

    coarse_prediction = F.avg_pool1d(
        spectrum[:, None, :],
        kernel_size=9,
        stride=1,
        padding=4,
    ).squeeze(1)
    medium_prediction = F.avg_pool1d(
        spectrum[:, None, :],
        kernel_size=5,
        stride=1,
        padding=2,
    ).squeeze(1)

    coarse = _weighted_bce(coarse_prediction, coarse_target)
    medium = _weighted_bce(medium_prediction, medium_target)
    fine = _weighted_bce(spectrum, fine_target)
    dice = _dice_loss(spectrum, fine_target)

    peak_losses = []
    shoulder_losses = []
    hard_negative_losses = []

    for batch_index in range(spectrum.shape[0]):
        k = int(source_number[batch_index].item())
        true_angles = doa[batch_index, :k]
        sampled_peak = sample_spectrum_at_angles(
            spectrum[batch_index:batch_index + 1],
            true_angles[None, :],
            float(angle_grid[0]),
            float(angle_grid[-1]),
        )[0]
        peak_losses.append(torch.mean((1.0 - sampled_peak) ** 2))

        left_angles = true_angles - 0.45
        right_angles = true_angles + 0.45
        left = sample_spectrum_at_angles(
            spectrum[batch_index:batch_index + 1],
            left_angles[None, :],
            float(angle_grid[0]),
            float(angle_grid[-1]),
        )[0]
        right = sample_spectrum_at_angles(
            spectrum[batch_index:batch_index + 1],
            right_angles[None, :],
            float(angle_grid[0]),
            float(angle_grid[-1]),
        )[0]
        shoulder = torch.maximum(left, right)
        shoulder_losses.append(
            F.relu(shoulder + 0.12 - sampled_peak).mean()
        )

        background_mask = torch.ones_like(
            angle_grid,
            dtype=torch.bool,
        )
        for source_index in range(k):
            background_mask &= (
                torch.abs(angle_grid - true_angles[source_index])
                > float(exclusion_radius)
            )
        negatives = spectrum[batch_index, background_mask]
        if negatives.numel() > 0:
            number = min(64, negatives.numel())
            hard_negative_losses.append(
                torch.topk(negatives, k=number).values.mean()
            )

    peak = torch.stack(peak_losses).mean()
    shoulder = torch.stack(shoulder_losses).mean()
    hard_negative = (
        torch.stack(hard_negative_losses).mean()
        if hard_negative_losses
        else spectrum.new_tensor(0.0)
    )

    local_angle, local_mae = local_softargmax_loss(
        spectrum,
        doa,
        source_number,
        angle_grid,
        window_radius=1.0,
        temperature=0.04,
    )

    # The residual is bounded and lightly regularized so the learned spectrum
    # cannot arbitrarily destroy the analytical MUSIC structure.
    residual_regularization = torch.mean(residual ** 2)
    preservation = torch.mean(
        torch.abs(spectrum - physical)
        * (1.0 - coarse_target)
    )

    total = (
        0.25 * coarse
        + 0.40 * medium
        + 1.00 * fine
        + 0.50 * dice
        + 1.50 * peak
        + 0.75 * shoulder
        + 0.50 * hard_negative
        + 2.00 * local_angle
        + 0.002 * residual_regularization
        + 0.05 * preservation
    )

    return {
        "total": total,
        "spectrum": fine,
        "coarse": coarse,
        "medium": medium,
        "dice": dice,
        "peak": peak,
        "shoulder": shoulder,
        "hard_negative": hard_negative,
        "local_angle": local_angle,
        "local_mae": local_mae,
        "residual_regularization": residual_regularization,
        "preservation": preservation,
    }


def joint_loss(
    output,
    doa,
    source_number,
    min_sources,
    fine_sigma=0.12,
    medium_sigma=0.32,
    coarse_sigma=0.75,
):
    count = count_loss(output, source_number, min_sources)
    spectrum = spectrum_loss(
        output,
        doa,
        source_number,
        fine_sigma=fine_sigma,
        medium_sigma=medium_sigma,
        coarse_sigma=coarse_sigma,
    )
    total = count["total"] + spectrum["total"]
    result = {"total": total}
    for key, value in count.items():
        if key != "total":
            result[key] = value
    for key, value in spectrum.items():
        if key != "total":
            result[key] = value
    return result
