import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fourth_order import (
    candidate_music_bank,
    focus_and_fuse,
    ideal_virtual_manifold,
    make_psd_covariance,
)


class Residual1D(nn.Module):
    def __init__(self, channels, dilation=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        groups = 8 if channels % 8 == 0 else 4
        self.norm = nn.GroupNorm(groups, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = self.conv2(self.dropout(F.gelu(self.conv1(x))))
        return F.gelu(self.norm(x + residual))


class PhysicsOrderHead(nn.Module):
    """Physics-dominant source-count estimator.

    Each candidate K is scored from its eigengap, MDL-like criterion, tail
    flatness, and MUSIC-peak statistics. A small learned residual is added to
    these structured candidate scores. This is substantially more stable than
    classifying a globally pooled arbitrary latent vector.
    """

    def __init__(
        self,
        virtual_dim,
        min_sources,
        max_sources,
        effective_snapshots,
        hidden_dim=128,
    ):
        super().__init__()
        self.L = int(virtual_dim)
        self.Kmin = int(min_sources)
        self.Kmax = int(max_sources)
        self.C = self.Kmax - self.Kmin + 1
        self.effective_snapshots = float(max(effective_snapshots, 8))

        candidate_feature_dim = 10
        self.candidate_scorer = nn.Sequential(
            nn.Linear(candidate_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        bank_channels = max(32, hidden_dim // 2)
        self.bank_encoder = nn.Sequential(
            nn.Conv1d(self.C, bank_channels, 9, padding=4),
            nn.GELU(),
            Residual1D(bank_channels, dilation=1, dropout=0.05),
            Residual1D(bank_channels, dilation=2, dropout=0.05),
            Residual1D(bank_channels, dilation=4, dropout=0.05),
        )
        self.global_residual = nn.Sequential(
            nn.Linear(2 * bank_channels + 2 * self.L, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, self.C),
        )

        self.mdl_scale = nn.Parameter(torch.tensor(1.0))
        self.gap_scale = nn.Parameter(torch.tensor(1.0))
        self.peak_scale = nn.Parameter(torch.tensor(0.5))

    @staticmethod
    def _standardize(values):
        return (
            values - values.mean(dim=-1, keepdim=True)
        ) / values.std(dim=-1, keepdim=True).clamp_min(1e-5)

    def _candidate_features(self, eigenvalues, music_bank):
        # torch.linalg.eigh returns ascending eigenvalues.
        descending = eigenvalues.flip(-1).clamp_min(1e-10)
        descending = descending / descending[:, :1].clamp_min(1e-10)
        log_values = torch.log(descending.clamp_min(1e-10))

        local_max = music_bank >= F.max_pool1d(
            music_bank,
            kernel_size=7,
            stride=1,
            padding=3,
        ) - 1e-7
        local_values = torch.where(
            local_max,
            music_bank,
            torch.zeros_like(music_bank),
        )
        maximum_peaks = min(self.Kmax + 2, music_bank.shape[-1])
        peak_values = torch.topk(
            local_values,
            k=maximum_peaks,
            dim=-1,
        ).values

        features = []
        mdl_scores = []
        gap_scores = []
        peak_consistency = []

        for candidate_index, source_number in enumerate(
            range(self.Kmin, self.Kmax + 1)
        ):
            k = int(source_number)
            signal = descending[:, :k]
            noise = descending[:, k:]

            boundary_gap = (
                log_values[:, k - 1] - log_values[:, k]
            )
            signal_margin = (
                log_values[:, k - 1]
                - log_values[:, -1]
            )
            noise_log = torch.log(noise.clamp_min(1e-10))
            tail_std = noise_log.std(dim=-1, unbiased=False)
            tail_mean = noise_log.mean(dim=-1)
            signal_mean = torch.log(signal.clamp_min(1e-10)).mean(dim=-1)

            arithmetic = noise.mean(dim=-1).clamp_min(1e-10)
            geometric = torch.exp(
                torch.log(noise.clamp_min(1e-10)).mean(dim=-1)
            )
            mdl = (
                -self.effective_snapshots
                * (self.L - k)
                * torch.log((geometric / arithmetic).clamp_min(1e-10))
                + 0.5
                * k
                * (2 * self.L - k)
                * math.log(self.effective_snapshots)
            )
            mdl_score = -mdl

            spectrum = music_bank[:, candidate_index]
            probability = spectrum / spectrum.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-8)
            entropy = -torch.sum(
                probability * torch.log(probability.clamp_min(1e-8)),
                dim=-1,
            ) / math.log(float(spectrum.shape[-1]))

            kth = peak_values[:, candidate_index, k - 1]
            next_index = min(k, maximum_peaks - 1)
            next_peak = peak_values[:, candidate_index, next_index]
            top_mean = peak_values[:, candidate_index, :k].mean(dim=-1)
            strong_count = (
                local_values[:, candidate_index] > 0.30
            ).float().sum(dim=-1) / float(max(self.Kmax, 1))
            peak_margin = kth - next_peak

            feature = torch.stack(
                [
                    boundary_gap,
                    signal_margin,
                    signal_mean,
                    tail_mean,
                    -tail_std,
                    mdl_score / self.effective_snapshots,
                    1.0 - entropy,
                    top_mean,
                    kth,
                    peak_margin + 0.1 * strong_count,
                ],
                dim=-1,
            )
            features.append(feature)
            mdl_scores.append(mdl_score)
            gap_scores.append(boundary_gap)
            peak_consistency.append(peak_margin + 0.15 * top_mean)

        return (
            torch.stack(features, dim=1),
            torch.stack(mdl_scores, dim=1),
            torch.stack(gap_scores, dim=1),
            torch.stack(peak_consistency, dim=1),
            log_values,
        )

    def forward(self, eigenvalues, music_bank):
        (
            candidate_features,
            mdl_score,
            gap_score,
            peak_score,
            log_values,
        ) = self._candidate_features(eigenvalues, music_bank)

        structured = self.candidate_scorer(
            candidate_features
        ).squeeze(-1)

        encoded = self.bank_encoder(torch.sqrt(music_bank.clamp_min(0.0)))
        average_pool = encoded.mean(dim=-1)
        maximum_pool = encoded.amax(dim=-1)
        eigengaps = torch.zeros_like(log_values)
        eigengaps[:, :-1] = log_values[:, :-1] - log_values[:, 1:]
        global_feature = torch.cat(
            [average_pool, maximum_pool, log_values, eigengaps],
            dim=-1,
        )
        residual = self.global_residual(global_feature)

        logits = (
            structured
            + residual
            + F.softplus(self.mdl_scale) * self._standardize(mdl_score)
            + F.softplus(self.gap_scale) * self._standardize(gap_score)
            + F.softplus(self.peak_scale) * self._standardize(peak_score)
        )
        return logits, candidate_features


class AnglePreservingSpectrumRefiner(nn.Module):
    """Refine a physical MUSIC spectrum without destroying angle geometry."""

    def __init__(self, input_channels, hidden_dim=96, residual_limit=5.0):
        super().__init__()
        channels = max(48, hidden_dim)
        self.residual_limit = float(residual_limit)

        self.input = nn.Conv1d(input_channels, channels, 9, padding=4)
        self.blocks = nn.Sequential(
            Residual1D(channels, dilation=1, dropout=0.05),
            Residual1D(channels, dilation=2, dropout=0.05),
            Residual1D(channels, dilation=4, dropout=0.05),
            Residual1D(channels, dilation=8, dropout=0.05),
            Residual1D(channels, dilation=16, dropout=0.05),
            Residual1D(channels, dilation=32, dropout=0.05),
            Residual1D(channels, dilation=8, dropout=0.05),
            Residual1D(channels, dilation=2, dropout=0.05),
        )
        self.output = nn.Conv1d(channels, 1, 5, padding=2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, music_bank, selected_spectrum):
        selected = selected_spectrum.clamp(1e-5, 1.0 - 1e-5)
        maximum = music_bank.amax(dim=1)
        average = music_bank.mean(dim=1)
        first_difference = F.pad(
            selected[:, 1:] - selected[:, :-1],
            (1, 0),
        )
        second_difference = F.pad(
            first_difference[:, 1:] - first_difference[:, :-1],
            (1, 0),
        )

        inputs = torch.cat(
            [
                torch.sqrt(music_bank.clamp_min(0.0)),
                selected[:, None, :],
                maximum[:, None, :],
                average[:, None, :],
                first_difference[:, None, :],
                second_difference[:, None, :],
            ],
            dim=1,
        )

        residual = self.residual_limit * torch.tanh(
            self.output(self.blocks(F.gelu(self.input(inputs))))
        ).squeeze(1)

        base_logit = torch.logit(selected)
        refined_logit = base_logit + residual
        refined = torch.sigmoid(refined_logit)
        refined = refined / refined.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        return refined_logit, refined, residual


class PhysicsSpectrumNet(nn.Module):
    """Unknown-K wideband fourth-order DOA estimator.

    The model never receives true K in forward propagation. It first builds a
    sign-corrected C4-MUSIC bank for every candidate K, estimates source count
    from structured model-order features, and refines the selected physical
    spectrum with an angle-preserving dilated CNN. DOAs are extracted as peaks,
    not directly regressed from a globally pooled feature vector.
    """

    def __init__(
        self,
        num_sensors,
        frequencies,
        reference_frequency,
        angle_grid,
        min_sources,
        max_sources,
        snapshots,
        hidden_dim=96,
        max_spacing_error=0.03,
        max_phase_slope_deg=3.0,
        residual_limit=5.0,
    ):
        super().__init__()
        self.M = int(num_sensors)
        self.L = 2 * self.M - 1
        self.Kmin = int(min_sources)
        self.Kmax = int(max_sources)
        self.C = self.Kmax - self.Kmin + 1
        self.snapshots = int(snapshots)

        if self.Kmax >= self.L:
            raise ValueError("max_sources must be smaller than L=2M-1.")

        frequencies = torch.as_tensor(frequencies, dtype=torch.float32)
        angle_grid = torch.as_tensor(angle_grid, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies)
        self.register_buffer("angle_grid", angle_grid)
        self.reference_frequency = float(reference_frequency)
        self.register_buffer(
            "ideal_manifold",
            ideal_virtual_manifold(angle_grid, self.L),
        )

        self.max_spacing_error = float(max_spacing_error)
        self.max_phase_slope = math.radians(float(max_phase_slope_deg))
        self.raw_spacing_error = nn.Parameter(torch.zeros(1))
        self.raw_phase_slope = nn.Parameter(torch.zeros(1))

        self.order_head = PhysicsOrderHead(
            virtual_dim=self.L,
            min_sources=self.Kmin,
            max_sources=self.Kmax,
            effective_snapshots=self.snapshots * frequencies.numel(),
            hidden_dim=max(96, hidden_dim),
        )
        self.refiner = AnglePreservingSpectrumRefiner(
            input_channels=self.C + 5,
            hidden_dim=hidden_dim,
            residual_limit=residual_limit,
        )

    def calibration_values(self):
        spacing = self.max_spacing_error * torch.tanh(
            self.raw_spacing_error
        )
        phase = self.max_phase_slope * torch.tanh(
            self.raw_phase_slope
        )
        return spacing, phase

    def set_trainable_stage(self, stage):
        for parameter in self.parameters():
            parameter.requires_grad = False

        if stage == "calibration":
            self.raw_spacing_error.requires_grad = True
            self.raw_phase_slope.requires_grad = True
        elif stage == "count":
            for parameter in self.order_head.parameters():
                parameter.requires_grad = True
        elif stage == "spectrum":
            for parameter in self.refiner.parameters():
                parameter.requires_grad = True
        elif stage == "joint":
            for name, parameter in self.named_parameters():
                if name not in (
                    "raw_spacing_error",
                    "raw_phase_slope",
                ):
                    parameter.requires_grad = True
        else:
            raise ValueError("Unknown stage: {}".format(stage))

    def _physics_frontend(self, lags, log_variance):
        spacing, phase = self.calibration_values()
        batch_size = lags.shape[0]
        with torch.no_grad():
            focused = focus_and_fuse(
                lags,
                log_variance,
                self.frequencies,
                self.reference_frequency,
                spacing.expand(batch_size),
                phase.expand(batch_size),
            )
            covariance, eigenvalues, eigenvectors = make_psd_covariance(
                focused["fused_lags"],
                self.L,
                loading=5e-4,
            )
            music_bank = candidate_music_bank(
                eigenvectors,
                self.ideal_manifold,
                self.Kmin,
                self.Kmax,
            )
        return focused, covariance, eigenvalues, eigenvectors, music_bank

    def forward(self, lags, log_variance, mode="full"):
        spacing, phase = self.calibration_values()
        if mode == "calibration":
            return {
                "spacing_error": spacing,
                "phase_slope": phase,
                "angle_grid": self.angle_grid,
            }

        (
            focused,
            covariance,
            eigenvalues,
            eigenvectors,
            music_bank,
        ) = self._physics_frontend(lags, log_variance)

        count_logits, candidate_features = self.order_head(
            eigenvalues,
            music_bank,
        )
        count_probability = torch.softmax(count_logits, dim=-1)
        predicted_class = count_probability.argmax(dim=-1)
        predicted_k = predicted_class + self.Kmin

        if mode == "count":
            return {
                "spacing_error": spacing,
                "phase_slope": phase,
                "virtual_covariance": covariance,
                "eigenvalues": eigenvalues,
                "eigenvectors": eigenvectors,
                "music_bank": music_bank,
                "candidate_features": candidate_features,
                "count_logits": count_logits,
                "count_probability": count_probability,
                "predicted_k": predicted_k,
                "predicted_class": predicted_class,
                "angle_grid": self.angle_grid,
            }

        batch_index = torch.arange(
            lags.shape[0],
            device=lags.device,
        )
        selected_spectrum = music_bank[
            batch_index,
            predicted_class,
        ]

        refined_logit, refined_spectrum, residual = self.refiner(
            music_bank,
            selected_spectrum,
        )

        return {
            "spacing_error": spacing,
            "phase_slope": phase,
            "corrected_lags": focused["corrected_lags"],
            "focused_lags": focused["focused_lags"],
            "fused_lags": focused["fused_lags"],
            "band_weight": focused["band_weight"],
            "polarity": focused["polarity"],
            "virtual_covariance": covariance,
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "music_bank": music_bank,
            "candidate_features": candidate_features,
            "count_logits": count_logits,
            "count_probability": count_probability,
            "predicted_k": predicted_k,
            "predicted_class": predicted_class,
            "physical_spectrum": selected_spectrum,
            "refined_logits": refined_logit,
            "refined_spectrum": refined_spectrum,
            "spectrum_residual": residual,
            "angle_grid": self.angle_grid,
        }
