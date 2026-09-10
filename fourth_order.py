import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourthOrderLagEstimator(nn.Module):
    """Estimate strict fourth-order cumulant lags for a ULA.

    Input
    -----
    x : complex tensor [B, F, M, T]

    Output
    ------
    lag_mean : complex tensor [B, F, 4M-3]
    log_variance : real tensor [B, F, 4M-3]
    """

    def __init__(self, num_sensors: int):
        super().__init__()
        self.M = int(num_sensors)
        self.M2 = self.M * self.M
        self.num_lags = 4 * self.M - 3
        self.center = 2 * (self.M - 1)

        pair_i = torch.arange(self.M).repeat_interleave(self.M)
        pair_j = torch.arange(self.M).repeat(self.M)

        i = pair_i[:, None]
        j = pair_j[:, None]
        k = pair_i[None, :]
        l = pair_j[None, :]

        # E[x_i x_j* x_k* x_l] has virtual lag i-j-k+l.
        lag_index = (i - j - k + l + self.center).reshape(-1).long()
        lag_counts = torch.bincount(lag_index, minlength=self.num_lags).float()

        self.register_buffer("pair_i", pair_i.long())
        self.register_buffer("pair_j", pair_j.long())
        self.register_buffer("lag_index", lag_index)
        self.register_buffer("lag_counts", lag_counts)

    def _estimate_once(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("x must have shape [B,F,M,T].")
        if x.shape[2] != self.M:
            raise ValueError(f"Expected M={self.M}, received {x.shape[2]}.")

        x = x.to(torch.complex64)
        x = x - x.mean(dim=-1, keepdim=True)
        batch_size, num_bands, _, snapshots = x.shape
        xf = x.reshape(batch_size * num_bands, self.M, snapshots)

        covariance = xf @ xf.conj().transpose(-1, -2)
        covariance = covariance / float(max(snapshots, 1))

        pseudo_covariance = xf @ xf.transpose(-1, -2)
        pseudo_covariance = pseudo_covariance / float(max(snapshots, 1))

        quadratic = (
            xf.unsqueeze(2) * xf.conj().unsqueeze(1)
        ).reshape(batch_size * num_bands, self.M2, snapshots)

        fourth_moment = quadratic @ quadratic.conj().transpose(-1, -2)
        fourth_moment = fourth_moment / float(max(snapshots, 1))

        covariance_vector = covariance.reshape(batch_size * num_bands, self.M2)
        term_1 = covariance_vector[:, :, None] * covariance_vector.conj()[:, None, :]

        pi = self.pair_i
        pj = self.pair_j
        term_2 = (
            covariance[:, pi[:, None], pi[None, :]]
            * covariance[:, pj[None, :], pj[:, None]]
        )
        term_3 = (
            pseudo_covariance[:, pi[:, None], pj[None, :]]
            * pseudo_covariance[:, pj[:, None], pi[None, :]].conj()
        )

        cumulant = fourth_moment - term_1 - term_2 - term_3
        cumulant = cumulant.reshape(batch_size * num_bands, self.M2 * self.M2)

        lag_sum = torch.zeros(
            batch_size * num_bands,
            self.num_lags,
            dtype=cumulant.dtype,
            device=cumulant.device,
        )
        lag_sum.scatter_add_(
            1,
            self.lag_index.unsqueeze(0).expand(batch_size * num_bands, -1),
            cumulant,
        )

        lag_mean = lag_sum / self.lag_counts.clamp_min(1.0).unsqueeze(0)
        return lag_mean.reshape(batch_size, num_bands, self.num_lags)

    def forward(self, x: torch.Tensor, num_blocks: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        full_lag = self._estimate_once(x)
        snapshots = x.shape[-1]
        num_blocks = int(max(1, min(num_blocks, snapshots)))
        block_length = snapshots // num_blocks

        if num_blocks < 2 or block_length < 16:
            return full_lag, torch.zeros_like(full_lag.real)

        block_lags = []
        for block in range(num_blocks):
            start = block * block_length
            end = snapshots if block == num_blocks - 1 else (block + 1) * block_length
            block_lags.append(self._estimate_once(x[..., start:end]))

        stacked = torch.stack(block_lags, dim=0)
        variance = torch.mean(
            torch.abs(stacked - stacked.mean(dim=0, keepdim=True)) ** 2,
            dim=0,
        )
        scale = torch.mean(torch.abs(full_lag) ** 2, dim=-1, keepdim=True)
        variance = variance / (scale + 1e-8)
        return full_lag, torch.log1p(variance.clamp_min(0.0))


def safe_complex(x: torch.Tensor) -> torch.Tensor:
    return torch.complex(
        torch.nan_to_num(x.real, nan=0.0, posinf=1e4, neginf=-1e4),
        torch.nan_to_num(x.imag, nan=0.0, posinf=1e4, neginf=-1e4),
    )


def enforce_conjugate_symmetry(lags: torch.Tensor) -> torch.Tensor:
    symmetric = 0.5 * (lags + lags.flip(-1).conj())
    center = lags.shape[-1] // 2
    center_value = torch.complex(
        symmetric[..., center].real,
        torch.zeros_like(symmetric[..., center].real),
    )
    symmetric = symmetric.clone()
    symmetric[..., center] = center_value
    return symmetric


def correct_cumulant_polarity(lags: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Make the zero-lag fourth-order cumulant positive.

    QPSK and many sub-Gaussian sources have negative fourth-order cumulants.
    Without this correction the signal eigenvectors are the *smallest* (most
    negative) eigenvectors and a conventional MUSIC implementation reverses
    signal/noise subspaces. This was the dominant error in the older versions.
    """
    center = lags.shape[-1] // 2
    center_real = lags[..., center].real
    polarity = torch.where(
        center_real < 0.0,
        -torch.ones_like(center_real),
        torch.ones_like(center_real),
    )
    corrected = lags * polarity[..., None].to(lags.dtype)
    return enforce_conjugate_symmetry(corrected), polarity


def calibrated_focus_lags(
    lags: torch.Tensor,
    frequencies: torch.Tensor,
    reference_frequency: float,
    spacing_error: torch.Tensor,
    phase_slope: torch.Tensor,
) -> torch.Tensor:
    """Differentiably focus wideband C4 lags to an ideal reference ULA.

    The physical lag phase is
        -pi * (f/f0) * (1+delta_d) * d * sin(theta) + delta_phi * d.

    We query the original sequence at
        q = d / ((f/f0) * (1+delta_d))
    and remove exp(+j delta_phi q), yielding the ideal reference-frequency
    phase exp(-j pi d sin(theta)).
    """
    if lags.ndim != 3:
        raise ValueError("lags must have shape [B,F,D].")

    batch_size, num_bands, num_lags = lags.shape
    if frequencies.numel() != num_bands:
        raise ValueError("Number of frequencies does not match lags.")

    half = (num_lags - 1) / 2.0
    lag_axis = torch.linspace(
        -half, half, num_lags, device=lags.device, dtype=torch.float32
    )

    ratio = frequencies.to(lags.device, torch.float32) / float(reference_frequency)
    scale = ratio[None, :, None] * (1.0 + spacing_error.reshape(-1, 1, 1))
    query = lag_axis[None, None, :] / scale.clamp_min(0.25)
    normalized_x = (query / max(half, 1.0)).clamp(-1.15, 1.15)

    # grid_sample input [N,C,H,W], grid [N,Hout,Wout,2].
    real_input = lags.real.reshape(batch_size * num_bands, 1, 1, num_lags)
    imag_input = lags.imag.reshape(batch_size * num_bands, 1, 1, num_lags)
    x_grid = normalized_x.reshape(batch_size * num_bands, 1, num_lags)
    y_grid = torch.zeros_like(x_grid)
    grid = torch.stack([x_grid, y_grid], dim=-1)

    focused_real = F.grid_sample(
        real_input,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(batch_size, num_bands, num_lags)
    focused_imag = F.grid_sample(
        imag_input,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(batch_size, num_bands, num_lags)

    focused = torch.complex(focused_real, focused_imag)
    phase_compensation = torch.exp(
        -1j * phase_slope.reshape(-1, 1, 1) * query
    )
    focused = focused * phase_compensation.to(focused.dtype)
    return enforce_conjugate_symmetry(safe_complex(focused))


def focus_and_fuse(
    lags: torch.Tensor,
    log_variance: torch.Tensor,
    frequencies: torch.Tensor,
    reference_frequency: float,
    spacing_error: torch.Tensor,
    phase_slope: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    corrected, polarity = correct_cumulant_polarity(lags)

    center = corrected.shape[-1] // 2
    center_scale = corrected[..., center].real.abs().clamp_min(1e-4)
    normalized = corrected / center_scale[..., None].to(corrected.dtype)

    focused = calibrated_focus_lags(
        normalized,
        frequencies,
        reference_frequency,
        spacing_error,
        phase_slope,
    )

    variance = torch.nan_to_num(log_variance, nan=8.0, posinf=8.0, neginf=0.0)
    band_quality = -variance.mean(dim=-1)
    valid_fraction = (torch.abs(focused) > 1e-8).float().mean(dim=-1)
    band_logits = band_quality + 1.5 * torch.log(valid_fraction.clamp_min(1e-3))
    band_weight = torch.softmax(band_logits, dim=-1)

    fused = torch.sum(band_weight[..., None] * focused, dim=1)
    fused = enforce_conjugate_symmetry(fused)

    return {
        "corrected_lags": corrected,
        "focused_lags": focused,
        "fused_lags": fused,
        "band_weight": band_weight,
        "polarity": polarity,
    }


def lags_to_toeplitz(lags: torch.Tensor, virtual_dim: int) -> torch.Tensor:
    if lags.shape[-1] != 2 * virtual_dim - 1:
        raise ValueError("Lag length must equal 2*virtual_dim-1.")
    row = torch.arange(virtual_dim, device=lags.device)[:, None]
    col = torch.arange(virtual_dim, device=lags.device)[None, :]
    index = row - col + virtual_dim - 1
    matrix = lags[..., index]
    return 0.5 * (matrix + matrix.conj().transpose(-1, -2))


def forward_backward_average(matrix: torch.Tensor) -> torch.Tensor:
    virtual_dim = matrix.shape[-1]
    exchange = torch.eye(
        virtual_dim, dtype=matrix.dtype, device=matrix.device
    ).flip(-1)
    fb = 0.5 * (
        matrix + exchange @ matrix.conj() @ exchange
    )
    return 0.5 * (fb + fb.conj().transpose(-1, -2))


def make_psd_covariance(lags: torch.Tensor, virtual_dim: int, loading: float = 1e-3):
    matrix = lags_to_toeplitz(lags, virtual_dim)
    matrix = forward_backward_average(matrix)

    with torch.no_grad():
        values, vectors = torch.linalg.eigh(safe_complex(matrix))
        maximum = values.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        floor = float(loading) * maximum
        positive = torch.maximum(values, floor)
        matrix_psd = (
            vectors
            @ torch.diag_embed(positive.to(matrix.dtype))
            @ vectors.conj().transpose(-1, -2)
        )
        trace = torch.diagonal(matrix_psd, dim1=-2, dim2=-1).real.sum(
            dim=-1, keepdim=True
        )
        matrix_psd = matrix_psd / trace[:, None].clamp_min(1e-8)
        values, vectors = torch.linalg.eigh(matrix_psd)

    return matrix_psd, values, vectors


def ideal_virtual_manifold(angle_grid: torch.Tensor, virtual_dim: int) -> torch.Tensor:
    position = torch.arange(
        virtual_dim, device=angle_grid.device, dtype=torch.float32
    )
    theta = torch.deg2rad(angle_grid.to(torch.float32))
    return torch.exp(
        -1j * math.pi * position[:, None] * torch.sin(theta)[None, :]
    ).transpose(0, 1).to(torch.complex64)


def candidate_music_bank(
    eigenvectors: torch.Tensor,
    manifold: torch.Tensor,
    min_sources: int,
    max_sources: int,
) -> torch.Tensor:
    """MUSIC spectra for every candidate source count, without true K."""
    batch_size, virtual_dim, _ = eigenvectors.shape
    spectra = []

    with torch.no_grad():
        for source_number in range(int(min_sources), int(max_sources) + 1):
            noise_dim = virtual_dim - source_number
            noise_basis = eigenvectors[:, :, :noise_dim]
            projector = noise_basis @ noise_basis.conj().transpose(-1, -2)
            denominator = torch.einsum(
                "gl,blm,gm->bg",
                manifold.conj(),
                projector,
                manifold,
            ).real.clamp_min(1e-8)
            spectrum = 1.0 / denominator
            spectrum = spectrum / spectrum.amax(dim=-1, keepdim=True).clamp_min(1e-8)
            spectra.append(spectrum)

    return torch.stack(spectra, dim=1)
