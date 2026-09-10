import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from fourth_order import FourthOrderLagEstimator


DATA_VERSION = 6


class WidebandC4Dataset(Dataset):
    def __init__(self, path):
        data = torch.load(path, map_location="cpu")
        if int(data.get("data_version", -1)) != DATA_VERSION:
            raise RuntimeError(
                "{} is not a v{} dataset. Use --regenerate and do not reuse "
                "older data files.".format(path, DATA_VERSION)
            )

        self.lags = data["lags"].to(torch.complex64)
        self.log_variance = data["log_variance"].to(torch.float32)
        self.doa = data["doa"].to(torch.float32)
        self.source_number = data["source_number"].to(torch.long)

        self.num_sensors = int(data["num_sensors"])
        self.virtual_dim = int(data["virtual_dim"])
        self.min_sources = int(data["min_sources"])
        self.max_sources = int(data["max_sources"])
        self.snapshots = int(data["snapshots"])
        self.frequencies = data["frequencies"].to(torch.float32)
        self.reference_frequency = float(data["reference_frequency"])
        self.true_spacing_error = float(data["true_spacing_error"])
        self.true_phase_slope = float(data["true_phase_slope"])
        self.angle_min = float(data["angle_min"])
        self.angle_max = float(data["angle_max"])
        self.min_separation = float(data["min_separation"])
        self.snr_min = float(data["snr_min"])
        self.snr_max = float(data["snr_max"])
        self.coherence_min = float(data["coherence_min"])
        self.coherence_max = float(data["coherence_max"])

    def __len__(self):
        return self.lags.shape[0]

    def __getitem__(self, index):
        return (
            self.lags[index],
            self.log_variance[index],
            self.doa[index],
            self.source_number[index],
            torch.tensor(self.true_spacing_error, dtype=torch.float32),
            torch.tensor(self.true_phase_slope, dtype=torch.float32),
        )


class WidebandC4DataGenerator:
    """Generate controlled wideband non-Gaussian array observations.

    The source-count classes are strictly balanced. All samples share the same
    physical array calibration parameters, which makes the calibration stage
    identifiable and physically meaningful.
    """

    def __init__(
        self,
        num_samples,
        num_sensors=8,
        snapshots=1536,
        min_sources=2,
        max_sources=10,
        num_subbands=8,
        frequency_min=700.0,
        frequency_max=1300.0,
        angle_min=-60.0,
        angle_max=60.0,
        min_separation=5.0,
        snr_min=26.0,
        snr_max=36.0,
        coherence_min=0.0,
        coherence_max=0.35,
        true_spacing_error=0.012,
        true_phase_slope_deg=1.2,
        num_blocks=4,
        seed=42,
    ):
        self.num_samples = int(num_samples)
        self.M = int(num_sensors)
        self.T = int(snapshots)
        self.Kmin = int(min_sources)
        self.Kmax = int(max_sources)
        self.F = int(num_subbands)

        self.angle_min = float(angle_min)
        self.angle_max = float(angle_max)
        self.min_separation = float(min_separation)
        self.snr_min = float(snr_min)
        self.snr_max = float(snr_max)
        self.coherence_min = float(coherence_min)
        self.coherence_max = float(coherence_max)

        self.true_spacing_error = float(true_spacing_error)
        self.true_phase_slope = math.radians(float(true_phase_slope_deg))
        self.num_blocks = int(num_blocks)

        self.frequencies = torch.linspace(
            frequency_min,
            frequency_max,
            self.F,
            dtype=torch.float32,
        )
        self.reference_frequency = float(self.frequencies.mean().item())
        self.virtual_dim = 2 * self.M - 1

        self.rng = np.random.default_rng(seed)
        self.torch_generator = torch.Generator().manual_seed(seed)

        if self.Kmax >= self.virtual_dim:
            raise ValueError(
                "max_sources={} must be smaller than L={}.".format(
                    self.Kmax, self.virtual_dim
                )
            )

        if (self.Kmax - 1) * self.min_separation > (
            self.angle_max - self.angle_min
        ):
            raise ValueError(
                "The source count and minimum separation do not fit the angle interval."
            )

    def _sample_doa(self, source_number):
        for _ in range(30000):
            doa = np.sort(
                self.rng.uniform(
                    self.angle_min,
                    self.angle_max,
                    source_number,
                )
            )
            if np.all(np.diff(doa) >= self.min_separation):
                return doa.astype(np.float32)
        raise RuntimeError("Could not sample separated DOAs.")

    def _qpsk(self, shape):
        real = 2 * torch.randint(
            0,
            2,
            shape,
            generator=self.torch_generator,
        ) - 1
        imag = 2 * torch.randint(
            0,
            2,
            shape,
            generator=self.torch_generator,
        ) - 1
        return (real + 1j * imag).to(torch.complex64) / math.sqrt(2.0)

    def _observation(self, doa, source_number):
        ratio = self.frequencies / self.reference_frequency
        sensor = torch.arange(self.M, dtype=torch.float32)
        theta = torch.deg2rad(torch.from_numpy(doa))

        phase = (
            -1j
            * math.pi
            * ratio[:, None, None]
            * (1.0 + self.true_spacing_error)
            * sensor[None, :, None]
            * torch.sin(theta)[None, None, :]
            + 1j
            * self.true_phase_slope
            * sensor[None, :, None]
        )
        steering = torch.exp(phase).to(torch.complex64)

        coherence = float(
            self.rng.uniform(self.coherence_min, self.coherence_max)
        )
        source = torch.zeros(
            self.F,
            source_number,
            self.T,
            dtype=torch.complex64,
        )

        for band in range(self.F):
            independent = self._qpsk((source_number, self.T))
            common = self._qpsk((1, self.T))
            common_phase = torch.exp(
                1j
                * 2.0
                * math.pi
                * torch.rand(
                    source_number,
                    1,
                    generator=self.torch_generator,
                )
            )
            source[band] = (
                math.sqrt(max(1.0 - coherence, 0.0)) * independent
                + math.sqrt(max(coherence, 0.0)) * common_phase * common
            )

        gain = torch.exp(
            0.08
            * torch.randn(
                self.F,
                source_number,
                1,
                generator=self.torch_generator,
            )
        ).to(torch.complex64)
        source = source * gain

        observation = torch.einsum("fmk,fkt->fmt", steering, source)

        snr_db = float(self.rng.uniform(self.snr_min, self.snr_max))
        signal_power = torch.mean(
            torch.abs(observation) ** 2,
            dim=(-2, -1),
        )
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))

        noise = (
            torch.randn(
                self.F,
                self.M,
                self.T,
                generator=self.torch_generator,
            )
            + 1j
            * torch.randn(
                self.F,
                self.M,
                self.T,
                generator=self.torch_generator,
            )
        ).to(torch.complex64)
        noise = noise * torch.sqrt(noise_power[:, None, None] / 2.0)
        return observation + noise

    def generate(self, save_path, batch_size=8, device=None):
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            device = torch.device(device)

        estimator = FourthOrderLagEstimator(self.M).to(device)

        source_numbers = np.arange(self.Kmin, self.Kmax + 1)
        source_numbers = np.resize(source_numbers, self.num_samples)
        self.rng.shuffle(source_numbers)

        lag_list = []
        variance_list = []
        doa_list = []
        source_number_list = []
        underdetermined = 0

        for start in range(0, self.num_samples, batch_size):
            end = min(start + batch_size, self.num_samples)
            observations = []
            padded_doa = []
            batch_source_number = []

            for index in range(start, end):
                source_number = int(source_numbers[index])
                doa = self._sample_doa(source_number)
                observations.append(self._observation(doa, source_number))

                padded = torch.full(
                    (self.Kmax,),
                    float("nan"),
                    dtype=torch.float32,
                )
                padded[:source_number] = torch.from_numpy(doa)
                padded_doa.append(padded)
                batch_source_number.append(source_number)
                underdetermined += int(source_number > self.M)

            observation = torch.stack(observations, dim=0).to(device)
            with torch.no_grad():
                lags, log_variance = estimator(
                    observation,
                    num_blocks=self.num_blocks,
                )

            lag_list.append(lags.cpu())
            variance_list.append(log_variance.cpu())
            doa_list.append(torch.stack(padded_doa))
            source_number_list.append(
                torch.tensor(batch_source_number, dtype=torch.long)
            )
            print("C4 preprocessing: {}/{}".format(end, self.num_samples))

        data = {
            "data_version": DATA_VERSION,
            "lags": torch.cat(lag_list, dim=0),
            "log_variance": torch.cat(variance_list, dim=0),
            "doa": torch.cat(doa_list, dim=0),
            "source_number": torch.cat(source_number_list, dim=0),
            "num_sensors": self.M,
            "virtual_dim": self.virtual_dim,
            "min_sources": self.Kmin,
            "max_sources": self.Kmax,
            "snapshots": self.T,
            "frequencies": self.frequencies,
            "reference_frequency": self.reference_frequency,
            "true_spacing_error": self.true_spacing_error,
            "true_phase_slope": self.true_phase_slope,
            "angle_min": self.angle_min,
            "angle_max": self.angle_max,
            "min_separation": self.min_separation,
            "snr_min": self.snr_min,
            "snr_max": self.snr_max,
            "coherence_min": self.coherence_min,
            "coherence_max": self.coherence_max,
        }

        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, path)
        print(
            "Saved {} | samples={} | underdetermined K>M={}".format(
                path,
                self.num_samples,
                underdetermined,
            )
        )
