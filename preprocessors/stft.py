import numpy as np
import torch
import torch.nn as nn
from scipy import signal

from .cache import cached_stft

SAMPLE_RATE_HZ = 2048


def _first(arr, axis):
    # from https://github.com/scipy/scipy/blob/v1.9.0/scipy/stats/_stats_py.py#L2662-L2730
    """Return arr[..., 0:1, ...] where 0:1 is in the `axis` position."""
    return np.take_along_axis(arr, np.array(0, ndmin=arr.ndim), axis)


def zscore(a, axis):
    # from https://github.com/scipy/scipy/blob/v1.9.0/scipy/stats/_stats_py.py#L2662-L2730
    mn = a.mean(axis=axis, keepdims=True)
    std = a.std(axis=axis, ddof=0, keepdims=True)

    std[(std == 0)] = (
        1.0  # this is a hack. I should eventually find where the bad data is
    )
    z = (a - mn) / std
    return z


class STFTPreprocessor(nn.Module):
    def get_stft(self, x, fs, show_fs=-1, normalizing=None, **kwargs):
        f, t, Zxx = signal.stft(x, fs, **kwargs)

        if "return_onesided" in kwargs and kwargs["return_onesided"] == True:
            Zxx = Zxx[:show_fs]
            f = f[:show_fs]
        else:
            pass  # TODO
            # Zxx = np.concatenate([Zxx[:,:,:show_fs], Zxx[:,:,-show_fs:]], axis=-1)
            # f = np.concatenate([f[:show_fs], f[-show_fs:]], axis=-1)

        Zxx = np.abs(Zxx)

        if normalizing == "zscore":
            Zxx = zscore(
                Zxx, axis=-1
            )  # TODO is this order correct? I put it this way to prevent input nans
            if (Zxx.std() == 0).any():
                Zxx = np.ones_like(Zxx)
            Zxx = Zxx[:, 10:-10]
        elif normalizing == "db":
            Zxx = np.log(Zxx)

        if np.isnan(Zxx).any():
            Zxx = np.nan_to_num(Zxx, nan=0.0)

        return f, t, torch.Tensor(np.transpose(Zxx))

    def __init__(self, cfg):
        super(STFTPreprocessor, self).__init__()
        self.cfg = cfg

    def frequency_axis(self) -> np.ndarray:
        """Return actual one-sided STFT bin centers, shape (frequencies,)."""
        return np.fft.rfftfreq(self.cfg.nperseg, d=1 / SAMPLE_RATE_HZ)[
            : self.cfg.freq_channel_cutoff
        ]

    def unnormalized(self, wav: np.ndarray) -> torch.Tensor:
        """Return magnitude STFT (frames, frequencies) after edge trimming.

        Parameters
        ----------
        wav : numpy.ndarray
            One electrode's waveform, shape (samples,).

        Returns
        -------
        torch.Tensor
            Finite float32 magnitude spectrum, shape (frames, frequencies).
        """
        _, _, linear = self.get_stft(
            wav,
            SAMPLE_RATE_HZ,
            show_fs=self.cfg.freq_channel_cutoff,
            nperseg=self.cfg.nperseg,
            noverlap=self.cfg.noverlap,
            normalizing=None,
            return_onesided=True,
        )
        if linear.shape[0] <= 20 or not torch.isfinite(linear).all():
            raise ValueError("STFT needs more than 20 finite time frames.")
        return linear[10:-10]

    def forward(self, wav: np.ndarray, statistics: dict | None = None):
        mode = self.cfg.normalizing
        if (
            mode in {"recording_zscore", "dataset_zscore"}
            and statistics is None
        ):
            raise ValueError(
                "STFT normalization requires fitted frequency statistics "
                "from a dataset with explicit training membership."
            )
        identity = statistics["identity"] if statistics is not None else None
        if self.cfg.get("feature_cache"):
            return cached_stft(
                wav,
                self.cfg,
                lambda: self.extract(wav, statistics),
                normalization_identity=identity,
            )
        return self.extract(wav, statistics)

    def extract(self, wav: np.ndarray, statistics: dict | None = None):
        """Normalize one (frames, frequencies) STFT with explicit mode."""
        mode = self.cfg.normalizing
        if mode in {"recording_zscore", "dataset_zscore"}:
            if statistics is None:
                raise ValueError(
                    "STFT requires explicit fitted frequency statistics."
                )
            spectrum = self.unnormalized(wav)
            means = np.asarray(statistics["mean"], dtype=np.float64)
            stds = np.asarray(statistics["std"], dtype=np.float64)
            if means.shape != spectrum.shape[1:] or stds.shape != means.shape:
                raise ValueError(
                    "Recording statistics must have one mean and standard "
                    "deviation per STFT frequency."
                )
            if (
                not np.isfinite(means).all()
                or not np.isfinite(stds).all()
                or np.any(stds <= 0)
            ):
                raise ValueError(
                    "Recording STFT statistics must be finite and "
                    "have positive standard deviations."
                )
            output = (spectrum.double().numpy() - means) / stds
            if not np.isfinite(output).all():
                raise ValueError("Normalized recording STFT is not finite.")
            return torch.from_numpy(output.astype(np.float32))
        if mode not in {"zscore", "window_zscore", "db"}:
            raise ValueError(
                f"Unknown STFT normalization: {self.cfg.normalizing!r}"
            )
        _, _, linear = self.get_stft(
            wav,
            SAMPLE_RATE_HZ,
            show_fs=self.cfg.freq_channel_cutoff,
            nperseg=self.cfg.nperseg,
            noverlap=self.cfg.noverlap,
            normalizing="zscore" if mode == "window_zscore" else mode,
            return_onesided=True,
        )  # TODO hardcode sampling rate
        return linear
