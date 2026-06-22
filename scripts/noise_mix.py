# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "pyyaml"]
# ///
"""noise_mix — add Gaussian (white) noise at a target signal-to-noise ratio.

Faithful to kws_testset ``_noise_mix``: the noise standard deviation is chosen
so the added noise sits ``snr_db`` decibels below the signal RMS, i.e.
``noise_rms = signal_rms / 10**(snr_db/20)``. Lower snr_db means louder noise.
``snr_db`` is clamped/validated to [-5, 40]. A reproducible RNG is derived from
the ``seed`` param via ``rng_from_params``. Silent input (signal RMS == 0) falls
back to a fixed noise RMS so the output is not left untouched.

Usage (uv):
    uv run scripts/noise_mix.py --set snr_db=10 --set seed=1
    uv run scripts/noise_mix.py --recipe recipes/noise_mix_v1.yaml
"""

from __future__ import annotations

from typing import Any

import numpy as np

from _common import (
    Pcm16Wav,
    db2amp,
    param_float,
    require_range,
    rng_from_params,
    run_augmentation,
)


def transform(wav: Pcm16Wav, params: dict[str, Any]) -> Pcm16Wav:
    snr_db = require_range("snr_db", param_float(params, "snr_db", 20.0), -5.0, 40.0)
    if wav.samples.size == 0:
        return wav
    rng = rng_from_params(params)
    signal_rms = float(np.sqrt(np.mean(np.square(wav.samples))))
    noise_rms = signal_rms / db2amp(snr_db) if signal_rms > 0 else 100.0
    noise = rng.normal(0.0, noise_rms, size=wav.samples.shape)
    return Pcm16Wav(
        channels=wav.channels,
        sample_rate=wav.sample_rate,
        samples=wav.samples + noise,
    )


if __name__ == "__main__":
    run_augmentation(
        name="noise_mix",
        transform=transform,
        default_params={"snr_db": 15.0, "seed": 0},
        description="Add Gaussian noise at a target signal-to-noise ratio (dB).",
    )
