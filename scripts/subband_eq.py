# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "pyyaml"]
# ///
"""subband_eq — random per-subband attenuation in the STFT domain.

Splits the STFT frequency axis into a randomly-sized set of contiguous
subbands (narrow at low frequencies, progressively wider toward the top),
then attenuates each subband by a random gain drawn uniformly in dB. The
four highest bands use a deeper attenuation floor (``high_min_gain_db``)
than the lower bands (``low_min_gain_db``), emphasizing high-frequency
roll-off. Each gain is in ``[min_db, 0.0]``, so the effect only ever
attenuates, never boosts.

Usage (uv):
    uv run scripts/subband_eq.py --set seed=1
    uv run scripts/subband_eq.py --set low_min_gain_db=-6 --set high_min_gain_db=-15
"""

from __future__ import annotations

from typing import Any

import numpy as np

from _common import (
    Pcm16Wav,
    apply_channelwise,
    db2amp,
    istft,
    param_float,
    rng_from_params,
    run_augmentation,
    stft,
)


def _generate_subband_bins(rng: np.random.Generator, bin_count: int) -> list[int]:
    boundaries = [0, 1]
    for index in range(2, 11):
        if index < 4:
            step = int(rng.integers(1, 3))
        elif index < 7:
            step = int(rng.integers(8, 10))
        elif index < 10:
            step = int(rng.integers(32, 64))
        else:
            step = bin_count
        boundaries.append(min(bin_count, boundaries[-1] + step))
    boundaries[-1] = bin_count
    deduped: list[int] = []
    for boundary in boundaries:
        if not deduped or boundary > deduped[-1]:
            deduped.append(boundary)
    if deduped[-1] != bin_count:
        deduped.append(bin_count)
    return deduped


def transform(wav: Pcm16Wav, params: dict[str, Any]) -> Pcm16Wav:
    rng = rng_from_params(params)
    low_min_db = param_float(params, "low_min_gain_db", -10.0)
    high_min_db = param_float(params, "high_min_gain_db", -20.0)

    def channel_transform(channel: np.ndarray) -> np.ndarray:
        freqs, _, spectrum, stft_p = stft(channel, wav.sample_rate)
        boundaries = _generate_subband_bins(rng, len(freqs))
        gains = np.ones(len(freqs), dtype=np.float64)
        for band_index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            min_db = high_min_db if band_index >= len(boundaries) - 4 else low_min_db
            gain_db = float(rng.uniform(min_db, 0.0))
            gains[start:end] = db2amp(gain_db)
        return istft(spectrum * gains[:, None], wav.sample_rate, stft_p, channel.shape[0])

    return apply_channelwise(wav, channel_transform)


if __name__ == "__main__":
    run_augmentation(
        name="subband_eq",
        transform=transform,
        default_params={"low_min_gain_db": -10.0, "high_min_gain_db": -20.0, "seed": 0},
        description="Random per-subband attenuation in the STFT domain.",
    )
