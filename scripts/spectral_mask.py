# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "pyyaml"]
# ///
"""spectral_mask — SpecAugment-style time/frequency masking in the STFT domain.

Each channel is transformed to its complex STFT, then a multiplicative gain
mask is built over the (frequency, time) grid:

  * ``frequency_masks`` horizontal bands of random width are attenuated. Each
    band spans between ``len(freqs)//24`` and ``len(freqs)//5`` frequency bins
    and is multiplied by a random gain drawn from ``[min_gain, max_gain]``.
  * ``time_masks`` vertical bands of random width (up to ``len(times)//4``
    frames) are attenuated the same way.

Bands may overlap, so their gains compound. The masked spectrum is inverted
back to PCM with ISTFT, preserving the original frame count. All randomness is
seeded via ``seed`` so runs are reproducible.

Faithful to the SpecAugment masking idea but using soft (gain) masking rather
than hard zeroing, which avoids harsh spectral discontinuities.

Usage (uv):
    uv run scripts/spectral_mask.py --set seed=1
    uv run scripts/spectral_mask.py --set frequency_masks=3 --set time_masks=2
"""

from __future__ import annotations

from typing import Any

import numpy as np

from _common import (
    Pcm16Wav,
    apply_channelwise,
    istft,
    param_float,
    require_range,
    rng_from_params,
    run_augmentation,
    stft,
)


def transform(wav: Pcm16Wav, params: dict[str, Any]) -> Pcm16Wav:
    rng = rng_from_params(params)
    frequency_masks = int(require_range("frequency_masks", param_float(params, "frequency_masks", 2), 1, 8))
    time_masks = int(require_range("time_masks", param_float(params, "time_masks", 1), 0, 8))
    min_gain = require_range("min_gain", param_float(params, "min_gain", 0.05), 0.0, 1.0)
    max_gain = require_range("max_gain", param_float(params, "max_gain", 0.6), min_gain, 1.0)

    def channel_transform(channel: np.ndarray) -> np.ndarray:
        freqs, times, spectrum, stft_p = stft(channel, wav.sample_rate)
        mask = np.ones(spectrum.shape, dtype=np.float64)
        for _ in range(frequency_masks):
            width = int(rng.integers(max(1, len(freqs) // 24), max(2, len(freqs) // 5)))
            start = int(rng.integers(0, max(1, len(freqs) - width + 1)))
            mask[start:start + width, :] *= float(rng.uniform(min_gain, max_gain))
        for _ in range(time_masks):
            width = int(rng.integers(1, max(2, len(times) // 4 + 1)))
            start = int(rng.integers(0, max(1, len(times) - width + 1)))
            mask[:, start:start + width] *= float(rng.uniform(min_gain, max_gain))
        return istft(spectrum * mask, wav.sample_rate, stft_p, channel.shape[0])

    return apply_channelwise(wav, channel_transform)


if __name__ == "__main__":
    run_augmentation(
        name="spectral_mask",
        transform=transform,
        default_params={"frequency_masks": 2, "time_masks": 1, "min_gain": 0.05, "max_gain": 0.6, "seed": 0},
        description="SpecAugment-style time/frequency masking in the STFT domain.",
    )
