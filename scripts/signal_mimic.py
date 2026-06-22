# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "pyyaml"]
# ///
"""signal_mimic — probabilistic chain of degradations mimicking real-world signal loss.

Randomly composes the sibling augmentations to imitate the cumulative damage a
signal accrues on its way to a recorder: spectral coloration, dropouts, limited
bandwidth, masking, and telephone-band narrowing. Each stage fires independently
with its own probability (all reproducibly seeded via ``seed``):

  * ``subband_probability``       random per-subband attenuation (subband_eq)
  * ``mute_probability``          zero out a random contiguous segment (dropout)
  * ``band_limit_probability``    brick-wall low-pass at a random cutoff (band_limit)
  * ``spectral_mask_probability`` SpecAugment-style time/freq masking (spectral_mask)
  * ``narrowband_probability``    telephone-band resample down-and-back (narrowband)

Each child augmentation receives a fresh seed derived from the master RNG so the
chain stays deterministic for a given ``seed``. If no stage happens to fire, a
single subband_eq pass is applied as a guaranteed fallback so the output is never
identical to the input.

Usage (uv):
    uv run scripts/signal_mimic.py --set seed=1
    uv run scripts/signal_mimic.py --set band_limit_probability=0.8 --set seed=7
"""

from __future__ import annotations

from typing import Any

from _common import Pcm16Wav, param_float, rng_from_params, run_augmentation
from subband_eq import transform as subband_eq
from spectral_mask import transform as spectral_mask
from band_limit import transform as band_limit
from narrowband import transform as narrowband


def transform(wav: Pcm16Wav, params: dict[str, Any]) -> Pcm16Wav:
    rng = rng_from_params(params)
    output = wav
    applied = False

    def child_seed() -> int:
        return int(rng.integers(0, 2**31 - 1))

    if rng.random() < param_float(params, "subband_probability", 0.4):
        output = subband_eq(output, {"seed": child_seed()})
        applied = True

    if output.samples.size and rng.random() < param_float(params, "mute_probability", 0.1):
        samples = output.samples.copy()
        frame_count = samples.shape[0]
        start = int(rng.integers(0, max(1, frame_count // 2)))
        length = int(rng.integers(max(1, frame_count // 20), max(2, frame_count // 4)))
        samples[start:min(frame_count, start + length), :] = 0
        output = Pcm16Wav(channels=output.channels, sample_rate=output.sample_rate, samples=samples)
        applied = True

    if rng.random() < param_float(params, "band_limit_probability", 0.5):
        cutoff_upper = int(min((output.sample_rate / 2) - 1, 4500))
        if cutoff_upper >= 20:
            cutoff_lower = min(3000, max(20, int(cutoff_upper * 0.5)))
            cutoff_hz = int(rng.integers(cutoff_lower, cutoff_upper + 1))
            output = band_limit(output, {"mode": "freq", "cutoff_hz": cutoff_hz})
            applied = True

    if rng.random() < param_float(params, "spectral_mask_probability", 0.1):
        output = spectral_mask(output, {"seed": child_seed()})
        applied = True

    if rng.random() < param_float(params, "narrowband_probability", 0.2) and output.sample_rate > 8000:
        output = narrowband(output, {"target_sample_rate": 8000})
        applied = True

    if not applied:
        output = subband_eq(output, {"seed": child_seed()})

    return output


if __name__ == "__main__":
    run_augmentation(
        name="signal_mimic",
        transform=transform,
        default_params={
            "subband_probability": 0.4,
            "mute_probability": 0.1,
            "band_limit_probability": 0.5,
            "spectral_mask_probability": 0.1,
            "narrowband_probability": 0.2,
            "seed": 0,
        },
        description="Probabilistic chain of degradations mimicking real-world signal loss.",
    )
