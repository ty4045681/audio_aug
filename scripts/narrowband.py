# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "pyyaml"]
# ///
"""narrowband — simulate telephone/narrowband audio via resample down-and-back.

Downsamples each channel to ``target_sample_rate`` and then upsamples it back to
the original rate. The round-trip discards high-frequency content above the
target band's Nyquist limit, reproducing the muffled quality of telephone-band
(narrowband) speech while keeping the file's original sample rate.

``target_sample_rate`` must satisfy ``1000 <= rate < source_sample_rate``
(the test wavs are 16 kHz, so the 8000 Hz default yields a telephone band).

Usage (Windows, uv):
    uv run scripts/narrowband.py --set target_sample_rate=8000
    uv run scripts/narrowband.py --recipe recipes/narrowband_v1.yaml
"""

from __future__ import annotations

from typing import Any

from _common import Pcm16Wav, param_int, apply_channelwise, resample_to_rate, run_augmentation


def transform(wav: Pcm16Wav, params: dict[str, Any]) -> Pcm16Wav:
    target_rate = param_int(params, "target_sample_rate", 8000)
    if not 1000 <= target_rate < wav.sample_rate:
        raise ValueError("target_sample_rate must be lower than the source sample rate")
    return apply_channelwise(wav, lambda channel: resample_to_rate(channel, wav.sample_rate, target_rate))


if __name__ == "__main__":
    run_augmentation(
        name="narrowband",
        transform=transform,
        default_params={"target_sample_rate": 8000},
        description="Simulate narrowband/telephone audio by resampling down to target_sample_rate and back.",
    )
