# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "pyyaml"]
# ///
"""speed_change — resample in time to play faster/slower (pitch shifts too).

Faithful to kws_testset ``_speed_change``: the waveform is linearly
resampled along the time axis so its duration is divided by
``speed_factor``. A factor > 1 makes the clip shorter (faster), a factor
< 1 makes it longer (slower); the sample rate is unchanged, so pitch
shifts along with speed (no time-stretch/pitch-preservation). The factor
lives in [0.5, 2.0] and must not be 1 (a 1.0 factor is a no-op variant).

Usage (Windows, uv):
    uv run scripts/speed_change.py --set speed_factor=1.2
    uv run scripts/speed_change.py --recipe recipes/speed_change_v1.yaml
"""

from __future__ import annotations

from typing import Any

import numpy as np

from _common import Pcm16Wav, param_float, require_range, run_augmentation


def transform(wav: Pcm16Wav, params: dict[str, Any]) -> Pcm16Wav:
    speed_factor = require_range("speed_factor", param_float(params, "speed_factor", 1.0), 0.5, 2.0)
    if speed_factor == 1.0:
        raise ValueError("speed_factor must not be 1 for a generated variant")
    if wav.samples.size == 0:
        return wav
    frame_count = wav.samples.shape[0]
    output_frame_count = max(1, int(round(frame_count / speed_factor)))
    source_positions = np.arange(frame_count, dtype=np.float64)
    output_positions = np.clip(
        np.arange(output_frame_count, dtype=np.float64) * speed_factor, 0, frame_count - 1
    )
    output = [
        np.interp(output_positions, source_positions, wav.samples[:, channel])
        for channel in range(wav.channels)
    ]
    return Pcm16Wav(
        channels=wav.channels,
        sample_rate=wav.sample_rate,
        samples=np.column_stack(output),
    )


if __name__ == "__main__":
    run_augmentation(
        name="speed_change",
        transform=transform,
        default_params={"speed_factor": 1.1},
        description="Resample in time to play faster/slower (pitch shifts with speed).",
    )
