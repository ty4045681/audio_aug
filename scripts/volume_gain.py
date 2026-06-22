# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "pyyaml"]
# ///
"""volume_gain — scale amplitude by a fixed dB gain.

Faithful to kws_testset ``_volume_gain``: factor = 10**(gain_db/20),
gain_db in [-30, 30] and must not be 0 (a 0 dB gain is a no-op variant).

Usage (Windows, uv):
    uv run scripts/volume_gain.py --set gain_db=6
    uv run scripts/volume_gain.py --recipe recipes/volume_gain_v1.yaml
"""

from __future__ import annotations

from typing import Any

from _common import Pcm16Wav, db2amp, param_float, require_range, run_augmentation


def transform(wav: Pcm16Wav, params: dict[str, Any]) -> Pcm16Wav:
    gain_db = require_range("gain_db", param_float(params, "gain_db", 0.0), -30.0, 30.0)
    if gain_db == 0.0:
        raise ValueError("gain_db must not be 0 for a generated variant")
    factor = db2amp(gain_db)
    return Pcm16Wav(channels=wav.channels, sample_rate=wav.sample_rate, samples=wav.samples * factor)


if __name__ == "__main__":
    run_augmentation(
        name="volume_gain",
        transform=transform,
        default_params={"gain_db": 6.0},
        description="Scale amplitude by a fixed dB gain.",
    )
