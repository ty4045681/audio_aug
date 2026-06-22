# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "pyyaml"]
# ///
"""amp_distortion — amplitude-domain distortion variants.

A fraction ``rate`` of samples are mutated; the rest pass through unchanged.
``distortion_type`` selects the shaping applied to the mutated samples:

  * gain_db          — scale by a fixed dB gain then clip to +/-0.997.
                       params: gain_db in [-30, 30] (default 6.0)
  * max_distortion   — replace each sample with its sign times a near-full
                       amplitude (hard square-wave clamp).
                       params: max_db (default -0.03)
  * fence_distortion — keep only samples whose abs amplitude falls inside a set
                       of randomly generated dB masks; replace those with
                       sign*max_amp, zero everything else.
                       params: mask_number in [0, 12] (default 4), max_db (-0.03)
  * jag_distortion   — same masking as fence_distortion, but pass the original
                       sample through inside the masks and zero outside.
                       params: mask_number in [0, 12] (default 4)
  * poly_distortion  — polynomial waveshaper in the dB domain.
                       params: a (default 1.0), m in [1, 8] (default 1),
                       n in [1, 8] (default 1)
  * quad_distortion  — poly_distortion with a=1.0, m=1, n=1 (no extra params).

Common params: rate in (0, 1] (default 0.8), seed (default 0).

Usage (Windows, uv):
    uv run scripts/amp_distortion.py --set distortion_type=max_distortion
    uv run scripts/amp_distortion.py --set distortion_type=gain_db --set gain_db=12
    uv run scripts/amp_distortion.py --recipe recipes/amp_distortion_v1.yaml
"""

from __future__ import annotations

import numpy as np
from typing import Any

from _common import Pcm16Wav, param_float, require_range, rng_from_params, db2amp, amp2db, run_augmentation, PCM16_MAX


def _generate_amp_masks(rng, mask_number):
    if mask_number <= 0:
        return [(db2amp(left), db2amp(right)) for left, right in [(-110, -95), (-90, -80), (-65, -60), (-50, -30), (-15, 0)]]
    steps = np.concatenate(([0.0], np.cumsum(rng.uniform(0.5, 1.0, size=(2 * mask_number) - 1))))
    maximum = float(steps[-1])
    masks = []
    for index in range(mask_number):
        left_db = ((float(steps[2 * index]) - maximum) / maximum) * 100.0
        right_db = ((float(steps[(2 * index) + 1]) - maximum) / maximum) * 100.0
        masks.append((db2amp(left_db), db2amp(right_db)))
    return masks


def _inside_amp_masks(abs_samples, masks):
    included = np.zeros(abs_samples.shape, dtype=bool)
    for left, right in masks:
        included |= (abs_samples >= left) & (abs_samples <= right)
    return included


def _poly_distortion(normalized, a, m, n):
    abs_samples = np.abs(normalized)
    db_norm = np.clip((amp2db(abs_samples) / 100.0) + 1.0, 0.0, 1.0)
    shaped = np.clip(a * np.power(db_norm, m) * np.power(1.0 - db_norm, n) + db_norm, 0.0, 1.0)
    amp = np.minimum(0.9997, np.power(10.0, ((shaped - 1.0) * 100.0) / 20.0))
    return np.where(abs_samples < 1e-6, 0.0, np.sign(normalized) * amp)


def transform(wav: Pcm16Wav, params: dict[str, Any]) -> Pcm16Wav:
    distortion_type = str(params.get("distortion_type", "max_distortion"))
    rate = require_range("rate", param_float(params, "rate", 0.8), 0.0, 1.0)
    if rate <= 0.0:
        raise ValueError("rate must be greater than 0")
    rng = rng_from_params(params)
    normalized = np.clip(wav.samples / PCM16_MAX, -1.0, 1.0)
    mutate = rng.random(normalized.shape) < rate
    if normalized.size and not mutate.any():
        flat_mutate = mutate.reshape(-1)
        flat_samples = normalized.reshape(-1)
        non_silent_indexes = np.flatnonzero(np.abs(flat_samples) > 1e-12)
        candidate_indexes = non_silent_indexes if non_silent_indexes.size else np.arange(flat_mutate.size)
        flat_mutate[int(rng.choice(candidate_indexes))] = True
    if distortion_type == "gain_db":
        gain_db = require_range("gain_db", param_float(params, "gain_db", 6.0), -30.0, 30.0)
        transformed = np.clip(normalized * db2amp(gain_db), -0.997, 0.997)
    elif distortion_type == "max_distortion":
        max_amp = min(0.997, db2amp(param_float(params, "max_db", -0.03)))
        transformed = np.sign(normalized) * max_amp
    elif distortion_type in {"fence_distortion", "jag_distortion"}:
        mask_number = int(require_range("mask_number", param_float(params, "mask_number", 4), 0, 12))
        positive_mask = _generate_amp_masks(rng, mask_number)
        negative_mask = _generate_amp_masks(rng, mask_number)
        abs_samples = np.abs(normalized)
        included = np.where(normalized >= 0,
                            _inside_amp_masks(abs_samples, positive_mask),
                            _inside_amp_masks(abs_samples, negative_mask))
        if distortion_type == "fence_distortion":
            max_amp = min(0.997, db2amp(param_float(params, "max_db", -0.03)))
            transformed = np.where(included, np.sign(normalized) * max_amp, 0.0)
        else:
            transformed = np.where(included, normalized, 0.0)
    elif distortion_type in {"poly_distortion", "quad_distortion"}:
        if distortion_type == "quad_distortion":
            a, m, n = 1.0, 1, 1
        else:
            a = param_float(params, "a", 1.0)
            m = int(require_range("m", param_float(params, "m", 1), 1, 8))
            n = int(require_range("n", param_float(params, "n", 1), 1, 8))
        transformed = _poly_distortion(normalized, a, m, n)
    else:
        raise ValueError("distortion_type must be one of: gain_db, max_distortion, fence_distortion, jag_distortion, poly_distortion, quad_distortion")
    output = np.where(mutate, transformed, normalized)
    return Pcm16Wav(channels=wav.channels, sample_rate=wav.sample_rate, samples=output * PCM16_MAX)


if __name__ == "__main__":
    run_augmentation(
        name="amp_distortion",
        transform=transform,
        default_params={"distortion_type": "max_distortion", "rate": 0.8, "seed": 0},
        description="Amplitude-domain distortion: gain_db, max, fence, jag, poly, quad.",
    )
