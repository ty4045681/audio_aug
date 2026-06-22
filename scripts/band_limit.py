# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "pyyaml"]
# ///
"""band_limit — restrict signal bandwidth via low-pass band limiting.

Three modes (param ``mode``, default ``freq``):

  * ``freq``      brick-wall low-pass in the FFT domain: bins above
                  ``cutoff_hz`` are zeroed (``lowpass_fft``). Sharpest
                  cutoff, no phase distortion.
  * ``iir``       Butterworth IIR low-pass of order ``filter_order``
                  (1..12, default 6). Zero-phase ``sosfiltfilt`` when the
                  channel is long enough (> order*6 samples), otherwise a
                  single-pass ``sosfilt`` to stay numerically safe.
  * ``resample``  decimate to ``target_sample_rate`` then restore the
                  original rate (``resample_to_rate``), discarding content
                  above the new Nyquist. Defaults to ``cutoff_hz * 2``,
                  clamped to [1000, source_rate - 1).

Params:
    mode               freq | iir | resample (default: freq)
    cutoff_hz          20 .. (nyquist - 1); default min(4000, nyquist-1)
    filter_order       1 .. 12 (iir only; default 6)
    target_sample_rate resample only; default cutoff_hz*2, clamped

Usage (uv):
    uv run scripts/band_limit.py --set mode=freq --set cutoff_hz=3400
    uv run scripts/band_limit.py --set mode=iir --set cutoff_hz=3400 --set filter_order=6
    uv run scripts/band_limit.py --set mode=resample --set cutoff_hz=4000
"""

from __future__ import annotations

from typing import Any

from scipy import signal

from _common import (
    Pcm16Wav,
    apply_channelwise,
    lowpass_fft,
    param_float,
    param_int,
    require_range,
    resample_to_rate,
    run_augmentation,
)


def transform(wav: Pcm16Wav, params: dict[str, Any]) -> Pcm16Wav:
    mode = str(params.get("mode", params.get("limit", "freq")))
    nyquist = wav.sample_rate / 2
    cutoff_hz = require_range(
        "cutoff_hz",
        param_float(params, "cutoff_hz", min(4000.0, nyquist - 1)),
        20.0,
        nyquist - 1,
    )

    if mode == "freq":
        return apply_channelwise(wav, lambda channel: lowpass_fft(channel, wav.sample_rate, cutoff_hz))

    if mode == "iir":
        order = int(require_range("filter_order", param_float(params, "filter_order", 6), 1, 12))
        sos = signal.butter(order, cutoff_hz / nyquist, btype="lowpass", output="sos")

        def ch(channel):
            if channel.shape[0] > order * 6:
                return signal.sosfiltfilt(sos, channel)
            return signal.sosfilt(sos, channel)

        return apply_channelwise(wav, ch)

    if mode == "resample":
        target_rate = param_int(
            params,
            "target_sample_rate",
            max(1000, min(wav.sample_rate - 1, int(cutoff_hz * 2))),
        )
        if not 1000 <= target_rate < wav.sample_rate:
            raise ValueError("target_sample_rate must be lower than the source sample rate")
        return apply_channelwise(wav, lambda channel: resample_to_rate(channel, wav.sample_rate, target_rate))

    raise ValueError("mode must be one of: freq, iir, resample")


if __name__ == "__main__":
    run_augmentation(
        name="band_limit",
        transform=transform,
        default_params={"mode": "freq", "cutoff_hz": 3400.0},
        description="Restrict signal bandwidth via low-pass band limiting (freq/iir/resample).",
    )
