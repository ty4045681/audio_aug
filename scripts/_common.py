"""Shared contract for all audio-augmentation scripts.

Every augmentation script imports from this module and calls ``run_augmentation``.
This module owns:

  * 16-bit PCM WAV read/write  (``read_pcm16`` / ``write_pcm16`` / ``Pcm16Wav``)
  * param parsing / validation helpers (``param_float`` / ``param_int`` /
    ``require_range`` / ``rng_from_params``)
  * DSP primitives shared by several methods (STFT/ISTFT, FFT lowpass,
    resample, channel-wise apply, length matching)
  * the batch CLI runner (``run_augmentation``) which:
      - walks the input dir (default ``raw/``) recursively for ``*.wav``
      - creates a per-batch run dir ``runs/<date>_<name>_run<NNN>/``
      - writes every output under ``<run>/audio/<relative-path>``
        keeping the ORIGINAL file name (directory = identity)
      - writes ``config.yaml`` (full params) and ``manifest.csv`` (io mapping)

A method script looks like::

    # /// script
    # requires-python = ">=3.10"
    # dependencies = ["numpy", "scipy", "pyyaml"]
    # ///
    from _common import Pcm16Wav, run_augmentation, param_float, require_range

    def transform(wav: Pcm16Wav, params: dict) -> Pcm16Wav:
        ...

    if __name__ == "__main__":
        run_augmentation(
            name="volume_gain",
            transform=transform,
            default_params={"gain_db": 6.0},
        )

Run on Windows with::

    uv run scripts/volume_gain.py --set gain_db=6
    uv run scripts/volume_gain.py --recipe recipes/volume_gain_v1.yaml
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import signal

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - pyyaml is declared per-script
    yaml = None


PCM16_MIN = -32768
PCM16_MAX = 32767


# --------------------------------------------------------------------------- #
# PCM16 WAV IO
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Pcm16Wav:
    channels: int
    sample_rate: int
    samples: np.ndarray  # shape (frames, channels), float64


def clamp_pcm16(samples: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(samples), PCM16_MIN, PCM16_MAX)


def read_pcm16(path: Path) -> Pcm16Wav:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if sample_width != 2:
        raise ValueError(f"only 16-bit PCM WAV files are supported, got sample_width={sample_width}")
    if channels <= 0:
        raise ValueError(f"invalid channel count: {channels}")
    if sample_rate <= 0:
        raise ValueError(f"invalid sample_rate: {sample_rate}")

    raw = np.frombuffer(frames, dtype="<i2").astype(np.float64)
    if raw.size % channels != 0:
        raise ValueError("PCM frame data is not aligned to channel count")
    samples = raw.reshape((-1, channels)) if raw.size else np.empty((0, channels), dtype=np.float64)
    return Pcm16Wav(channels=channels, sample_rate=sample_rate, samples=samples)


def write_pcm16(path: Path, wav_data: Pcm16Wav) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = clamp_pcm16(wav_data.samples).astype("<i2").reshape(-1).tobytes()
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(wav_data.channels)
        wav.setsampwidth(2)
        wav.setframerate(wav_data.sample_rate)
        wav.writeframes(frames)


# --------------------------------------------------------------------------- #
# param helpers
# --------------------------------------------------------------------------- #
def require_range(name: str, value: float, minimum: float, maximum: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
    return value


def param_float(params: dict[str, Any], name: str, default: float) -> float:
    raw_value = params.get(name, default)
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def param_int(params: dict[str, Any], name: str, default: int) -> int:
    raw_value = params.get(name, default)
    try:
        return int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def rng_from_params(params: dict[str, Any]) -> np.random.Generator:
    return np.random.default_rng(param_int(params, "seed", 0))


# --------------------------------------------------------------------------- #
# DSP primitives shared across methods
# --------------------------------------------------------------------------- #
def match_length(samples: np.ndarray, frame_count: int) -> np.ndarray:
    if samples.shape[0] == frame_count:
        return samples
    if samples.shape[0] > frame_count:
        return samples[:frame_count]
    if samples.shape[0] == 0:
        return np.zeros(frame_count, dtype=np.float64)
    return np.pad(samples, (0, frame_count - samples.shape[0]), mode="edge")


def apply_channelwise(wav_data: Pcm16Wav, transform: Callable[[np.ndarray], np.ndarray]) -> Pcm16Wav:
    if wav_data.samples.size == 0:
        return wav_data
    output = [
        match_length(
            np.asarray(transform(wav_data.samples[:, channel]), dtype=np.float64),
            wav_data.samples.shape[0],
        )
        for channel in range(wav_data.channels)
    ]
    return Pcm16Wav(channels=wav_data.channels, sample_rate=wav_data.sample_rate, samples=np.column_stack(output))


def stft_params(samples: np.ndarray) -> tuple[int, int, int]:
    nfft = 512
    nperseg = min(480, max(16, samples.shape[0]))
    hop = min(160, max(1, nperseg // 2))
    noverlap = nperseg - hop
    return nfft, nperseg, noverlap


def stft(samples: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int]]:
    nfft, nperseg, noverlap = stft_params(samples)
    freqs, times, spectrum = signal.stft(
        samples,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary="zeros",
        padded=True,
    )
    return freqs, times, spectrum, (nfft, nperseg, noverlap)


def istft(spectrum: np.ndarray, sample_rate: int, params: tuple[int, int, int], frame_count: int) -> np.ndarray:
    nfft, nperseg, noverlap = params
    _, output = signal.istft(
        spectrum,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        input_onesided=True,
        boundary=True,
    )
    return match_length(np.asarray(output, dtype=np.float64), frame_count)


def lowpass_fft(channel: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    spectrum = np.fft.rfft(channel)
    freqs = np.fft.rfftfreq(channel.shape[0], d=1.0 / sample_rate)
    spectrum[freqs > cutoff_hz] = 0
    return np.fft.irfft(spectrum, n=channel.shape[0])


def resample_to_rate(channel: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    down_gcd = math.gcd(source_rate, target_rate)
    downsampled = signal.resample_poly(channel, target_rate // down_gcd, source_rate // down_gcd)
    up_gcd = math.gcd(target_rate, source_rate)
    restored = signal.resample_poly(downsampled, source_rate // up_gcd, target_rate // up_gcd)
    return match_length(np.asarray(restored, dtype=np.float64), channel.shape[0])


def db2amp(db: float) -> float:
    return math.pow(10.0, db / 20.0)


def amp2db(amp: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(amp, 1e-12))


# --------------------------------------------------------------------------- #
# batch CLI runner
# --------------------------------------------------------------------------- #
def _params_id(params: dict[str, Any]) -> str:
    blob = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    return "p" + hashlib.sha1(blob).hexdigest()[:7]


def _parse_set(pairs: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--set expects key=value, got: {pair!r}")
        key, _, raw = pair.partition("=")
        key = key.strip()
        raw = raw.strip()
        # try int -> float -> json (true/false/null/strings) -> raw string
        try:
            out[key] = int(raw)
            continue
        except ValueError:
            pass
        try:
            value = float(raw)
            if math.isfinite(value):
                out[key] = value
                continue
        except ValueError:
            pass
        try:
            out[key] = json.loads(raw)
            continue
        except Exception:
            out[key] = raw
    return out


def _load_recipe(path: Path) -> tuple[dict[str, Any], str | None, str | None]:
    """Return (params, recipe_id, notes) from a recipe yaml.

    Accepts either a flat ``params:`` mapping or a single-stage
    ``pipeline: [{name, params}]`` entry.
    """
    if yaml is None:
        raise SystemExit("pyyaml is required to read recipes; add it to the script dependencies")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    recipe_id = data.get("recipe_id") or path.stem
    notes = data.get("notes")
    if "params" in data and isinstance(data["params"], dict):
        return dict(data["params"]), recipe_id, notes
    pipeline = data.get("pipeline")
    if isinstance(pipeline, list) and pipeline:
        stage = pipeline[0] or {}
        return dict(stage.get("params") or {}), recipe_id, notes
    return {}, recipe_id, notes


def _next_run_dir(runs_dir: Path, name: str, today: str, explicit: str | None) -> Path:
    if explicit:
        return runs_dir / explicit
    pattern = re.compile(rf"^{re.escape(today)}_{re.escape(name)}_run(\d+)$")
    existing = 0
    if runs_dir.exists():
        for child in runs_dir.iterdir():
            m = pattern.match(child.name)
            if m:
                existing = max(existing, int(m.group(1)))
    return runs_dir / f"{today}_{name}_run{existing + 1:04d}"


def run_augmentation(
    name: str,
    transform: Callable[[Pcm16Wav, dict[str, Any]], Pcm16Wav],
    default_params: dict[str, Any] | None = None,
    description: str | None = None,
) -> None:
    """Parse CLI args, run ``transform`` over every wav, write run artifacts."""
    default_params = dict(default_params or {})
    parser = argparse.ArgumentParser(
        prog=name,
        description=description or f"{name} audio augmentation (batch).",
    )
    parser.add_argument("--input-dir", default="raw", help="source audio root (default: raw)")
    parser.add_argument("--runs-dir", default="runs", help="output runs root (default: runs)")
    parser.add_argument("--recipe", default=None, help="recipe yaml with params/notes")
    parser.add_argument("--run-id", default=None, help="override the auto-generated run id")
    parser.add_argument("--suffix", action="store_true", help=f"append __{name} to output file names")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a single param (repeatable); wins over the recipe",
    )
    parser.add_argument("--notes", default=None, help="free-text note recorded in config.yaml")
    args = parser.parse_args()

    params = dict(default_params)
    recipe_id = "default"
    notes = args.notes
    if args.recipe:
        recipe_params, recipe_id, recipe_notes = _load_recipe(Path(args.recipe))
        params.update(recipe_params)
        if notes is None:
            notes = recipe_notes
    params.update(_parse_set(args.overrides))

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"input dir not found: {input_dir}")
    sources = sorted(
        (p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"),
        key=lambda p: p.as_posix(),
    )
    if not sources:
        raise SystemExit(f"no .wav files found under {input_dir}")

    today = _dt.date.today().isoformat()
    run_dir = _next_run_dir(Path(args.runs_dir), name, today, args.run_id)
    run_id = run_dir.name
    audio_dir = run_dir / "audio"
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    params_id = _params_id(params)

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    seen_outputs: set[Path] = set()
    for source in sources:
        rel = source.relative_to(input_dir)
        out_name = rel.stem + (f"__{name}" if args.suffix else "") + ".wav"
        out_path = audio_dir / rel.parent / out_name
        if out_path in seen_outputs:
            failures.append(f"{source}: output path collides with an earlier file -> {out_path}")
            print(f"  ! FAILED {rel}: output path collides with an earlier file -> {out_path}")
            continue
        seen_outputs.add(out_path)
        try:
            wav_data = read_pcm16(source)
            result = transform(wav_data, dict(params))
            write_pcm16(out_path, result)
        except Exception as exc:  # keep going; record the failure
            failures.append(f"{source}: {exc}")
            print(f"  ! FAILED {rel}: {exc}")
            continue
        duration = (wav_data.samples.shape[0] / wav_data.sample_rate) if wav_data.sample_rate else 0.0
        rows.append(
            {
                "source_path": source.as_posix(),
                "output_path": out_path.as_posix(),
                "recipe_id": recipe_id,
                "run_id": run_id,
                "enhancement_chain": name,
                "params_id": params_id,
                "duration_sec": f"{duration:.3f}",
                "sample_rate": wav_data.sample_rate,
            }
        )
        print(f"  ok {rel} -> {out_path.relative_to(run_dir)}")

    manifest_path = run_dir / "manifest.csv"
    fieldnames = [
        "source_path",
        "output_path",
        "recipe_id",
        "run_id",
        "enhancement_chain",
        "params_id",
        "duration_sec",
        "sample_rate",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    config = {
        "run_id": run_id,
        "input_dir": input_dir.as_posix(),
        "output_dir": audio_dir.as_posix(),
        "audio_format": "wav",
        "bit_depth": 16,
        "recipe_id": recipe_id,
        "params_id": params_id,
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "pipeline": [{"name": name, "params": params}],
        "files_total": len(sources),
        "files_ok": len(rows),
        "files_failed": len(failures),
        "notes": notes,
    }
    config_path = run_dir / "config.yaml"
    if yaml is not None:
        config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    else:  # pragma: no cover
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    if failures:
        (run_dir / "logs" / "failures.log").write_text("\n".join(failures), encoding="utf-8")

    print(
        f"\n[{name}] run_id={run_id}  ok={len(rows)}/{len(sources)}  "
        f"failed={len(failures)}  params_id={params_id}\n  -> {run_dir}"
    )
