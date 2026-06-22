# audio_aug — 语音数据增强工程

把 9 种音频增强手段做成**独立的 uv 脚本**，批量处理 `raw/` 下的 16-bit PCM WAV，
结果按"处理方案 / 实验批次"保存到 `runs/`，参数写入 `config.yaml`，输入输出映射写入 `manifest.csv`。
算法实现忠实移植自 `kws_testset` 仓库的 `audio_transform_service`。

## 核心原则

> **原始音频不覆盖；增强结果按批次保存；参数进配置文件和清单，不堆进文件名。**

## 目录结构

```text
audio_aug/
├── raw/                      # 原始音频（只读，永不修改），按说话人/数据集分子目录
│   └── speaker001/0001.wav
├── recipes/                  # 增强方案参数模板（每种手段一个 v1）
│   ├── volume_gain_v1.yaml
│   └── ...
├── scripts/                  # 9 个增强脚本 + 共享库 _common.py
│   ├── _common.py            # PCM IO / DSP 原语 / 批处理 CLI 运行器（勿改契约）
│   ├── volume_gain.py  speed_change.py  noise_mix.py
│   ├── subband_eq.py   band_limit.py    narrowband.py
│   ├── spectral_mask.py amp_distortion.py signal_mimic.py
├── runs/                     # 每次批处理生成一个独立批次目录
│   └── 2026-06-22_volume_gain_run001/
│       ├── config.yaml       # 本批次完整参数
│       ├── manifest.csv      # 每个输出对应的原文件 / 参数 / 时长
│       ├── logs/             # 失败记录（如有）
│       └── audio/            # 结果音频，保留原始相对路径与文件名
│           └── speaker001/0001.wav
└── metadata/
    └── raw_manifest.csv      # 原始音频清单
```

输出音频**保留原始文件名**——批次目录本身即标识来源方案与批次：
`runs/2026-06-22_volume_gain_run001/audio/speaker001/0001.wav` 天然表示
"`0001.wav` 经 `volume_gain` 方案在 `run001` 批次生成"。

## 环境要求

- Windows + [uv](https://docs.astral.sh/uv/)。每个脚本头部用 PEP 723 内联声明依赖
  （`numpy`、`scipy`、`pyyaml`），`uv run` 会自动建临时环境，无需手动 `pip install`。
- 输入必须是 **16-bit PCM WAV**（单/多声道均可，采样率任意）。

## 用法

放原始音频到 `raw/`（可按说话人/集合分子目录），然后：

```bat
:: 1) 用命令行参数直接指定
uv run scripts\volume_gain.py --set gain_db=6
uv run scripts\noise_mix.py --set snr_db=15 --set seed=1

:: 2) 用 recipe 模板（推荐，参数可复用、可追溯）
uv run scripts\band_limit.py --recipe recipes\band_limit_v1.yaml

:: 3) recipe 打底，再用 --set 覆盖个别参数
uv run scripts\noise_mix.py --recipe recipes\noise_mix_v1.yaml --set snr_db=5
```

每次运行会在 `runs/` 下创建 `<日期>_<方法>_run<NNN>/`（NNN 自动递增），写入
`config.yaml` + `manifest.csv` + `audio/`。

### 通用命令行参数（所有脚本一致）

| 参数 | 说明 | 默认 |
|---|---|---|
| `--input-dir DIR` | 原始音频根目录（递归找 `*.wav`） | `raw` |
| `--runs-dir DIR` | 批次输出根目录 | `runs` |
| `--recipe FILE` | 读取 yaml 中的 `params:`/`notes:` | 无 |
| `--set KEY=VALUE` | 覆盖单个参数，可重复；优先级高于 recipe | 无 |
| `--run-id ID` | 指定批次目录名（默认自动按日期编号） | 自动 |
| `--suffix` | 输出文件名追加 `__<方法>` | 关 |
| `--notes "..."` | 写入 `config.yaml` 的备注 | 无 |

> 处理过程中单个文件失败不会中断整批：失败计入 `config.yaml` 的 `files_failed`
> 并记到 `logs/failures.log`，其余文件照常输出。

## 9 种增强手段与参数

| 脚本 | 说明 | 主要参数（默认） | 取值范围 |
|---|---|---|---|
| `volume_gain` | 固定 dB 音量增益 | `gain_db`(6) | -30~30，且 ≠0 |
| `speed_change` | 线性插值变速（变调） | `speed_factor`(1.1) | 0.5~2.0，且 ≠1 |
| `noise_mix` | 按 SNR 叠加高斯白噪声 | `snr_db`(15) `seed`(0) | -5~40 |
| `subband_eq` | STFT 域随机子带衰减 | `low_min_gain_db`(-10) `high_min_gain_db`(-20) `seed` | — |
| `band_limit` | 低通限带，3 模式 | `mode`(freq) `cutoff_hz` `filter_order`(6) `target_sample_rate` | mode∈{freq,iir,resample}；cutoff 20~nyquist-1；order 1~12 |
| `narrowband` | 降采样再升采样（窄带电话） | `target_sample_rate`(8000) | 1000~源采样率-1 |
| `spectral_mask` | SpecAugment 式时/频掩蔽 | `frequency_masks`(2) `time_masks`(1) `min_gain`(0.05) `max_gain`(0.6) `seed` | f:1~8 t:0~8 gain:0~1 |
| `amp_distortion` | 多种幅度失真/削波 | `distortion_type`(max_distortion) `rate`(0.8) `seed` | 见下 |
| `signal_mimic` | 按概率随机复合多种退化 | `*_probability` 见 recipe `seed` | 各为 0~1 概率 |

**`band_limit` 模式**：`freq`(FFT 置零) / `iir`(Butterworth) / `resample`(降采样往返)。

**`amp_distortion` 失真类型 (`distortion_type`)**：
`gain_db`(增益后硬削波) / `max_distortion`(全推到最大幅度) /
`fence_distortion`(掩码内推最大、其余置0) / `jag_distortion`(掩码内保留、其余置0) /
`poly_distortion`(多项式整形，参数 `a`/`m`/`n`) / `quad_distortion`(固定 a=1,m=1,n=1)。
`rate` 控制被处理样点比例；`mask_number`(fence/jag) 控制随机幅度掩码数。

> 所有带 `seed` 的方法用 `numpy.random.default_rng(seed)` 保证结果可复现。

## manifest.csv 字段

`source_path, output_path, recipe_id, run_id, enhancement_chain, params_id, duration_sec, sample_rate`

其中 `params_id` 是参数集合的短哈希（`p` + sha1 前 7 位），便于区分同一方案的不同参数组合。
