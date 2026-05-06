# DiffSinger Unit Frontend + DDSP-SVC Backend Notes

## Goal

This experiment changes the DiffSinger acoustic target from mel-spectrograms to
ContentVec units, then uses a frozen DDSP-SVC backend to render audio.

The intended split is:

```text
lyrics / phonemes / notes / durations / f0
  -> DiffSinger acoustic frontend
  -> ContentVec units (+ volume)
  -> DDSP-SVC backend
  -> mel / waveform
```

The DDSP-SVC model is kept as the acoustic backend. The DiffSinger side should
stay close to the original acoustic task, so this can be tested as a separate
frontend model without deeply rewriting DiffSinger.

## Implemented Components

- `modules/contentvec.py`
  - Loads ContentVec checkpoints.
  - Supports the DDSP-SVC-style `contentvec768l12tta2x` extractor.
  - Provides unit extraction and volume extraction for binarization.

- `preprocessing/unit_binarizer.py`
  - Builds normal DiffSinger acoustic binary data.
  - Adds `units` and `volume` extracted from audio.

- `training/unit_acoustic_task.py`
  - Trains the acoustic model to predict ContentVec units instead of mel.
  - Optionally predicts volume as an extra output channel.
  - Reuses DiffSinger's existing acoustic model stack as much as possible.

- `configs/unit_acoustic.yaml`
  - Base configuration for unit prediction.

- `configs/unit_acoustic_p40.yaml`
  - Smaller P40-friendly configuration for architecture validation.

- `scripts/debug/vocode_units.py`
  - Sends predicted units through the DDSP-SVC backend.

- `scripts/debug/vocode_binary_units.py`
  - Sends ground-truth binary units through the DDSP-SVC backend.
  - Used to verify that the backend side is wired correctly.

- `scripts/debug/analyze_units.py`
  - Compares predicted units against ground-truth binary units.
  - Reports unit statistics, frame cosine, L1/L2, delta statistics, volume
    statistics, and f0 statistics.

- `scripts/debug/predict_binary_units.py`
  - Runs a trained unit acoustic checkpoint on one binary item.
  - Saves a `.units.pt`-compatible payload for analysis or DDSP-SVC vocoding.

## Current Validation Result

The DDSP-SVC backend was first tested with ground-truth binary units:

```bash
python scripts/debug/vocode_binary_units.py \
  --binary-data-dir data/unit_frontend/binary \
  --prefix valid \
  --name 2001000001 \
  --ddsp-svc ../DDSP-SVC \
  --model /root/DiffSinger/checkpoints/ddspmodel/model_1600.pt \
  --out outputs/gt_2001000001.wav \
  --spk-id 1 \
  --infer-step 20
```

Result: the output had clear voice and intelligible semantics. This means the
DDSP-SVC backend, HifiGAN path, f0, volume, and speaker id wiring are basically
correct.

Then the trained frontend checkpoint was tested with DDSP-SVS acoustic
inference using auxiliary output only:

```bash
python scripts/infer.py acoustic samples/123.ds \
  --exp 123 \
  --ddsp-svc ../DDSP-SVC \
  --model /root/DiffSinger/checkpoints/ddspmodel/model_1600.pt \
  --out outputs/step10k_aux.wav \
  --spk-id 1 \
  --infer-step 20 \
  --save-units
```

Result: the output was normal and reached usable DiffSinger-like quality.

This is the strongest current evidence that the main architecture works:

```text
DiffSinger encoder + auxiliary decoder -> ContentVec units -> DDSP-SVC backend
```

The two-stage debug tools are still useful for inspecting saved units, but
normal usage should go through the unified acoustic command:

```bash
python scripts/infer.py acoustic samples/123.ds \
  --exp 123 \
  --ddsp-svc ../DDSP-SVC \
  --model /root/DiffSinger/checkpoints/ddspmodel/model_1600.pt \
  --out outputs/step10k_aux.wav \
  --spk-id 1 \
  --infer-step 20
```

Add `--save-units` when the intermediate frontend payload should be kept for
analysis or later re-rendering.

## Important Finding

The poor result from the earlier `step10k` output was not caused by the frontend
failing to learn semantics. It was caused by the shallow diffusion / reflow stage
being unsuitable for the current unit target and inference settings.

For a matched validation item at 10k steps, `scripts/debug/analyze_units.py` reported:

```text
cosine(frame): mean=0.589678, median=0.609473
pred_minus_gt: std=0.307415
gt_units/delta: std=0.121780
pred_units/delta: std=0.314740
volume_corr=0.278748
```

Interpretation:

- The model has learned semantic content to a usable degree.
- The predicted units are not collapsed or over-smoothed.
- The frame-to-frame delta is much larger than ground truth, so the sampled unit
  trajectory is too jittery.
- The predicted volume is currently poor and may further hurt the backend.

When `--depth 0` is used, DiffSinger inference uses the auxiliary decoder output
and skips the shallow diffusion / reflow refinement. That output vocodes well
through DDSP-SVC.

## Recommended Direction

Treat DDSP-SVC as the acoustic backend that replaces the role originally played
by the mel refinement and vocoder side of DiffSinger.

The preferred architecture for now is:

```text
DiffSinger acoustic frontend:
  linguistic / note / duration / f0 conditions
  -> auxiliary decoder
  -> ContentVec units (+ optional volume)

DDSP-SVC backend:
  units + f0 + volume + speaker id
  -> mel / waveform
```

In this setup, DiffSinger shallow diffusion / reflow should not be part of the
default inference path. It currently acts like a second generator placed before
DDSP-SVC and can destroy the unit trajectory expected by the backend.

## Suggested Config Change

For the next round, train the unit frontend as an auxiliary-decoder-first model:

```yaml
unit_frontend_infer_aux: true
use_shallow_diffusion: true
T_start: 0.85
T_start_infer: 1.0
sampling_steps: 3
shallow_diffusion_args:
  train_aux_decoder: true
  train_diffusion: false

freezing_enabled: true
frozen_params:
  - model.diffusion
```

Normal unit frontend inference now defaults to auxiliary decoder output:

```bash
python scripts/infer.py acoustic samples/123.ds \
  --exp 123 \
  --ddsp-svc ../DDSP-SVC \
  --model checkpoints/ddspmodel/model_1600.pt \
  --out outputs/my_unit_frontend.wav \
  --save-units
```

The saved `.units.pt` file contains units, not real mel-spectrograms. It can be
re-rendered or inspected with tools under `scripts/debug/`.

If diffusion / reflow is tested again later, pass `--depth` explicitly. This
disables `unit_frontend_infer_aux` for that run. It should be a very light
refinement, for example:

```yaml
T_start: 0.85
T_start_infer: 0.9
sampling_steps: 3
```

But this should be treated as optional follow-up work, not the default path.

## Remaining Risks

- Volume prediction is weak:
  - predicted volume can be negative before clamping;
  - variance is much larger than ground truth;
  - correlation with ground truth is low.

- The current unit target uses broad global ranges (`spec_min: [-5]`,
  `spec_max: [5]`). Per-channel unit normalization may improve training later.

- ContentVec units are more sensitive to temporal jitter than mel. Loss curves
  can look good while DDSP-SVC output still sounds bad if the unit trajectory is
  unstable.

- The backend can use existing DDSP-SVC checkpoints only if their expected
  ContentVec extractor, hop sizes, f0 handling, volume handling, and speaker id
  convention match the generated frontend features.

## Practical Debug Checklist

1. Verify backend first with ground-truth binary units:

   ```bash
   python scripts/debug/vocode_binary_units.py ...
   ```

2. Verify the frontend on a known binary item:

   ```bash
   python scripts/debug/predict_binary_units.py ...
   python scripts/debug/analyze_units.py ...
   ```

3. For normal DS inference, prefer auxiliary-only output:

   ```bash
   python scripts/debug/infer_ddsp_svs.py ...
   ```

4. If output is intelligible with `--depth 0` but bad without it, the problem is
   the shallow diffusion / reflow stage, not the DDSP-SVC backend.

5. If ground-truth binary units sound bad, debug backend paths, ContentVec
   compatibility, f0, volume, speaker id, and HifiGAN assets before changing the
   frontend.
