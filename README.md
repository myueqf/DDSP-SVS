# DDSP-SVS

DDSP-SVS is an experimental singing voice synthesis frontend based on
[DiffSinger](https://github.com/openvpi/DiffSinger). Instead of predicting mel
spectrograms directly, it predicts ContentVec units and renders audio with a
DDSP-SVC backend.

The current recommended pipeline is:

```text
lyrics / phonemes / notes / durations / f0
  -> DiffSinger unit frontend
  -> ContentVec units (+ volume)
  -> DDSP-SVC backend
  -> waveform
```

The variance model and most of the DiffSinger data pipeline are kept. The unit
frontend is intended to replace the acoustic feature extractor part, while the
DDSP-SVC model stays as the backend renderer.

## Status

This project is experimental. The currently tested path is the auxiliary-decoder
unit frontend with DDSP-SVC as a frozen backend. Shallow diffusion / reflow on
units is not the default inference path.

## Setup

Install dependencies:

```bash
# Install PyTorch and torchaudio for your CUDA version first.
# See https://pytorch.org/get-started/locally/
python -m pip install pip==24.0
pip install -r requirements.txt
```

`fairseq==0.12.2` is used for legacy ContentVec checkpoints and is known to
need `pip==24.0` during installation.

The backend uses `gin-config` (`import gin`). Do not install the unrelated
`gin` package as a replacement.

Prepare a DDSP-SVC repository beside this project, or pass its path explicitly:

```text
../DDSP-SVC
```

The DDSP-SVC backend should contain its normal model config, reflow checkpoint,
ContentVec checkpoint, and NsfHifiGAN assets.

## Binarization

Edit `configs/unit_acoustic.yaml` for your dataset paths, dictionary, and
ContentVec checkpoint path, then run:

```bash
python scripts/binarize.py --config configs/unit_acoustic.yaml
```

## Training

Train the unit frontend:

```bash
python scripts/train.py \
  --config configs/unit_acoustic.yaml \
  --exp_name my_unit_frontend \
  --reset
```

Checkpoints are saved in:

```text
checkpoints/my_unit_frontend/
```

## Inference

Run end-to-end inference with a DDSP-SVC backend:

```bash
python scripts/infer_ddsp_svs.py samples/example.ds \
  --exp my_unit_frontend \
  --ddsp-svc ../DDSP-SVC \
  --model checkpoints/ddspmodel/model_1600.pt \
  --out outputs/example.wav \
  --spk-id 1 \
  --infer-step 50
```

To keep the intermediate unit payload:

```bash
python scripts/infer_ddsp_svs.py samples/example.ds \
  --exp my_unit_frontend \
  --ddsp-svc ../DDSP-SVC \
  --model checkpoints/ddspmodel/model_1600.pt \
  --out outputs/example.wav \
  --save-units
```

You can also run the two stages separately:

```bash
python scripts/infer.py acoustic samples/example.ds \
  --exp my_unit_frontend \
  --mel \
  --out outputs \
  --title example

python scripts/vocode_units.py outputs/example.units.pt \
  --ddsp-svc ../DDSP-SVC \
  --model checkpoints/ddspmodel/model_1600.pt \
  --out outputs/example.wav \
  --spk-id 1 \
  --infer-step 50
```

## Diagnostics

Render ground-truth binary units through the backend:

```bash
python scripts/vocode_binary_units.py \
  --binary-data-dir data/unit_frontend/binary \
  --prefix valid \
  --name ITEM_NAME \
  --ddsp-svc ../DDSP-SVC \
  --model checkpoints/ddspmodel/model_1600.pt \
  --out outputs/gt_item.wav
```

Compare predicted units against binary ground truth:

```bash
python scripts/predict_binary_units.py \
  --exp my_unit_frontend \
  --binary-data-dir data/unit_frontend/binary \
  --prefix valid \
  --name ITEM_NAME \
  --out outputs/pred_item.units.pt

python scripts/analyze_units.py \
  --pred outputs/pred_item.units.pt \
  --binary-data-dir data/unit_frontend/binary \
  --prefix valid \
  --name ITEM_NAME
```

## Notes

More implementation notes are in
[`docs/UnitFrontendDDSP.md`](docs/UnitFrontendDDSP.md).

This repository is a fork of DiffSinger and keeps the original Apache 2.0
license. Please also follow the license and model usage terms of the DDSP-SVC
backend and any pretrained checkpoints you use.
