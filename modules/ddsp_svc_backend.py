import pathlib
from ast import literal_eval

import numpy as np
import torch

from utils.infer_utils import cross_fade, save_wav


def load_backend(ddsp_svc_path, model_path, device, vocoder_ckpt=None):
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    requested_asset_root = pathlib.Path(ddsp_svc_path).resolve()
    asset_roots = []
    if requested_asset_root.exists():
        asset_roots.append(requested_asset_root)
    asset_roots.append(repo_root)
    model_path = pathlib.Path(model_path).resolve()
    from backend.ddsp.reflow.vocoder import load_model_vocoder

    model, vocoder, args = load_model_vocoder(
        str(model_path),
        device=device,
        asset_root=asset_roots,
        vocoder_ckpt=vocoder_ckpt
    )
    return model, vocoder, args


def split_prediction(pred, unit_dim=768, volume_default=0.1):
    pred = pred.float()
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
    if pred.shape[-1] < unit_dim:
        raise ValueError(f'Expected at least {unit_dim} channels, got {pred.shape[-1]}.')
    units = pred[..., :unit_dim]
    if pred.shape[-1] > unit_dim:
        volume = pred[..., unit_dim:unit_dim + 1].clamp_min(0.0)
    else:
        volume = torch.full((*pred.shape[:2], 1), volume_default, dtype=pred.dtype)
    return units, volume


@torch.no_grad()
def synthesize_segment(model, vocoder, args, entry, options, device):
    pred = entry['mel'].to(device)
    units, volume = split_prediction(pred, options.unit_dim, options.volume_default)
    f0 = entry['f0'].float().to(device)
    if f0.dim() == 2:
        f0 = f0.unsqueeze(-1)
    f0 = f0 * (2 ** (options.backend_key / 12))

    length = min(units.shape[1], volume.shape[1], f0.shape[1])
    units = units[:, :length]
    volume = volume[:, :length]
    f0 = f0[:, :length]

    spk_mix_dict = literal_eval(options.spk_mix_dict)
    spk_id = torch.LongTensor(np.array([[options.spk_id]])).to(device)
    aug_shift = torch.FloatTensor([[options.formant_shift_key]]).to(device)

    infer_step = args.infer.infer_step if options.infer_step < 0 else options.infer_step
    method = args.infer.method if options.method == 'auto' else options.method
    if options.t_start is None:
        t_start = args.model.t_start if args.model.t_start is not None else 0.0
    else:
        t_start = options.t_start

    wav = model(
        units, f0, volume,
        spk_id=spk_id,
        spk_mix_dict=spk_mix_dict,
        aug_shift=aug_shift,
        vocoder=vocoder,
        infer=True,
        return_wav=True,
        infer_step=infer_step,
        method=method,
        t_start=t_start,
        use_tqdm=True
    )
    return wav.squeeze().detach().cpu().numpy()


def render_payload(model, vocoder, args, payload, options, device):
    output_sr = args.data.sampling_rate
    result = np.zeros(0, dtype=np.float32)
    current_length = 0
    for entry in payload:
        wav = synthesize_segment(model, vocoder, args, entry, options, device)
        silent_length = round(entry.get('offset', 0.0) * output_sr) - current_length
        if silent_length >= 0:
            result = np.append(result, np.zeros(silent_length, dtype=np.float32))
            result = np.append(result, wav)
        else:
            result = cross_fade(result, wav, current_length + silent_length)
        current_length = current_length + silent_length + wav.shape[0]
    return result, output_sr


def save_rendered_payload(model, vocoder, args, payload, options, device, out_path):
    result, output_sr = render_payload(model, vocoder, args, payload, options, device)
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_wav(result, out_path, output_sr)
    return out_path
