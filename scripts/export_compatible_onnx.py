import argparse
import json
import os
import pathlib
import re
import shutil
import sys
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import yaml
import onnx

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/numba_cache')

from deployment.modules.toplevel import DiffSingerAcousticONNX
from modules.fastspeech.param_adaptor import VARIANCE_CHECKLIST
from scripts.export import find_exp, parse_spk_settings
from scripts.export_ddsp_backend_onnx import DotDict, Unit2Wav
from utils import load_ckpt, onnx_helper
from utils.hparams import hparams, set_hparams
from utils.phoneme_utils import load_phoneme_dictionary


class DDSPBackendMel(nn.Module):
    def __init__(
            self,
            backend: Unit2Wav,
            *,
            unit_dim: int,
            predict_volume: bool,
            volume_default: float,
            n_spk: Optional[int],
            frozen_backend_spk_mix: Optional[torch.Tensor],
            backend_t_start: float,
            backend_steps: int,
            backend_method: str,
            ddsp_noise_mode: str,
            reflow_noise_mode: str
    ):
        super().__init__()
        self.backend = backend
        self.unit_dim = unit_dim
        self.predict_volume = predict_volume
        self.volume_default = volume_default
        self.n_spk = n_spk
        self.backend_t_start = float(backend_t_start)
        self.backend_steps = int(backend_steps)
        self.backend_method = backend_method
        self.ddsp_noise_mode = ddsp_noise_mode
        self.reflow_noise_mode = reflow_noise_mode
        if frozen_backend_spk_mix is not None:
            self.register_buffer('frozen_backend_spk_mix', frozen_backend_spk_mix, persistent=False)

    def _make_ddsp_noise(self, units, ddsp_noise):
        if self.ddsp_noise_mode == 'input':
            return ddsp_noise
        shape = (units.size(0), units.size(1) * self.backend.block_size)
        noise = units.new_zeros(shape)
        if self.ddsp_noise_mode == 'random':
            noise = torch.randn_like(noise)
        return noise

    def _make_reflow_noise(self, x, reflow_noise):
        if self.reflow_noise_mode == 'input':
            return reflow_noise
        noise = torch.zeros_like(x)
        if self.reflow_noise_mode == 'random':
            noise = torch.randn_like(noise)
        return noise

    def _make_mel2ph(self, units):
        mel2ph = torch.arange(units.size(1), device=units.device, dtype=torch.long)
        return mel2ph.unsqueeze(0).repeat(units.size(0), 1)

    def _make_spk_mix(self, units, spk_mix):
        if self.n_spk is None or self.n_spk <= 1:
            return None
        if spk_mix is not None:
            return spk_mix
        if hasattr(self, 'frozen_backend_spk_mix'):
            mix = self.frozen_backend_spk_mix.to(dtype=units.dtype, device=units.device)
        else:
            mix = torch.full(
                (1, 1, self.n_spk),
                1.0 / float(self.n_spk),
                dtype=units.dtype,
                device=units.device
            )
        return mix.repeat(units.size(0), units.size(1), 1)

    def _split_prediction(self, pred):
        units = pred[..., :self.unit_dim]
        if self.predict_volume:
            volume = pred[..., self.unit_dim:self.unit_dim + 1].clamp_min(0.0)
        else:
            volume = torch.full(
                (pred.size(0), pred.size(1), 1),
                self.volume_default,
                dtype=pred.dtype,
                device=pred.device
            )
        return units, volume.squeeze(-1)

    def _run_backend_reflow(self, x, ddsp_mel, reflow_noise, steps=None):
        if self.backend_steps <= 0 or self.backend_t_start >= 1.0:
            return ddsp_mel.transpose(1, 2)
        if steps is None:
            steps = self.backend_steps
        x = self.backend_t_start * x + (1.0 - self.backend_t_start) * self._make_reflow_noise(x, reflow_noise)
        cond = ddsp_mel
        steps = torch.as_tensor(steps, dtype=torch.long, device=x.device)
        steps = torch.clamp(steps, min=1)
        dt = (1.0 - self.backend_t_start) / steps.to(dtype=x.dtype)
        if self.backend_method == 'euler':
            for t in torch.arange(steps, dtype=torch.long, device=x.device)[:, None].to(dtype=x.dtype) * dt + self.backend_t_start:
                x = x + self.backend.reflow_model.velocity_fn(x, 1000.0 * t, cond) * dt
        elif self.backend_method == 'rk4':
            for t in torch.arange(steps, dtype=torch.long, device=x.device)[:, None].to(dtype=x.dtype) * dt + self.backend_t_start:
                k1 = self.backend.reflow_model.velocity_fn(x, 1000.0 * t, cond)
                k2 = self.backend.reflow_model.velocity_fn(x + 0.5 * k1 * dt, 1000.0 * (t + 0.5 * dt), cond)
                k3 = self.backend.reflow_model.velocity_fn(x + 0.5 * k2 * dt, 1000.0 * (t + 0.5 * dt), cond)
                k4 = self.backend.reflow_model.velocity_fn(x + k3 * dt, 1000.0 * (t + dt), cond)
                x = x + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * dt / 6.0
        else:
            raise RuntimeError(f'Unsupported backend reflow method: {self.backend_method}')
        x = x.squeeze(1).transpose(1, 2)
        return self.backend.reflow_model.denorm_spec(x)

    def encode(self, pred, f0, ddsp_noise=None, backend_spk_mix=None):
        units, volume = self._split_prediction(pred)
        mel2ph = self._make_mel2ph(units)
        ddsp_noise = self._make_ddsp_noise(units, ddsp_noise)
        backend_spk_mix = self._make_spk_mix(units, backend_spk_mix)
        return self.backend.ddsp_model(units, mel2ph, f0, volume, backend_spk_mix, ddsp_noise)

    def forward(self, pred, f0, ddsp_noise=None, backend_spk_mix=None, reflow_noise=None, steps=None):
        x, ddsp_mel = self.encode(pred, f0, ddsp_noise=ddsp_noise, backend_spk_mix=backend_spk_mix)
        return self._run_backend_reflow(x, ddsp_mel, reflow_noise, steps=steps)


class DDSPUnifiedAcoustic(nn.Module):
    def __init__(self, frontend: DiffSingerAcousticONNX, backend_mel: DDSPBackendMel):
        super().__init__()
        self.frontend = frontend
        self.backend_mel = backend_mel

    def forward(
            self,
            tokens,
            durations,
            f0,
            variances: Dict[str, torch.Tensor],
            gender=None,
            velocity=None,
            spk_embed=None,
            languages=None,
            ddsp_noise=None,
            backend_spk_mix=None,
            reflow_noise=None,
            steps=None
    ):
        _, pred = self.frontend.forward_fs2_aux(
            tokens, durations, f0,
            variances=variances,
            gender=gender,
            velocity=velocity,
            spk_embed=spk_embed,
            languages=languages
        )
        return self.backend_mel(
            pred, f0,
            ddsp_noise=ddsp_noise,
            backend_spk_mix=backend_spk_mix,
            reflow_noise=reflow_noise,
            steps=steps
        )


class DDSPUnifiedEncoder(nn.Module):
    def __init__(self, frontend: DiffSingerAcousticONNX, backend_mel: DDSPBackendMel):
        super().__init__()
        self.frontend = frontend
        self.backend_mel = backend_mel

    def forward(
            self,
            tokens,
            durations,
            f0,
            variances: Dict[str, torch.Tensor],
            gender=None,
            velocity=None,
            spk_embed=None,
            languages=None,
            ddsp_noise=None,
            backend_spk_mix=None
    ):
        _, pred = self.frontend.forward_fs2_aux(
            tokens, durations, f0,
            variances=variances,
            gender=gender,
            velocity=velocity,
            spk_embed=spk_embed,
            languages=languages
        )
        return self.backend_mel.encode(pred, f0, ddsp_noise=ddsp_noise, backend_spk_mix=backend_spk_mix)


class BackendReflowONNX(nn.Module):
    def __init__(self, velocity_fn, *, t_start: float, method: str, spec_min=-12., spec_max=2.):
        super().__init__()
        self.velocity_fn = velocity_fn
        self.t_start = float(t_start)
        self.method = method
        self.spec_min = float(spec_min)
        self.spec_max = float(spec_max)

    def denorm_spec(self, x):
        return (x + 1.) / 2. * (self.spec_max - self.spec_min) + self.spec_min

    def forward(self, x, cond, steps: int = 10):
        device = x.device
        noise = torch.randn_like(x)
        x = self.t_start * x + (1. - self.t_start) * noise
        t_width = 1. - self.t_start
        dt = t_width / max(1, steps)
        if self.method == 'euler':
            for t in torch.arange(steps, dtype=torch.long, device=device)[:, None].float() * dt + self.t_start:
                x = x + self.velocity_fn(x, 1000. * t, cond) * dt
        elif self.method == 'rk4':
            for t in torch.arange(steps, dtype=torch.long, device=device)[:, None].float() * dt + self.t_start:
                k1 = self.velocity_fn(x, 1000. * t, cond)
                k2 = self.velocity_fn(x + 0.5 * k1 * dt, 1000. * (t + 0.5 * dt), cond)
                k3 = self.velocity_fn(x + 0.5 * k2 * dt, 1000. * (t + 0.5 * dt), cond)
                k4 = self.velocity_fn(x + k3 * dt, 1000. * (t + dt), cond)
                x = x + (k1 + 2. * k2 + 2. * k3 + k4) * dt / 6.
        else:
            raise RuntimeError(f'Unsupported backend reflow method: {self.method}')
        x = x[:, 0].transpose(1, 2)
        # x = x.squeeze(1).transpose(1, 2)
        return self.denorm_spec(x)


def parse_backend_spk_mix(value: Optional[str], n_spk: Optional[int]) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if n_spk is None or n_spk <= 1:
        raise ValueError('--backend-spk-mix can only be used with multi-speaker DDSP backend models.')
    pieces = [float(v) for v in re.split(r'[,|]', value) if v.strip()]
    if len(pieces) != n_spk:
        raise ValueError(f'Expected {n_spk} backend speaker mix values, got {len(pieces)}.')
    total = sum(pieces)
    if total <= 0:
        raise ValueError('Backend speaker mix sum must be positive.')
    mix = torch.tensor([v / total for v in pieces], dtype=torch.float32)
    return mix.view(1, 1, n_spk)


def build_lang_map() -> dict:
    lang_map_path = pathlib.Path(hparams['work_dir']) / 'lang_map.json'
    if not lang_map_path.exists():
        return {}
    with open(lang_map_path, encoding='utf8') as f:
        return json.load(f)


def export_dictionaries(path: pathlib.Path):
    dictionaries = hparams.get('dictionaries')
    if dictionaries is not None:
        for lang in dictionaries.keys():
            filename = f'dictionary-{lang}.txt'
            shutil.copy(pathlib.Path(hparams['work_dir']) / filename, path)
            print(f'| export dictionary => {path / filename}')
    else:
        filename = 'dictionary.txt'
        shutil.copy(pathlib.Path(hparams['work_dir']) / filename, path)
        print(f'| export dictionary => {path / filename}')


def export_attachments(
        path: pathlib.Path,
        model_filename: str,
        variance_embed_list: List[str],
        expose_steps: bool
):
    model_name = hparams['exp_name']
    phoneme_dictionary = load_phoneme_dictionary()
    phoneme_path = path / f'{model_name}.phonemes.json'
    phoneme_dictionary.dump(phoneme_path)
    print(f'| export phonemes => {phoneme_path}')
    lang_path = path / f'{model_name}.languages.json'
    with open(lang_path, 'w', encoding='utf8') as f:
        json.dump(build_lang_map(), f, ensure_ascii=False, indent=2)
    print(f'| export languages => {lang_path}')
    export_dictionaries(path)

    dsconfig = {
        'phonemes': f'{model_name}.phonemes.json',
        'languages': f'{model_name}.languages.json',
        'use_lang_id': hparams.get('use_lang_id', False) and len(phoneme_dictionary.cross_lingual_phonemes) > 0,
        'acoustic': model_filename,
        'hidden_size': hparams['hidden_size'],
        'vocoder': 'pc_nsf_hifigan_44.1k_hop512_128bin_2025.02',
        'use_key_shift_embed': hparams.get('use_key_shift_embed', False),
        'use_speed_embed': hparams.get('use_speed_embed', False),
        'use_continuous_acceleration': expose_steps,
        'use_variable_depth': False,
        'max_depth': 0,
        'sample_rate': hparams['audio_sample_rate'],
        'hop_size': hparams['hop_size'],
        'win_size': hparams['win_size'],
        'fft_size': hparams['fft_size'],
        'num_mel_bins': 128,
        'mel_fmin': hparams['fmin'],
        'mel_fmax': hparams['fmax'] if hparams['fmax'] is not None else hparams['audio_sample_rate'] / 2,
        'mel_base': 'e',
        'mel_scale': 'slaney',
    }
    for variance in VARIANCE_CHECKLIST:
        dsconfig[f'use_{variance}_embed'] = variance in variance_embed_list
    config_path = path / 'dsconfig.yaml'
    with open(config_path, 'w', encoding='utf8') as f:
        yaml.safe_dump(dsconfig, f, sort_keys=False)
    print(f'| export configs => {config_path} **PLEASE EDIT BEFORE USE**')


def build_frontend(exp: str, ckpt_steps: Optional[int], freeze_spk: Optional[str], device):
    phoneme_dictionary = load_phoneme_dictionary()
    model = DiffSingerAcousticONNX(
        vocab_size=len(phoneme_dictionary),
        out_dims=hparams['unit_dim'] + (1 if hparams.get('predict_volume', False) else 0),
        cross_lingual_token_idx=sorted({
            phoneme_dictionary.encode_one(p)
            for p in phoneme_dictionary.cross_lingual_phonemes
        })
    ).eval().to(device)
    if not hparams.get('use_shallow_diffusion', False):
        raise RuntimeError('Unified DDSP-SVS ONNX export requires use_shallow_diffusion=true for the aux unit frontend.')
    if not hparams.get('unit_frontend_infer_aux', True):
        raise RuntimeError('Unified DDSP-SVS ONNX export expects unit_frontend_infer_aux=true.')
    load_ckpt(model, hparams['work_dir'], ckpt_steps=ckpt_steps, prefix_in_ckpt='model', strict=True, device=device)
    if hparams.get('use_spk_id', False) and freeze_spk is not None:
        _, freeze_spk_mix = parse_spk_settings(None, freeze_spk)
        spk_map_path = pathlib.Path(hparams['work_dir']) / 'spk_map.json'
        with open(spk_map_path, encoding='utf8') as f:
            spk_map = json.load(f)
        spk_ids = []
        spk_values = []
        for name, value in freeze_spk_mix[1].items():
            spk_ids.append(spk_map[name])
            spk_values.append(value)
        spk_id = torch.LongTensor(spk_ids).to(device)[None]
        spk_value = torch.FloatTensor(spk_values).to(device)[None]
        spk_value = spk_value / spk_value.sum()
        frozen_spk = torch.sum(model.fs2.spk_embed(spk_id) * spk_value.unsqueeze(2), dim=1, keepdim=True)
        model.fs2.register_buffer('frozen_spk_embed', frozen_spk)
    return model


def build_backend(model_path: pathlib.Path, backend_spk_mix: Optional[str], device):
    config_file = model_path.parent / 'config.yaml'
    with open(config_file, encoding='utf8') as f:
        args = DotDict(yaml.safe_load(f))
    backend = Unit2Wav(
        args.data.sampling_rate,
        args.data.block_size,
        args.model.win_length,
        args.data.encoder_out_channels,
        args.model.n_spk,
        args.model.use_norm,
        args.model.use_attention,
        args.model.use_pitch_aug,
        128,
        args.model.n_aux_layers,
        args.model.n_aux_chans,
        args.model.n_layers,
        args.model.n_chans,
        args.data.f0_min if args.data.f0_min is not None else 65
    ).eval().to(device)
    ckpt = torch.load(model_path, map_location=device)
    backend.load_state_dict(ckpt['model'], strict=True)
    n_spk = args.model.n_spk
    if n_spk is not None and n_spk > 1:
        backend.ddsp_model.unit2ctrl.export_chara_mix(n_spk)
    frozen_mix = parse_backend_spk_mix(backend_spk_mix, n_spk)
    if frozen_mix is not None:
        frozen_mix = frozen_mix.to(device)
    return backend, args, frozen_mix


def export_unified_onnx(cmd):
    exp = find_exp(cmd.exp)
    sys.argv = [sys.argv[0], '--exp_name', exp, '--infer']
    set_hparams()
    if hparams.get('task_cls') != 'training.unit_acoustic_task.UnitAcousticTask':
        raise RuntimeError('This exporter only supports UnitAcousticTask checkpoints.')

    device = torch.device(cmd.device)
    frontend = build_frontend(exp, cmd.ckpt, cmd.freeze_spk, device)
    backend, backend_args, frozen_backend_spk_mix = build_backend(cmd.backend_model, cmd.backend_spk_mix, device)
    backend_steps = cmd.backend_steps
    backend_t_start = cmd.backend_t_start
    backend_method = cmd.backend_method
    if backend_steps is None:
        backend_steps = int(getattr(backend_args.infer, 'infer_step', 0))
    if backend_t_start is None:
        backend_t_start = float(getattr(backend_args.model, 't_start', 1.0) or 0.0)
    if backend_method == 'auto':
        backend_method = getattr(backend_args.infer, 'method', 'euler')
    expose_steps = backend_steps > 0 and backend_t_start < 1.0

    unified = DDSPUnifiedAcoustic(
        frontend,
        DDSPBackendMel(
            backend,
            unit_dim=hparams.get('unit_dim', 768),
            predict_volume=hparams.get('predict_volume', False),
            volume_default=cmd.volume_default,
            n_spk=backend_args.model.n_spk,
            frozen_backend_spk_mix=frozen_backend_spk_mix,
            backend_t_start=backend_t_start,
            backend_steps=backend_steps,
            backend_method=backend_method,
            ddsp_noise_mode=cmd.ddsp_noise,
            reflow_noise_mode=cmd.reflow_noise
        )
    ).eval().to(device)

    n_frames = cmd.frames
    tokens = torch.LongTensor([[1]]).to(device)
    durations = torch.LongTensor([[n_frames]]).to(device)
    f0 = torch.FloatTensor([[440.0] * n_frames]).to(device)
    variances = {
        v_name: torch.zeros(1, n_frames, dtype=torch.float32, device=device)
        for v_name in unified.frontend.fs2.variance_embed_list
    }
    kwargs: Dict[str, torch.Tensor] = {}
    input_names: List[str] = ['tokens', 'durations', 'f0'] + unified.frontend.fs2.variance_embed_list
    dynamic_axes = {
        'tokens': {1: 'n_tokens'},
        'durations': {1: 'n_tokens'},
        'f0': {1: 'n_frames'},
        'mel': {1: 'n_frames'},
        **{v_name: {1: 'n_frames'} for v_name in unified.frontend.fs2.variance_embed_list}
    }
    if hparams.get('use_key_shift_embed', False):
        kwargs['gender'] = torch.zeros(1, n_frames, dtype=torch.float32, device=device)
        input_names.append('gender')
        dynamic_axes['gender'] = {1: 'n_frames'}
    if hparams.get('use_speed_embed', False):
        kwargs['velocity'] = torch.ones(1, n_frames, dtype=torch.float32, device=device)
        input_names.append('velocity')
        dynamic_axes['velocity'] = {1: 'n_frames'}
    if hparams.get('use_spk_id', False) and not hasattr(frontend.fs2, 'frozen_spk_embed'):
        kwargs['spk_embed'] = torch.zeros(1, n_frames, hparams['hidden_size'], dtype=torch.float32, device=device)
        input_names.append('spk_embed')
        dynamic_axes['spk_embed'] = {1: 'n_frames'}
    use_lang_id = hparams.get('use_lang_id', False) and getattr(frontend.fs2, 'use_lang_id', False)
    if use_lang_id:
        kwargs['languages'] = torch.zeros_like(tokens)
        input_names.append('languages')
        dynamic_axes['languages'] = {1: 'n_tokens'}
    if cmd.ddsp_noise == 'input':
        kwargs['ddsp_noise'] = torch.randn(1, n_frames * backend.block_size, dtype=torch.float32, device=device)
        input_names.append('ddsp_noise')
        dynamic_axes['ddsp_noise'] = {1: 'audio_length'}
    if backend_args.model.n_spk is not None and backend_args.model.n_spk > 1 and frozen_backend_spk_mix is None:
        kwargs['backend_spk_mix'] = torch.full(
            (1, n_frames, backend_args.model.n_spk),
            1.0 / float(backend_args.model.n_spk),
            dtype=torch.float32,
            device=device
        )
        input_names.append('backend_spk_mix')
        dynamic_axes['backend_spk_mix'] = {1: 'n_frames'}
    if cmd.reflow_noise == 'input' and backend_steps > 0 and backend_t_start < 1.0:
        kwargs['reflow_noise'] = torch.randn(1, 1, 128, n_frames, dtype=torch.float32, device=device)
        input_names.append('reflow_noise')
        dynamic_axes['reflow_noise'] = {3: 'n_frames'}
    if expose_steps:
        kwargs['steps'] = torch.tensor(backend_steps, dtype=torch.long, device=device)
        input_names.append('steps')

    model_name = hparams['exp_name']
    if cmd.freeze_spk is not None:
        _, freeze_spk_mix = parse_spk_settings(None, cmd.freeze_spk)
        model_name += '.' + freeze_spk_mix[0]
    if cmd.output is None:
        output_dir = cmd.out if cmd.out is not None else ROOT_DIR / 'artifacts' / exp
        output_path = output_dir / f'{model_name}.onnx'
    else:
        output_path = cmd.output
        output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
#     torch.onnx.export(
#         unified,
#         (tokens, durations, f0, variances, kwargs),
#         output_path,
#         input_names=input_names,
#         output_names=['mel'],
#         dynamic_axes=dynamic_axes,
#         opset_version=18,
#         dynamo=False,
#         do_constant_folding=False,
#         verbose=False
#     )
    if expose_steps and cmd.dynamic_steps:
        encoder_path = output_path.with_suffix('.encoder.cache.onnx')
        reflow_path = output_path.with_suffix('.reflow.cache.onnx')
        encoder_kwargs = {k: v for k, v in kwargs.items() if k not in {'steps', 'reflow_noise'}}
        encoder_input_names = [name for name in input_names if name not in {'steps', 'reflow_noise'}]
        encoder_dynamic_axes = {
            k: v for k, v in dynamic_axes.items()
            if k not in {'steps', 'reflow_noise', 'mel'}
        }
        encoder_dynamic_axes.update({
            'x': {3: 'n_frames'},
            'cond': {2: 'n_frames'},
        })
        torch.onnx.export(
            DDSPUnifiedEncoder(frontend, unified.backend_mel).eval().to(device),
            (tokens, durations, f0, variances, encoder_kwargs),
            encoder_path,
            input_names=encoder_input_names,
            output_names=['x', 'cond'],
            dynamic_axes=encoder_dynamic_axes,
            opset_version=18,
            dynamo=False,
            do_constant_folding=False,
            verbose=False
        )

        dummy_x = torch.randn(1, 1, 128, n_frames, dtype=torch.float32, device=device)
        dummy_cond = torch.randn(1, 128, n_frames, dtype=torch.float32, device=device)
        dummy_time = torch.tensor([0], dtype=torch.int64, device=device)
        dummy_steps = int(backend_steps)
        traced_velocity = torch.jit.trace(backend.reflow_model.velocity_fn, (dummy_x, dummy_time, dummy_cond))
        reflow = torch.jit.script(
            BackendReflowONNX(traced_velocity, t_start=backend_t_start, method=backend_method).eval().to(device),
            example_inputs=[(dummy_x, dummy_cond, dummy_steps)]
        )
        torch.onnx.export(
            reflow,
            (dummy_x, dummy_cond, dummy_steps),
            reflow_path,
            input_names=['x', 'cond', 'steps'],
            output_names=['mel'],
            dynamic_axes={
                'x': {3: 'n_frames'},
                'cond': {2: 'n_frames'},
                'mel': {1: 'n_frames'},
            },
            opset_version=18,
            dynamo=False,
            do_constant_folding=False,
            verbose=False
        )
        encoder_onnx = onnx.load(encoder_path)
#         reflow_onnx = onnx.load(reflow_path)
        reflow_onnx = onnx.compose.add_prefix(
            onnx.load(reflow_path),
            'backend_reflow.',
            rename_inputs=False,
            rename_outputs=False,
            rename_initializers=False,
            inplace=False
        )
        model_onnx = onnx.compose.merge_models(
            encoder_onnx,
            reflow_onnx,
            io_map=[('x', 'x'), ('cond', 'cond')],
            prefix1='', prefix2='',
            producer_name=encoder_onnx.producer_name,
            producer_version=encoder_onnx.producer_version,
            domain=encoder_onnx.domain,
            model_version=encoder_onnx.model_version
        )
        onnx.save(model_onnx, output_path)
        encoder_path.unlink(missing_ok=True)
        reflow_path.unlink(missing_ok=True)
    else:
        torch.onnx.export(
            unified,
            (tokens, durations, f0, variances, kwargs),
            output_path,
            input_names=input_names,
            output_names=['mel'],
            dynamic_axes=dynamic_axes,
            opset_version=18,
            dynamo=False,
            do_constant_folding=False,
            verbose=False
        )
    model_onnx = onnx.load(output_path)
    onnx_helper.model_reorder_io_list(
        model_onnx, 'input',
        target_name='languages', insert_after_name='tokens'
    )
    onnx_helper.model_override_io_shapes(model_onnx, output_shapes={
        'mel': (1, 'n_frames', 128)
    })
    onnx.save(model_onnx, output_path)
    print(f'| export unified DDSP-SVS mel ONNX => {output_path}')
    if not cmd.no_attachments:
        export_attachments(output_dir, output_path.name, unified.frontend.fs2.variance_embed_list, expose_steps)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Export unit frontend + DDSP-SVC backend as one acoustic-like ONNX model.'
    )
    parser.add_argument('--exp', required=True, help='DiffSinger unit frontend experiment name or prefix.')
    parser.add_argument('--ckpt', type=int, default=None, help='Frontend checkpoint steps.')
    parser.add_argument('--backend-model', type=pathlib.Path, required=True, help='DDSP-SVC backend checkpoint path.')
    parser.add_argument('--out', type=pathlib.Path, default=None,
                        help='Output artifact directory. Defaults to artifacts/<exp>, like scripts/export.py acoustic.')
    parser.add_argument('--output', type=pathlib.Path, default=None,
                        help='Exact output ONNX file path. Overrides --out.')
    parser.add_argument('--no-attachments', action='store_true',
                        help='Only write the ONNX file and skip dsconfig/dictionary attachments.')
    parser.add_argument('--device', default='cpu', help='Export device, usually cpu or cuda.')
    parser.add_argument('--frames', type=int, default=25, help='Dummy frame count used during tracing.')
    parser.add_argument('--freeze-spk', default=None, help='Freeze DiffSinger frontend speaker mix, same syntax as scripts/export.py.')
    parser.add_argument('--backend-spk-mix', default=None, help='Freeze DDSP backend speaker mix, comma-separated weights.')
    parser.add_argument('--volume-default', type=float, default=0.1, help='Volume used when the frontend does not predict volume.')
    parser.add_argument('--backend-steps', type=int, default=None, help='Backend reflow steps; default reads DDSP config.')
    parser.add_argument('--backend-t-start', type=float, default=None, help='Backend reflow t_start; default reads DDSP config.')
    parser.add_argument('--backend-method', choices=['auto', 'euler', 'rk4'], default='auto', help='Backend reflow sampler.')
    parser.add_argument('--ddsp-noise', choices=['random', 'zero', 'input'], default='random',
                        help='Noise source for the DDSP subtractive noise branch.')
    parser.add_argument('--reflow-noise', choices=['random', 'zero', 'input'], default='random',
                        help='Initial noise source when backend reflow is enabled.')
    parser.add_argument('--static-steps', dest='dynamic_steps', action='store_false',
                        help='Trace backend reflow as a fixed loop instead of exporting a dynamic steps input.')
    parser.set_defaults(dynamic_steps=True)
    return parser.parse_args(args=args)


if __name__ == '__main__':
    export_unified_onnx(parse_args())
