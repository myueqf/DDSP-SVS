import argparse
import json
import pathlib
import sys

import numpy as np
import torch
import tqdm

root_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from modules.ddsp_svc_backend import load_backend, save_rendered_payload


def find_exp(exp):
    ckpt_dir = root_dir / 'checkpoints' / exp
    if ckpt_dir.exists():
        print(f'| found ckpt by name: {exp}')
        return exp
    for subdir in (root_dir / 'checkpoints').iterdir():
        if subdir.is_dir() and subdir.name.startswith(exp):
            print(f'| match ckpt by prefix: {subdir.name}')
            return subdir.name
    raise ValueError(
        f"There are no matching exp starting with {exp!r} in 'checkpoints' folder."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run DiffSinger unit frontend and DDSP-SVC backend in one command.'
    )
    parser.add_argument('proj', type=pathlib.Path, help='Input DS file')
    parser.add_argument('--exp', type=str, required=True, help='DiffSinger unit frontend experiment')
    parser.add_argument('--config', type=str, default='', help='Config file used by the frontend experiment')
    parser.add_argument('--ckpt', type=int, default=None, help='Frontend checkpoint steps')
    parser.add_argument('--ddsp-svc', type=pathlib.Path, default=pathlib.Path('../DDSP-SVC'),
                        help='Path to DDSP-SVC repository')
    parser.add_argument('--model', type=pathlib.Path, required=True,
                        help='Path to DDSP-SVC reflow checkpoint')
    parser.add_argument('--vocoder-ckpt', type=pathlib.Path, default=None,
                        help='Override DDSP-SVC vocoder checkpoint path')
    parser.add_argument('--out', type=pathlib.Path, default=None, help='Output folder or wav path')
    parser.add_argument('--title', type=str, default=None, help='Output title')
    parser.add_argument('--num', type=int, default=1, help='Number of frontend runs')
    parser.add_argument('--spk', type=str, default=None, help='DiffSinger frontend speaker mix')
    parser.add_argument('--lang', type=str, default=None, help='Default frontend language')
    parser.add_argument('--key', type=int, default=0, help='Transpose DS/frontend pitch in semitones')
    parser.add_argument('--gender', type=float, default=None, help='Frontend formant/gender control')
    parser.add_argument('--seed', type=int, default=-1, help='Frontend random seed')
    parser.add_argument('--depth', type=float, default=None,
                        help='Frontend shallow diffusion depth; unset uses aux-only unit output')
    parser.add_argument('--steps', type=int, default=None, help='Frontend diffusion sampling steps')
    parser.add_argument('--save-units', action='store_true', help='Save intermediate unit payload')
    parser.add_argument('--units-out', type=pathlib.Path, default=None,
                        help='Intermediate unit payload path; implies --save-units')

    parser.add_argument('--unit-dim', type=int, default=768, help='Number of ContentVec unit channels')
    parser.add_argument('--spk-id', type=int, default=1, help='DDSP-SVC speaker id, 1-based')
    parser.add_argument('--spk-mix-dict', type=str, default='None',
                        help='DDSP-SVC speaker mix dict, e.g. "{1:0.5,2:0.5}"')
    parser.add_argument('--backend-key', type=float, default=0.0,
                        help='DDSP-SVC backend pitch shift in semitones')
    parser.add_argument('--formant-shift-key', type=float, default=0.0,
                        help='DDSP-SVC aug_shift/formant key')
    parser.add_argument('--infer-step', type=int, default=-1,
                        help='DDSP-SVC backend reflow steps; -1 uses backend config')
    parser.add_argument('--method', type=str, default='auto', choices=['auto', 'euler', 'rk4'],
                        help='DDSP-SVC backend sampling method')
    parser.add_argument('--t-start', type=float, default=None,
                        help='DDSP-SVC backend reflow t_start; default uses backend config or 0')
    parser.add_argument('--volume-default', type=float, default=0.1,
                        help='Fallback volume if frontend tensor has no volume channel')
    parser.add_argument('--device', type=str, default=None, help='cpu/cuda; default auto')
    return parser.parse_args()


def load_params(proj):
    with open(proj, 'r', encoding='utf-8') as f:
        params = json.load(f)
    if not isinstance(params, list):
        params = [params]
    if len(params) == 0:
        raise ValueError('The input file is empty.')
    return params


def setup_hparams(cmd):
    exp = find_exp(cmd.exp)
    sys.argv = [sys.argv[0], '--exp_name', exp, '--infer']
    if cmd.config:
        sys.argv.extend(['--config', cmd.config])
    from utils.hparams import set_hparams, hparams
    set_hparams()

    is_unit_frontend = hparams.get('task_cls') == 'training.unit_acoustic_task.UnitAcousticTask'
    if not is_unit_frontend:
        raise ValueError('infer_ddsp_svs.py requires a UnitAcousticTask checkpoint.')
    if 'unit_frontend_infer_aux' not in hparams:
        hparams['unit_frontend_infer_aux'] = True

    if 'diff_speedup' not in hparams and 'pndm_speedup' in hparams:
        hparams['diff_speedup'] = hparams['pndm_speedup']
    if 'T_start' not in hparams:
        hparams['T_start'] = 1 - hparams['K_step'] / hparams['timesteps']
    if 'T_start_infer' not in hparams:
        hparams['T_start_infer'] = 1 - hparams['K_step_infer'] / hparams['timesteps']
    if 'sampling_steps' not in hparams:
        if hparams['use_shallow_diffusion']:
            hparams['sampling_steps'] = hparams['K_step_infer'] // hparams['diff_speedup']
        else:
            hparams['sampling_steps'] = hparams['timesteps'] // hparams['diff_speedup']
    if 'time_scale_factor' not in hparams:
        hparams['time_scale_factor'] = hparams['timesteps']

    if cmd.depth is not None:
        hparams['unit_frontend_infer_aux'] = False
        if cmd.depth > 1 - hparams['T_start']:
            raise ValueError(f"Depth should not be larger than 1 - T_start ({1 - hparams['T_start']}).")
        hparams['K_step_infer'] = round(hparams['timesteps'] * cmd.depth)
        hparams['T_start_infer'] = 1 - cmd.depth
        print('| unit frontend: use shallow diffusion because --depth was specified')
    elif hparams.get('unit_frontend_infer_aux', False):
        hparams['T_start_infer'] = 1.0
        hparams['K_step_infer'] = 0
        print('| unit frontend: use auxiliary decoder output')

    if cmd.steps is not None:
        if hparams['use_shallow_diffusion']:
            step_size = (1 - hparams['T_start_infer']) / cmd.steps
            if 'K_step_infer' in hparams:
                hparams['diff_speedup'] = round(step_size * hparams['K_step_infer'])
        else:
            if 'timesteps' in hparams:
                hparams['diff_speedup'] = round(hparams['timesteps'] / cmd.steps)
        hparams['sampling_steps'] = cmd.steps

    return hparams


@torch.no_grad()
def run_frontend(cmd, params, hparams, device):
    from inference.ds_acoustic import DiffSingerAcousticInfer
    from utils.infer_utils import parse_commandline_spk_mix, trans_key

    if cmd.key != 0:
        params = trans_key(params, cmd.key)
        print(f'| key transition: {cmd.key:+d}')

    spk_mix = parse_commandline_spk_mix(cmd.spk) if hparams['use_spk_id'] and cmd.spk is not None else None
    for param in params:
        if cmd.gender is not None and hparams['use_key_shift_embed']:
            param['gender'] = cmd.gender
        if spk_mix is not None:
            param['spk_mix'] = spk_mix
        if cmd.lang is not None:
            param['lang'] = cmd.lang

    infer = DiffSingerAcousticInfer(device=device, load_vocoder=False, ckpt_steps=cmd.ckpt)
    print(f'| Frontend model: {type(infer.model)}')
    batches = [infer.preprocess_input(param, idx=i) for i, param in enumerate(params)]

    payload = []
    for param, batch in tqdm.tqdm(zip(params, batches), desc='frontend segments', total=len(params)):
        if 'seed' in param:
            torch.manual_seed(param['seed'] & 0xffff_ffff)
            torch.cuda.manual_seed_all(param['seed'] & 0xffff_ffff)
        elif cmd.seed >= 0:
            torch.manual_seed(cmd.seed & 0xffff_ffff)
            torch.cuda.manual_seed_all(cmd.seed & 0xffff_ffff)

        units = infer.forward_model(batch)
        payload.append({
            'offset': param.get('offset', 0.0),
            'mel': units.cpu(),
            'f0': batch['f0'].cpu()
        })
    return payload


def resolve_output_paths(cmd):
    title = cmd.title or cmd.proj.stem
    if cmd.out is None:
        wav_path = cmd.proj.with_name(title + '.wav')
    elif cmd.out.suffix.lower() == '.wav':
        wav_path = cmd.out
    else:
        wav_path = cmd.out / f'{title}.wav'

    if cmd.units_out is not None:
        units_path = cmd.units_out
    else:
        units_path = wav_path.with_suffix('.units.pt')
    return wav_path, units_path


def main():
    cmd = parse_args()
    device = cmd.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    params = load_params(cmd.proj)
    hparams = setup_hparams(cmd)
    wav_path, units_path = resolve_output_paths(cmd)

    backend_model, backend_vocoder, backend_args = load_backend(
        cmd.ddsp_svc, cmd.model, device=device, vocoder_ckpt=cmd.vocoder_ckpt
    )

    for i in range(cmd.num):
        if cmd.num > 1:
            run_wav_path = wav_path.with_name(f'{wav_path.stem}-{str(i).zfill(3)}{wav_path.suffix}')
            run_units_path = units_path.with_name(f'{units_path.stem}-{str(i).zfill(3)}{units_path.suffix}')
        else:
            run_wav_path = wav_path
            run_units_path = units_path

        payload = run_frontend(cmd, params, hparams, device)
        if cmd.save_units or cmd.units_out is not None:
            run_units_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, run_units_path)
            print(f'| save units: {run_units_path}')
        save_rendered_payload(
            backend_model, backend_vocoder, backend_args, payload, cmd, device, run_wav_path
        )
        print(f'| save audio: {run_wav_path}')


if __name__ == '__main__':
    main()
