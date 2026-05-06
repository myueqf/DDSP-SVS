import json
import os
import pathlib
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import click
from typing import Tuple

root_dir = Path(__file__).resolve().parent.parent
os.environ['PYTHONPATH'] = str(root_dir)
sys.path.insert(0, str(root_dir))


def find_exp(exp):
    if not (root_dir / 'checkpoints' / exp).exists():
        for subdir in (root_dir / 'checkpoints').iterdir():
            if not subdir.is_dir():
                continue
            if subdir.name.startswith(exp):
                print(f'| match ckpt by prefix: {subdir.name}')
                exp = subdir.name
                break
        else:
            raise click.BadParameter(
                f'There are no matching exp starting with \'{exp}\' in \'checkpoints\' folder. '
                'Please specify \'--exp\' as the folder name or prefix.'
            )
    else:
        print(f'| found ckpt by name: {exp}')
    return exp


@click.group()
def main():
    pass


@main.command(help='Run DDSP-SVS acoustic frontend and DDSP-SVC backend inference')
@click.argument(
    'proj', type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True,
        path_type=pathlib.Path, resolve_path=True
    ),
    metavar='DS_FILE'
)
@click.option(
    '--exp', type=str,
    required=True, metavar='EXP',
    callback=lambda ctx, param, value: find_exp(value),
    help='Selection of model'
)
@click.option(
    '--config', type=click.Path(file_okay=True, dir_okay=False),
    required=False,
    help='Config file used by the frontend experiment'
)
@click.option(
    '--ckpt', type=click.IntRange(min=0),
    required=False, metavar='STEPS',
    help='Selection of checkpoint training steps'
)
@click.option(
    '--ddsp-svc', type=click.Path(file_okay=False, dir_okay=True, path_type=pathlib.Path),
    required=False, default=pathlib.Path('../DDSP-SVC'),
    help='Path to DDSP-SVC repository'
)
@click.option(
    '--model', type=click.Path(file_okay=True, dir_okay=False, path_type=pathlib.Path),
    required=True,
    help='Path to DDSP-SVC reflow checkpoint'
)
@click.option(
    '--vocoder-ckpt', type=click.Path(file_okay=True, dir_okay=False, path_type=pathlib.Path),
    required=False,
    help='Override DDSP-SVC vocoder checkpoint path'
)
@click.option(
    '--spk', type=click.STRING,
    required=False,
    help='Speaker name or mixture of speakers'
)
@click.option(
    '--lang', type=click.STRING,
    required=False,
    help='Default language name'
)
@click.option(
    '--out', type=click.Path(
        file_okay=True, dir_okay=True, path_type=pathlib.Path
    ),
    required=False,
    help='Output wav path or output folder'
)
@click.option(
    '--title', type=click.STRING,
    required=False,
    help='Title of output file'
)
@click.option(
    '--num', type=click.IntRange(min=1),
    required=False, default=1,
    help='Number of runs'
)
@click.option(
    '--key', type=click.INT,
    required=False, default=0,
    help='Key transition of pitch'
)
@click.option(
    '--gender', type=click.FloatRange(min=-1, max=1),
    required=False,
    help='Formant shifting (gender control)'
)
@click.option(
    '--seed', type=click.INT,
    required=False, default=-1,
    help='Random seed of the inference'
)
@click.option(
    '--depth', type=click.FloatRange(min=0, max=1),
    required=False,
    help='Shallow diffusion depth'
)
@click.option(
    '--steps', type=click.IntRange(min=1),
    required=False,
    help='Frontend diffusion sampling steps'
)
@click.option(
    '--save-units', is_flag=True,
    help='Save intermediate unit payload'
)
@click.option(
    '--units-out', type=click.Path(file_okay=True, dir_okay=False, path_type=pathlib.Path),
    required=False,
    help='Intermediate unit payload path; implies --save-units'
)
@click.option(
    '--unit-dim', type=click.IntRange(min=1),
    required=False, default=768,
    help='Number of ContentVec unit channels'
)
@click.option(
    '--spk-id', type=click.INT,
    required=False, default=1,
    help='DDSP-SVC speaker id, 1-based'
)
@click.option(
    '--spk-mix-dict', type=click.STRING,
    required=False, default='None',
    help='DDSP-SVC speaker mix dict, e.g. "{1:0.5,2:0.5}"'
)
@click.option(
    '--backend-key', type=click.FLOAT,
    required=False, default=0.0,
    help='DDSP-SVC backend pitch shift in semitones'
)
@click.option(
    '--formant-shift-key', type=click.FLOAT,
    required=False, default=0.0,
    help='DDSP-SVC aug_shift/formant key'
)
@click.option(
    '--infer-step', type=click.INT,
    required=False, default=-1,
    help='DDSP-SVC backend reflow steps; -1 uses backend config'
)
@click.option(
    '--method', type=click.Choice(['auto', 'euler', 'rk4']),
    required=False, default='auto',
    help='DDSP-SVC backend sampling method'
)
@click.option(
    '--t-start', type=click.FLOAT,
    required=False,
    help='DDSP-SVC backend reflow t_start; default uses backend config or 0'
)
@click.option(
    '--volume-default', type=click.FLOAT,
    required=False, default=0.1,
    help='Fallback volume if frontend tensor has no volume channel'
)
@click.option(
    '--device', type=click.STRING,
    required=False,
    help='cpu/cuda; default auto'
)
def acoustic(
        proj: pathlib.Path,
        exp: str,
        config: str,
        ckpt: int,
        ddsp_svc: pathlib.Path,
        model: pathlib.Path,
        vocoder_ckpt: pathlib.Path,
        spk: str,
        lang: str,
        out: pathlib.Path,
        title: str,
        num: int,
        key: int,
        gender: float,
        seed: int,
        depth: float,
        steps: int,
        save_units: bool,
        units_out: pathlib.Path,
        unit_dim: int,
        spk_id: int,
        spk_mix_dict: str,
        backend_key: float,
        formant_shift_key: float,
        infer_step: int,
        method: str,
        t_start: float,
        volume_default: float,
        device: str
):
    name = proj.stem if not title else title
    if out is None:
        wav_path = proj.with_name(name + '.wav')
    elif out.suffix.lower() == '.wav':
        wav_path = out
    else:
        wav_path = out / f'{name}.wav'
    units_path = units_out or wav_path.with_suffix('.units.pt')

    with open(proj, 'r', encoding='utf-8') as f:
        params = json.load(f)

    if not isinstance(params, list):
        params = [params]

    if len(params) == 0:
        print('The input file is empty.')
        exit()

    from utils.infer_utils import trans_key, parse_commandline_spk_mix

    if key != 0:
        params = trans_key(params, key)
        key_suffix = '%+dkey' % key
        if not title:
            name += key_suffix
        print(f'| key transition: {key:+d}')

    sys.argv = [
        sys.argv[0],
        '--exp_name',
        exp,
        '--infer'
    ]
    if config:
        sys.argv.extend(['--config', config])
    from utils.hparams import set_hparams, hparams
    set_hparams()

    is_unit_frontend = hparams.get('task_cls') == 'training.unit_acoustic_task.UnitAcousticTask'
    if not is_unit_frontend:
        raise click.ClickException('DDSP-SVS acoustic inference requires a UnitAcousticTask checkpoint.')
    if is_unit_frontend and 'unit_frontend_infer_aux' not in hparams:
        hparams['unit_frontend_infer_aux'] = True

    # For compatibility:
    # migrate timesteps, K_step, K_step_infer, diff_speedup to time_scale_factor, T_start, T_start_infer, sampling_steps
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

    if depth is not None:
        hparams['unit_frontend_infer_aux'] = False
        print('| unit frontend: use shallow diffusion because --depth was specified')
        assert depth <= 1 - hparams['T_start'], (
            f"Depth should not be larger than 1 - T_start ({1 - hparams['T_start']})"
        )
        hparams['K_step_infer'] = round(hparams['timesteps'] * depth)
        hparams['T_start_infer'] = 1 - depth
    elif hparams.get('unit_frontend_infer_aux', False):
        hparams['T_start_infer'] = 1.0
        hparams['K_step_infer'] = 0
        print('| unit frontend: use auxiliary decoder output')
    if steps is not None:
        if hparams['use_shallow_diffusion']:
            step_size = (1 - hparams['T_start_infer']) / steps
            if 'K_step_infer' in hparams:
                hparams['diff_speedup'] = round(step_size * hparams['K_step_infer'])
        else:
            if 'timesteps' in hparams:
                hparams['diff_speedup'] = round(hparams['timesteps'] / steps)
        hparams['sampling_steps'] = steps

    spk_mix = parse_commandline_spk_mix(spk) if hparams['use_spk_id'] and spk is not None else None
    for param in params:
        if gender is not None and hparams['use_key_shift_embed']:
            param['gender'] = gender
        if spk_mix is not None:
            param['spk_mix'] = spk_mix
        if lang is not None:
            param['lang'] = lang

    import torch
    import tqdm
    from inference.ds_acoustic import DiffSingerAcousticInfer
    from modules.ddsp_svc_backend import load_backend, save_rendered_payload

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    infer_ins = DiffSingerAcousticInfer(device=device, load_vocoder=False, ckpt_steps=ckpt)
    print(f'| Model: {type(infer_ins.model)}')

    backend_model, backend_vocoder, backend_args = load_backend(
        ddsp_svc, model, device=device, vocoder_ckpt=vocoder_ckpt
    )
    backend_options = SimpleNamespace(
        unit_dim=unit_dim,
        volume_default=volume_default,
        backend_key=backend_key,
        spk_mix_dict=spk_mix_dict,
        spk_id=spk_id,
        formant_shift_key=formant_shift_key,
        infer_step=infer_step,
        method=method,
        t_start=t_start,
    )

    try:
        batches = [infer_ins.preprocess_input(param, idx=i) for i, param in enumerate(params)]
        for i in range(num):
            if num > 1:
                run_wav_path = wav_path.with_name(f'{wav_path.stem}-{str(i).zfill(3)}{wav_path.suffix}')
                run_units_path = units_path.with_name(f'{units_path.stem}-{str(i).zfill(3)}{units_path.suffix}')
            else:
                run_wav_path = wav_path
                run_units_path = units_path

            payload = []
            for param, batch in tqdm.tqdm(zip(params, batches), desc='frontend segments', total=len(params)):
                if 'seed' in param:
                    torch.manual_seed(param['seed'] & 0xffff_ffff)
                    torch.cuda.manual_seed_all(param['seed'] & 0xffff_ffff)
                elif seed >= 0:
                    torch.manual_seed(seed & 0xffff_ffff)
                    torch.cuda.manual_seed_all(seed & 0xffff_ffff)

                units = infer_ins.forward_model(batch)
                payload.append({
                    'offset': param.get('offset', 0.0),
                    'mel': units.cpu(),
                    'f0': batch['f0'].cpu(),
                })

            if save_units or units_out is not None:
                run_units_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, run_units_path)
                print(f'| save units: {run_units_path}')
            save_rendered_payload(
                backend_model, backend_vocoder, backend_args, payload, backend_options, device, run_wav_path
            )
            print(f'| save audio: {run_wav_path}')
    except KeyboardInterrupt:
        exit(-1)


@main.command(help='Run DiffSinger variance model inference')
@click.argument(
    'proj', type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True,
        path_type=pathlib.Path, resolve_path=True
    ),
    metavar='DS_FILE'
)
@click.option(
    '--exp', type=str,
    required=True, metavar='EXP',
    callback=lambda ctx, param, value: find_exp(value),
    help='Selection of model'
)
@click.option(
    '--ckpt', type=click.IntRange(min=0),
    required=False, metavar='STEPS',
    help='Selection of checkpoint training steps'
)
@click.option(
    '--predict', type=click.STRING,
    multiple=True, metavar='TAGS',
    help='Parameters to predict'
)
@click.option(
    '--spk', type=click.STRING,
    required=False,
    help='Speaker name or mixture of speakers'
)
@click.option(
    '--lang', type=click.STRING,
    required=False,
    help='Default language name'
)
@click.option(
    '--out', type=click.Path(
        file_okay=False, dir_okay=True, path_type=pathlib.Path
    ),
    required=False,
    help='Path of the output folder'
)
@click.option(
    '--title', type=click.STRING,
    required=False,
    help='Title of output file'
)
@click.option(
    '--num', type=click.IntRange(min=1),
    required=False, default=1,
    help='Number of runs'
)
@click.option(
    '--key', type=click.INT,
    required=False, default=0,
    help='Key transition of pitch'
)
@click.option(
    '--expr', type=click.FloatRange(min=0, max=1),
    required=False, help='Static expressiveness control'
)
@click.option(
    '--seed', type=click.INT,
    required=False, default=-1,
    help='Random seed of the inference'
)
@click.option(
    '--steps', type=click.IntRange(min=1),
    required=False,
    help='Diffusion sampling steps'
)
def variance(
        proj: pathlib.Path,
        exp: str,
        ckpt: int,
        spk: str,
        lang: str,
        predict: Tuple[str],
        out: pathlib.Path,
        title: str,
        num: int,
        key: int,
        expr: float,
        seed: int,
        steps: int
):
    name = proj.stem if not title else title
    if out is None:
        out = proj.parent
    if (not out or out.resolve() == proj.parent.resolve()) and not title:
        name += '_variance'

    with open(proj, 'r', encoding='utf-8') as f:
        params = json.load(f)

    if not isinstance(params, list):
        params = [params]
    params = [OrderedDict(p) for p in params]

    if len(params) == 0:
        print('The input file is empty.')
        exit()

    from utils.infer_utils import trans_key, parse_commandline_spk_mix

    if key != 0:
        params = trans_key(params, key)
        key_suffix = '%+dkey' % key
        if not title:
            name += key_suffix
        print(f'| key transition: {key:+d}')

    sys.argv = [
        sys.argv[0],
        '--exp_name',
        exp,
        '--infer'
    ]
    from utils.hparams import set_hparams, hparams
    set_hparams()

    # For compatibility:
    # migrate timesteps, K_step, K_step_infer, diff_speedup to time_scale_factor, T_start, T_start_infer, sampling_steps
    if 'diff_speedup' not in hparams and 'pndm_speedup' in hparams:
        hparams['diff_speedup'] = hparams['pndm_speedup']
    if 'sampling_steps' not in hparams:
        hparams['sampling_steps'] = hparams['timesteps'] // hparams['diff_speedup']
    if 'time_scale_factor' not in hparams:
        hparams['time_scale_factor'] = hparams['timesteps']

    if steps is not None:
        if 'timesteps' in hparams:
            hparams['diff_speedup'] = round(hparams['timesteps'] / steps)
        hparams['sampling_steps'] = steps

    spk_mix = parse_commandline_spk_mix(spk) if hparams['use_spk_id'] and spk is not None else None
    for param in params:
        if expr is not None:
            param['expr'] = expr
        if spk_mix is not None:
            param['ph_spk_mix_backup'] = param.get('ph_spk_mix')
            param['spk_mix_backup'] = param.get('spk_mix')
            param['ph_spk_mix'] = param['spk_mix'] = spk_mix
        if lang is not None:
            param['lang'] = lang

    from inference.ds_variance import DiffSingerVarianceInfer
    infer_ins = DiffSingerVarianceInfer(ckpt_steps=ckpt, predictions=set(predict))
    print(f'| Model: {type(infer_ins.model)}')

    try:
        infer_ins.run_inference(
            params, out_dir=out, title=name,
            num_runs=num, seed=seed
        )
    except KeyboardInterrupt:
        exit(-1)


if __name__ == '__main__':
    main()
