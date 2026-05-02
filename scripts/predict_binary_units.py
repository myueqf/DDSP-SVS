import argparse
import pathlib
import pickle
import sys

import torch

root_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a unit acoustic checkpoint on one binary dataset item and save .units.pt-compatible output.'
    )
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--config', type=str, default='', help='Config file used by the experiment')
    parser.add_argument('--ckpt', type=int, default=None)
    parser.add_argument('--binary-data-dir', type=pathlib.Path, required=True)
    parser.add_argument('--prefix', type=str, default='valid', choices=['train', 'valid'])
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--out', type=pathlib.Path, required=True)
    parser.add_argument('--device', type=str, default=None)
    return parser.parse_args()


def resolve_index(binary_data_dir, prefix, index, name):
    if name is None:
        return index
    with open(binary_data_dir / f'{prefix}.meta', 'rb') as f:
        meta = pickle.load(f)
    if 'names' not in meta:
        raise ValueError('--name lookup is only available when metadata contains names, usually valid set.')
    names = [pathlib.Path(n).stem for n in meta['names']]
    if name not in names:
        raise ValueError(f'Name {name!r} not found in {prefix}.meta.')
    return names.index(name)


def main():
    cmd = parse_args()
    sys.argv = [sys.argv[0], '--exp_name', cmd.exp, '--infer']
    if cmd.config:
        sys.argv.extend(['--config', cmd.config])
    from utils.hparams import set_hparams, hparams
    set_hparams()
    if hparams.get('task_cls') == 'training.unit_acoustic_task.UnitAcousticTask' \
            and 'unit_frontend_infer_aux' not in hparams:
        hparams['unit_frontend_infer_aux'] = True

    from inference.ds_acoustic import DiffSingerAcousticInfer
    from utils.indexed_datasets import IndexedDataset
    from modules.fastspeech.param_adaptor import VARIANCE_CHECKLIST

    device = cmd.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    infer = DiffSingerAcousticInfer(device=device, load_vocoder=False, ckpt_steps=cmd.ckpt)
    item_index = resolve_index(cmd.binary_data_dir, cmd.prefix, cmd.index, cmd.name)
    ds = IndexedDataset(cmd.binary_data_dir, cmd.prefix)
    item = ds[item_index]

    sample = {
        'tokens': item['tokens'][None].to(device),
        'mel2ph': item['mel2ph'][None].to(device),
        'f0': item['f0'][None].to(device),
    }
    if hparams['use_spk_id']:
        sample['spk_embed_id'] = torch.LongTensor([int(item['spk_id'])]).to(device)
    else:
        sample['spk_embed_id'] = None
    if hparams['use_lang_id']:
        sample['languages'] = item['languages'][None].to(device)
    else:
        sample['languages'] = None
    for v_name in VARIANCE_CHECKLIST:
        if hparams.get(f'use_{v_name}_embed', False):
            sample[v_name] = item[v_name][None].to(device)
    if hparams['use_key_shift_embed']:
        sample['key_shift'] = torch.FloatTensor([[float(item['key_shift'])]]).to(device)
    else:
        sample['key_shift'] = None
    if hparams['use_speed_embed']:
        sample['speed'] = torch.FloatTensor([[float(item['speed'])]]).to(device)
    else:
        sample['speed'] = None

    variances = {
        v_name: sample[v_name]
        for v_name in VARIANCE_CHECKLIST
        if hparams.get(f'use_{v_name}_embed', False)
    }
    with torch.no_grad():
        output = infer.model(
            sample['tokens'],
            mel2ph=sample['mel2ph'],
            f0=sample['f0'],
            **variances,
            key_shift=sample['key_shift'],
            speed=sample['speed'],
            spk_embed_id=sample['spk_embed_id'],
            languages=sample['languages'],
            infer=True
        )

    pred = output.aux_out if hparams.get('unit_frontend_infer_aux', False) else output.diff_out
    if pred is None:
        pred = output.diff_out if output.aux_out is None else output.aux_out

    payload = [{
        'offset': 0.0,
        'mel': pred.detach().cpu(),
        'f0': sample['f0'].detach().cpu(),
    }]
    cmd.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cmd.out)
    print(f'| item: {item_index}')
    print(f'| pred: {tuple(pred.shape)}')
    print(f'| save: {cmd.out}')


if __name__ == '__main__':
    main()
