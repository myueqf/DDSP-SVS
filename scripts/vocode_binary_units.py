import argparse
import pathlib
import pickle
import sys

import torch

root_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / 'scripts'))

from utils.indexed_datasets import IndexedDataset
from vocode_units import load_backend, synthesize_segment
from utils.infer_utils import save_wav


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run DDSP-SVC backend from ground-truth units in a DiffSinger binary dataset.'
    )
    parser.add_argument('--binary-data-dir', type=pathlib.Path, required=True)
    parser.add_argument('--prefix', type=str, default='valid', choices=['train', 'valid'])
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--name', type=str, default=None, help='Valid-set item name without extension')
    parser.add_argument('--ddsp-svc', type=pathlib.Path, default=pathlib.Path('../DDSP-SVC'))
    parser.add_argument('--model', type=pathlib.Path, required=True)
    parser.add_argument('--vocoder-ckpt', type=pathlib.Path, default=None)
    parser.add_argument('--out', type=pathlib.Path, required=True)
    parser.add_argument('--unit-dim', type=int, default=768)
    parser.add_argument('--spk-id', type=int, default=1)
    parser.add_argument('--spk-mix-dict', type=str, default='None')
    parser.add_argument('--key', type=float, default=0.0)
    parser.add_argument('--formant-shift-key', type=float, default=0.0)
    parser.add_argument('--infer-step', type=int, default=-1)
    parser.add_argument('--method', type=str, default='auto', choices=['auto', 'euler', 'rk4'])
    parser.add_argument('--t-start', type=float, default=None)
    parser.add_argument('--volume-default', type=float, default=0.1)
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
    device = cmd.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    item_index = resolve_index(cmd.binary_data_dir, cmd.prefix, cmd.index, cmd.name)
    ds = IndexedDataset(cmd.binary_data_dir, cmd.prefix)
    item = ds[item_index]

    units = item['units'].float()
    volume = item['volume'].float()
    f0 = item['f0'].float()
    pred = torch.cat([units, volume[:, None]], dim=-1)[None]
    entry = {
        'mel': pred,
        'f0': f0[None],
        'offset': 0.0,
    }

    model, vocoder, args = load_backend(
        cmd.ddsp_svc, cmd.model, device=device, vocoder_ckpt=cmd.vocoder_ckpt
    )
    wav = synthesize_segment(model, vocoder, args, entry, cmd, device)
    save_wav(wav, cmd.out, args.data.sampling_rate)
    print(f'| item: {item_index}')
    print(f'| units: {tuple(units.shape)}, volume: {tuple(volume.shape)}, f0: {tuple(f0.shape)}')
    print(f'| save audio: {cmd.out}')


if __name__ == '__main__':
    main()
