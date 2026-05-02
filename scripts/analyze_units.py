import argparse
import pathlib
import pickle
import sys

import torch
import torch.nn.functional as F

root_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from utils.indexed_datasets import IndexedDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Compare predicted unit frontend output against binary gt units.')
    parser.add_argument('--pred', type=pathlib.Path, required=True, help='.units.pt saved by scripts/infer.py acoustic --mel')
    parser.add_argument('--binary-data-dir', type=pathlib.Path, required=True)
    parser.add_argument('--prefix', type=str, default='valid', choices=['train', 'valid'])
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--unit-dim', type=int, default=768)
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


def load_pred(path, unit_dim):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(payload, list):
        if len(payload) != 1:
            print(f'| pred payload has {len(payload)} segments; using segment 0')
        pred = payload[0]['mel']
        pred_f0 = payload[0].get('f0')
    else:
        pred = payload
        pred_f0 = None
    if pred.dim() == 3:
        pred = pred[0]
    units = pred[:, :unit_dim].float()
    volume = pred[:, unit_dim].float() if pred.shape[-1] > unit_dim else None
    return units, volume, pred_f0


def describe(name, x):
    x = x.float()
    flat = x.flatten()
    print(
        f'{name}: shape={tuple(x.shape)} mean={flat.mean().item():.6f} '
        f'std={flat.std().item():.6f} min={flat.min().item():.6f} max={flat.max().item():.6f}'
    )


def describe_units(prefix, x):
    describe(prefix, x)
    frame_std = x.std(dim=0)
    chan_std = x.std(dim=1)
    delta = x[1:] - x[:-1] if x.shape[0] > 1 else x.new_zeros(0, x.shape[-1])
    print(
        f'{prefix}/channel_std: mean={frame_std.mean().item():.6f} '
        f'median={frame_std.median().item():.6f} min={frame_std.min().item():.6f} max={frame_std.max().item():.6f}'
    )
    print(
        f'{prefix}/frame_std: mean={chan_std.mean().item():.6f} '
        f'median={chan_std.median().item():.6f} min={chan_std.min().item():.6f} max={chan_std.max().item():.6f}'
    )
    if delta.numel() > 0:
        describe(prefix + '/delta', delta)


def main():
    args = parse_args()
    item_index = resolve_index(args.binary_data_dir, args.prefix, args.index, args.name)
    ds = IndexedDataset(args.binary_data_dir, args.prefix)
    item = ds[item_index]

    gt_units = item['units'].float()
    gt_volume = item['volume'].float()
    gt_f0 = item['f0'].float()
    pred_units, pred_volume, pred_f0 = load_pred(args.pred, args.unit_dim)

    length = min(gt_units.shape[0], pred_units.shape[0])
    gt_units = gt_units[:length]
    pred_units = pred_units[:length]
    gt_volume = gt_volume[:length]
    if pred_volume is not None:
        pred_volume = pred_volume[:length]

    print(f'| item index: {item_index}')
    print(f'| compare length: {length}')
    describe_units('gt_units', gt_units)
    describe_units('pred_units', pred_units)

    diff = pred_units - gt_units
    describe('pred_minus_gt', diff)
    cos = F.cosine_similarity(pred_units, gt_units, dim=-1)
    print(
        f'cosine(frame): mean={cos.mean().item():.6f} median={cos.median().item():.6f} '
        f'min={cos.min().item():.6f} max={cos.max().item():.6f}'
    )
    print(f'l1={diff.abs().mean().item():.6f} l2={torch.mean(diff ** 2).item():.6f}')

    describe('gt_volume', gt_volume)
    if pred_volume is not None:
        describe('pred_volume', pred_volume)
        vdiff = pred_volume - gt_volume
        describe('pred_minus_gt_volume', vdiff)
        vcorr = torch.corrcoef(torch.stack([gt_volume, pred_volume]))[0, 1]
        print(f'volume_corr={vcorr.item():.6f}')
    if pred_f0 is not None:
        if pred_f0.dim() == 3:
            pred_f0 = pred_f0[0, :, 0]
        elif pred_f0.dim() == 2:
            pred_f0 = pred_f0[0]
        pred_f0 = pred_f0[:min(gt_f0.shape[0], pred_f0.shape[0])].float()
        describe('gt_f0', gt_f0[:pred_f0.shape[0]])
        describe('pred_f0', pred_f0)


if __name__ == '__main__':
    main()
