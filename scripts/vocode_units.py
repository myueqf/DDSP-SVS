import argparse
import pathlib
import sys

import torch

root_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from modules.ddsp_svc_backend import load_backend, save_rendered_payload


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a DDSP-SVC backend from DiffSinger unit frontend output.'
    )
    parser.add_argument('input', type=pathlib.Path, help='Path to .units.pt saved by scripts/infer.py acoustic --mel')
    parser.add_argument('--ddsp-svc', type=pathlib.Path, default=pathlib.Path('../DDSP-SVC'),
                        help='Path to DDSP-SVC repository')
    parser.add_argument('--model', type=pathlib.Path, required=True,
                        help='Path to DDSP-SVC reflow checkpoint')
    parser.add_argument('--vocoder-ckpt', type=pathlib.Path, default=None,
                        help='Override DDSP-SVC vocoder checkpoint path')
    parser.add_argument('--out', type=pathlib.Path, default=None, help='Output wav path')
    parser.add_argument('--unit-dim', type=int, default=768, help='Number of ContentVec unit channels')
    parser.add_argument('--spk-id', type=int, default=1, help='DDSP-SVC speaker id, 1-based')
    parser.add_argument('--spk-mix-dict', type=str, default='None',
                        help='DDSP-SVC speaker mix dict, e.g. "{1:0.5,2:0.5}"')
    parser.add_argument('--key', dest='backend_key', type=float, default=0.0, help='Pitch shift in semitones')
    parser.add_argument('--formant-shift-key', type=float, default=0.0,
                        help='DDSP-SVC aug_shift/formant key')
    parser.add_argument('--infer-step', type=int, default=-1,
                        help='Backend reflow steps; -1 uses DDSP-SVC config')
    parser.add_argument('--method', type=str, default='auto', choices=['auto', 'euler', 'rk4'],
                        help='Backend reflow sampling method')
    parser.add_argument('--t-start', type=float, default=None,
                        help='Backend reflow t_start; default uses DDSP-SVC config or 0')
    parser.add_argument('--volume-default', type=float, default=0.1,
                        help='Fallback volume if input tensor has no predicted volume channel')
    parser.add_argument('--device', type=str, default=None, help='cpu/cuda; default auto')
    return parser.parse_args()


def main():
    cmd = parse_args()
    device = cmd.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    payload = torch.load(cmd.input, map_location='cpu', weights_only=False)
    if not isinstance(payload, list):
        raise ValueError('Expected a list saved by scripts/infer.py acoustic --mel.')

    model, vocoder, args = load_backend(
        cmd.ddsp_svc, cmd.model, device=device, vocoder_ckpt=cmd.vocoder_ckpt
    )
    out_path = cmd.out or cmd.input.with_suffix('.wav')
    save_rendered_payload(model, vocoder, args, payload, cmd, device, out_path)
    print(f'| save audio: {out_path}')


if __name__ == '__main__':
    main()
