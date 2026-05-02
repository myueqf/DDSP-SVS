import pathlib

import numpy as np
import torch

from modules.contentvec import UnitsEncoder, VolumeExtractor
from preprocessing.acoustic_binarizer import AcousticBinarizer
from utils.hparams import hparams


UNIT_ITEM_ATTRIBUTES = [
    'spk_id',
    'languages',
    'tokens',
    'mel2ph',
    'f0',
    'units',
    'volume',
    'energy',
    'breathiness',
    'voicing',
    'tension',
    'key_shift',
    'speed',
]

units_encoder = None
volume_extractor = None


def _trim_frame_fields(item, length):
    for key in ['mel2ph', 'f0', 'units', 'volume', 'energy', 'breathiness', 'voicing', 'tension']:
        if key in item and item[key] is not None:
            item[key] = item[key][:length]
    item['length'] = length


class UnitAcousticBinarizer(AcousticBinarizer):
    def __init__(self):
        super().__init__()
        self.data_attrs = UNIT_ITEM_ATTRIBUTES

    @torch.no_grad()
    def process_item(self, item_name, meta_data, binarization_args):
        processed_input = super().process_item(item_name, meta_data, binarization_args)
        if processed_input is None:
            return None

        global units_encoder, volume_extractor
        if units_encoder is None or volume_extractor is None:
            contentvec_args = hparams.get('contentvec_args', {})
            encoder = contentvec_args.get('encoder', 'contentvec768l12tta2x')
            encoder_ckpt = contentvec_args.get('encoder_ckpt', 'pretrain/contentvec/checkpoint_best_legacy_500.pt')
            encoder_sample_rate = contentvec_args.get('encoder_sample_rate', 16000)
            encoder_hop_size = contentvec_args.get('encoder_hop_size', 160)
            units_encoder = UnitsEncoder(
                encoder,
                str(pathlib.Path(hparams.get('ddsp_svc_path', '../DDSP-SVC')).resolve() / encoder_ckpt)
                if not pathlib.Path(encoder_ckpt).is_absolute()
                else encoder_ckpt,
                encoder_sample_rate,
                encoder_hop_size,
                device=self.device
            )
            volume_extractor = VolumeExtractor(
                hparams['hop_size'],
                hparams.get('volume_smooth_size', hparams['win_size'])
            )

        waveform, _ = __import__('librosa').load(
            meta_data['wav_fn'], sr=hparams['audio_sample_rate'], mono=True
        )
        audio_t = torch.from_numpy(waveform).float().to(self.device)[None, :]
        units = units_encoder.encode(audio_t, hparams['audio_sample_rate'], hparams['hop_size'])
        processed_input['units'] = units.squeeze(0).detach().cpu().numpy().astype(np.float32)
        processed_input['volume'] = volume_extractor.extract(waveform).astype(np.float32)

        frame_len = min(
            processed_input['mel2ph'].shape[0],
            processed_input['f0'].shape[0],
            processed_input['units'].shape[0],
            processed_input['volume'].shape[0],
        )
        _trim_frame_fields(processed_input, frame_len)
        processed_input['seconds'] = frame_len * hparams['hop_size'] / hparams['audio_sample_rate']
        return processed_input
