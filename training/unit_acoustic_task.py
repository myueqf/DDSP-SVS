import torch
import torch.nn.functional as F

import utils
from basics.base_dataset import BaseDataset
from modules.aux_decoder import build_aux_loss
from modules.losses import DiffusionLoss, RectifiedFlowLoss
from modules.toplevel import DiffSingerAcoustic
from training.acoustic_task import AcousticTask, ShallowDiffusionOutput
from utils.hparams import hparams


def unit_model_out_dims():
    return hparams['unit_dim'] + (1 if hparams.get('predict_volume', False) else 0)


class UnitAcousticDataset(BaseDataset):
    def __init__(self, prefix, preload=False):
        super().__init__(prefix, hparams['dataset_size_key'], preload)
        self.required_variances = {}
        if hparams['use_energy_embed']:
            self.required_variances['energy'] = 0.0
        if hparams['use_breathiness_embed']:
            self.required_variances['breathiness'] = 0.0
        if hparams['use_voicing_embed']:
            self.required_variances['voicing'] = 0.0
        if hparams['use_tension_embed']:
            self.required_variances['tension'] = 0.0
        self.need_key_shift = hparams['use_key_shift_embed']
        self.need_speed = hparams['use_speed_embed']
        self.need_spk_id = hparams['use_spk_id']
        self.need_lang_id = hparams['use_lang_id']

    def collater(self, samples):
        batch = super().collater(samples)
        if batch['size'] == 0:
            return batch
        batch.update({
            'tokens': utils.collate_nd([s['tokens'] for s in samples], 0),
            'mel2ph': utils.collate_nd([s['mel2ph'] for s in samples], 0),
            'f0': utils.collate_nd([s['f0'] for s in samples], 0.0),
            'units': utils.collate_nd([s['units'] for s in samples], 0.0),
            'volume': utils.collate_nd([s['volume'] for s in samples], 0.0),
        })
        for v_name, v_pad in self.required_variances.items():
            batch[v_name] = utils.collate_nd([s[v_name] for s in samples], v_pad)
        if self.need_key_shift:
            batch['key_shift'] = torch.FloatTensor([s['key_shift'] for s in samples])[:, None]
        if self.need_speed:
            batch['speed'] = torch.FloatTensor([s['speed'] for s in samples])[:, None]
        if self.need_spk_id:
            batch['spk_ids'] = torch.LongTensor([s['spk_id'] for s in samples])
        if self.need_lang_id:
            batch['languages'] = utils.collate_nd([s['languages'] for s in samples], 0)
        return batch


class UnitAcousticTask(AcousticTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = UnitAcousticDataset
        self.lambda_aux_unit_cosine = hparams.get('lambda_aux_unit_cosine', 0.0)
        self.lambda_aux_unit_delta = hparams.get('lambda_aux_unit_delta', 0.0)

    def _build_model(self):
        return DiffSingerAcoustic(
            vocab_size=len(self.phoneme_dictionary),
            out_dims=unit_model_out_dims()
        )

    def build_losses_and_metrics(self):
        if self.use_shallow_diffusion:
            self.aux_mel_loss = build_aux_loss(self.shallow_args['aux_decoder_arch'])
            self.lambda_aux_mel_loss = hparams['lambda_aux_mel_loss']
            self.register_validation_loss('aux_unit_loss')
        train_diffusion = not self.use_shallow_diffusion or self.train_diffusion
        if train_diffusion:
            if self.diffusion_type == 'ddpm':
                self.mel_loss = DiffusionLoss(loss_type=hparams['main_loss_type'])
            elif self.diffusion_type == 'reflow':
                self.mel_loss = RectifiedFlowLoss(
                    loss_type=hparams['main_loss_type'], log_norm=hparams['main_loss_log_norm']
                )
            else:
                raise ValueError(f"Unknown diffusion type: {self.diffusion_type}")
            self.register_validation_loss('unit_loss')

    @staticmethod
    def _masked_cosine_loss(pred, target, non_padding):
        cos = 1.0 - F.cosine_similarity(pred.float(), target.float(), dim=-1)
        mask = non_padding.squeeze(-1).float()
        return (cos * mask).sum() / mask.sum().clamp_min(1.0)

    @staticmethod
    def _delta(x):
        return x[:, 1:] - x[:, :-1]

    def run_model(self, sample, infer=False):
        txt_tokens = sample['tokens']
        target = sample['units']
        if hparams.get('predict_volume', False):
            target = torch.cat([target, sample['volume'][..., None]], dim=-1)
        mel2ph = sample['mel2ph']
        f0 = sample['f0']
        variances = {
            v_name: sample[v_name]
            for v_name in self.required_variances
        }
        key_shift = sample.get('key_shift')
        speed = sample.get('speed')
        spk_embed_id = sample['spk_ids'] if hparams['use_spk_id'] else None
        languages = sample['languages'] if hparams['use_lang_id'] else None

        output: ShallowDiffusionOutput = self.model(
            txt_tokens, mel2ph=mel2ph, f0=f0, **variances,
            key_shift=key_shift, speed=speed,
            spk_embed_id=spk_embed_id, languages=languages,
            gt_mel=target, infer=infer
        )

        if infer:
            return output

        losses = {}
        non_padding = (mel2ph > 0).unsqueeze(-1).float()
        if output.aux_out is not None:
            aux_out = output.aux_out
            norm_gt = self.model.aux_decoder.norm_spec(target)
            aux_unit_loss = self.lambda_aux_mel_loss * self.aux_mel_loss(aux_out, norm_gt)
            if self.lambda_aux_unit_cosine > 0:
                aux_unit_loss = aux_unit_loss + self.lambda_aux_unit_cosine * self._masked_cosine_loss(
                    aux_out, norm_gt, non_padding
                )
            if self.lambda_aux_unit_delta > 0 and aux_out.shape[1] > 1:
                delta_mask = non_padding[:, 1:] * non_padding[:, :-1]
                aux_unit_loss = aux_unit_loss + self.lambda_aux_unit_delta * F.l1_loss(
                    self._delta(aux_out) * delta_mask,
                    self._delta(norm_gt) * delta_mask
                )
            losses['aux_unit_loss'] = aux_unit_loss

        if output.diff_out is not None:
            if self.diffusion_type == 'ddpm':
                x_recon, x_noise = output.diff_out
                unit_loss = self.mel_loss(x_recon, x_noise, non_padding=non_padding)
            elif self.diffusion_type == 'reflow':
                v_pred, v_gt, t = output.diff_out
                unit_loss = self.mel_loss(v_pred, v_gt, t=t, non_padding=non_padding)
            else:
                raise ValueError(f"Unknown diffusion type: {self.diffusion_type}")
            losses['unit_loss'] = unit_loss

        return losses

    def _validation_step(self, sample, batch_idx):
        losses = self.run_model(sample, infer=False)
        return losses, sample['size']
