import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample
from transformers import HubertConfig, HubertModel


def _patch_fairseq_pad_to_multiple():
    import fairseq.models.wav2vec.wav2vec2 as fairseq_wav2vec2

    def pad_to_multiple(x, multiple, dim=-1, value=0):
        if x is None:
            return None, 0
        tsz = x.size(dim)
        if isinstance(tsz, torch.Tensor):
            tsz = int(tsz.item())
        remainder = ((tsz + multiple - 1) // multiple) * multiple - tsz
        if remainder == 0:
            return x, 0
        pad_offset = (0,) * (-1 - dim) * 2
        return F.pad(x, (*pad_offset, 0, remainder), value=value), remainder

    fairseq_wav2vec2.pad_to_multiple = pad_to_multiple


class FairseqContentVec768L12(torch.nn.Module):
    def __init__(self, path, device='cpu'):
        super().__init__()
        _patch_fairseq_pad_to_multiple()
        from fairseq import checkpoint_utils
        real_load = torch.load

        def patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return real_load(*args, **kwargs)

        torch.load = patched_load
        try:
            models, _, _ = checkpoint_utils.load_model_ensemble_and_task([path])
        finally:
            torch.load = real_load
        self.hubert = models[0].to(device)
        self.hubert.eval()

    def _forward_once(self, audio):
        out = self.hubert(source=audio, padding_mask=None, mask=False, features_only=True, output_layer=12)
        return out["x"]

    def forward(self, audio):
        with torch.no_grad():
            return self._forward_once(audio)


class FairseqContentVec768L12TTA2X(FairseqContentVec768L12):
    def forward(self, audio):
        with torch.no_grad():
            feats = self._forward_once(audio)
            audio_shift = F.pad(audio, (160, 0))
            feats2 = self._forward_once(audio_shift)
            feats_tail = feats[:, -1:, :]
            feats_aligned = torch.cat((feats, feats_tail), dim=1)[:, :feats2.shape[1], :]
            feats_tta = torch.cat((feats2, feats_aligned), dim=2).reshape(feats.shape[0], -1, feats.shape[-1])
            target_frames = feats.shape[1] + feats2.shape[1] - 1
            return feats_tta[:, 1:1 + target_frames, :]


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = torch.nn.Linear(config.hidden_size, config.classifier_proj_size)


class Audio2ContentVec768L12:
    def __init__(self, path, device='cpu'):
        self.device = device
        self.hubert = HubertModelWithFinalProj(HubertConfig())
        # ContentVec legacy checkpoints may contain fairseq metadata objects.
        # This mirrors DDSP-SVC's trusted local checkpoint loading behavior.
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        self.hubert.load_state_dict(checkpoint)
        self.hubert = self.hubert.to(self.device)
        self.hubert.eval()

    def __call__(self, audio):
        with torch.no_grad():
            return self.hubert(audio)["last_hidden_state"]


class Audio2ContentVec768L12TTA2X(Audio2ContentVec768L12):
    def __call__(self, audio):
        with torch.no_grad():
            feats = self.hubert(audio)["last_hidden_state"]
            audio = F.pad(audio, (160, 0))
            feats2 = self.hubert(audio)["last_hidden_state"]
            n = feats2.shape[1] - feats.shape[1]
            if n > 0:
                feats = F.pad(feats, (0, 0, 0, 1))
            feats_tta = torch.cat((feats2, feats), dim=2).reshape(feats.shape[0], -1, feats.shape[-1])
            feats_tta = feats_tta[:, 1:, :]
            if n > 0:
                feats_tta = feats_tta[:, :-1, :]
            return feats_tta


class UnitsEncoder:
    def __init__(self, encoder, encoder_ckpt, encoder_sample_rate=16000, encoder_hop_size=320, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        if encoder == 'contentvec768l12':
            try:
                self.model = Audio2ContentVec768L12(encoder_ckpt, device=device)
            except Exception:
                self.model = FairseqContentVec768L12(encoder_ckpt, device=device)
        elif encoder == 'contentvec768l12tta2x':
            try:
                self.model = Audio2ContentVec768L12TTA2X(encoder_ckpt, device=device)
            except Exception:
                self.model = FairseqContentVec768L12TTA2X(encoder_ckpt, device=device)
        else:
            raise ValueError(f'Unsupported local units encoder: {encoder}')
        self.resample_kernel = {}
        self.encoder_sample_rate = encoder_sample_rate
        self.encoder_hop_size = encoder_hop_size

    def encode(self, audio, sample_rate, hop_size):
        if sample_rate == self.encoder_sample_rate:
            audio_res = audio
        else:
            key = str(sample_rate)
            if key not in self.resample_kernel:
                self.resample_kernel[key] = Resample(
                    sample_rate, self.encoder_sample_rate, lowpass_filter_width=128
                ).to(self.device)
            audio_res = self.resample_kernel[key](audio)

        if audio_res.size(-1) < 400:
            audio_res = F.pad(audio_res, (0, 400 - audio_res.size(-1)))
        units = self.model(audio_res)

        n_frames = audio.size(-1) // hop_size + 1
        ratio = (hop_size / sample_rate) / (self.encoder_hop_size / self.encoder_sample_rate)
        index = torch.clamp(
            torch.round(ratio * torch.arange(n_frames).to(self.device)).long(),
            max=units.size(1) - 1
        )
        return torch.gather(units, 1, index[None, :, None].repeat([1, 1, units.size(-1)]))


class VolumeExtractor:
    def __init__(self, hop_size=512, win_size=2048):
        self.hop_size = hop_size
        self.win_size = win_size

    def extract(self, audio):
        n_frames = int(len(audio) // self.hop_size) + 1
        audio = np.pad(audio, (self.win_size // 2, (self.win_size + 1) // 2), mode='reflect')
        audio2 = audio ** 2
        mean = np.array([
            np.mean(audio[n * self.hop_size:n * self.hop_size + self.win_size])
            for n in range(n_frames)
        ])
        mean_square = np.array([
            np.mean(audio2[n * self.hop_size:n * self.hop_size + self.win_size])
            for n in range(n_frames)
        ])
        return np.sqrt(np.clip(mean_square - mean ** 2, 0, None))
