import numpy as np
import torch
import torch.nn.functional as F

from .unit2control import Unit2Control


class CombSubSuperFast(torch.nn.Module):
    def __init__(self,
            sampling_rate,
            block_size,
            win_length,
            n_unit=256,
            n_spk=1,
            num_layers=3,
            dim_model=256,
            use_norm=False,
            use_attention=False,
            use_pitch_aug=False):
        super().__init__()

        print(' [DDSP Model] Combtooth Subtractive Synthesiser')
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("win_length", torch.tensor(win_length))
        self.register_buffer("window", torch.hann_window(win_length))
        #Unit2Control
        split_map = {
            'harmonic_magnitude': win_length // 2 + 1,
            'harmonic_phase': win_length // 2 + 1,
            'noise_magnitude': win_length // 2 + 1,
            'noise_phase': win_length // 2 + 1
        }
        self.unit2ctrl = Unit2Control(
                            n_unit,
                            block_size,
                            n_spk,
                            split_map,
                            num_layers=num_layers,
                            dim_model=dim_model,
                            use_norm=use_norm,
                            use_attention=use_attention,
                            use_pitch_aug=use_pitch_aug)

    def fast_source_gen(self, f0_frames):
        n = torch.arange(self.block_size, device=f0_frames.device)
        s0 = f0_frames / self.sampling_rate
        ds0 = F.pad(s0[:, 1:, :] - s0[:, :-1, :], (0, 0, 0, 1))
        rad = s0 * (n + 1) + 0.5 * ds0 * n * (n + 1) / self.block_size
        s0 = s0 + ds0 * n / self.block_size
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0_frames)
        rad += F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        rad -= torch.round(rad)
        combtooth = torch.sinc(rad / (s0 + 1e-5)).reshape(f0_frames.shape[0], -1)
        return combtooth

    def forward(self, units_frames, f0_frames, volume_frames, spk_id=None, spk_mix_dict=None, aug_shift=None, initial_phase=None, infer=True, **kwargs):
        '''
            units_frames: B x n_frames x n_unit
            f0_frames: B x n_frames x 1
            volume_frames: B x n_frames x 1
            spk_id: B x 1
        '''
        # combtooth exciter signal
        combtooth = self.fast_source_gen(f0_frames)
        combtooth_frames = combtooth.unfold(1, self.block_size, self.block_size)

        # noise exciter signal
        noise = torch.randn_like(combtooth)
        noise_frames = noise.unfold(1, self.block_size, self.block_size)

        # parameter prediction
        ctrls, hidden = self.unit2ctrl(units_frames, combtooth_frames, noise_frames, volume_frames, spk_id=spk_id, spk_mix_dict=spk_mix_dict, aug_shift=aug_shift)

        src_filter = torch.exp(ctrls['harmonic_magnitude'] + 1.j * np.pi * ctrls['harmonic_phase'])
        src_filter = torch.cat((src_filter, src_filter[:,-1:,:]), 1)
        noise_filter= torch.exp(ctrls['noise_magnitude'] + 1.j * np.pi * ctrls['noise_phase']) / 128
        noise_filter = torch.cat((noise_filter, noise_filter[:,-1:,:]), 1)

        # harmonic part filter
        if combtooth.shape[-1] > self.win_length // 2:
            pad_mode = 'reflect'
        else:
            pad_mode = 'constant'
        combtooth_stft = torch.stft(
                            combtooth,
                            n_fft = self.win_length,
                            win_length = self.win_length,
                            hop_length = self.block_size,
                            window = self.window,
                            center = True,
                            return_complex = True,
                            pad_mode = pad_mode)

        # noise part filter
        noise_stft = torch.stft(
                            noise,
                            n_fft = self.win_length,
                            win_length = self.win_length,
                            hop_length = self.block_size,
                            window = self.window,
                            center = True,
                            return_complex = True,
                            pad_mode = pad_mode)

        # apply the filters
        signal_stft = combtooth_stft * src_filter.permute(0, 2, 1) + noise_stft * noise_filter.permute(0, 2, 1)

        # take the istft to resynthesize audio.
        signal = torch.istft(
                        signal_stft,
                        n_fft = self.win_length,
                        win_length = self.win_length,
                        hop_length = self.block_size,
                        window = self.window,
                        center = True)

        return signal, hidden