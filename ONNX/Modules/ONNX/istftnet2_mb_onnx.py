"""
iSTFTNet2-MB ONNX Conversion Support
ONNX-compatible implementation with custom STFT operations for iSTFTNet2 Multi-Band
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
import numpy as np
from scipy.signal import get_window


class STFTONNX2D(torch.nn.Module):
    """Enhanced STFT ONNX module for iSTFTNet2-MB with multi-band support"""
    
    def __init__(self, model_type, n_fft, n_mels, hop_len, max_frames, window_type, pad_mode, num_bands=4):
        super(STFTONNX2D, self).__init__()
        self.model_type = model_type
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_len = hop_len
        self.max_frames = max_frames
        self.window_type = window_type
        self.half_n_fft = self.n_fft // 2
        self.pad_mode = pad_mode
        self.num_bands = num_bands
        
        # Get window function
        window = {
            'bartlett': torch.bartlett_window,
            'blackman': torch.blackman_window,
            'hamming': torch.hamming_window,
            'hann': torch.hann_window,
            'kaiser': lambda x: torch.kaiser_window(x, periodic=True, beta=12.0)
        }.get(self.window_type, torch.hann_window)(self.n_fft).float()
        
        # Multi-band frequency splitting
        freq_per_band = (self.half_n_fft + 1) // self.num_bands
        self.band_indices = []
        for i in range(self.num_bands):
            start_idx = i * freq_per_band
            end_idx = min((i + 1) * freq_per_band, self.half_n_fft + 1)
            self.band_indices.append((start_idx, end_idx))
        
        if self.model_type in ['stft_A', 'stft_B']:
            time_steps = torch.arange(self.n_fft).unsqueeze(0).float()
            frequencies = torch.arange(self.half_n_fft + 1).unsqueeze(1).float()
            omega = 2 * torch.pi * frequencies * time_steps / self.n_fft
            window = window.unsqueeze(0)
            self.register_buffer('cos_kernel', (torch.cos(omega) * window).unsqueeze(1))
            self.register_buffer('sin_kernel', (-torch.sin(omega) * window).unsqueeze(1))
            self.padding_zero = torch.zeros((1, 1, self.half_n_fft), dtype=torch.float32)
            
            # Multi-band kernels
            self.register_buffer('band_cos_kernels', self._create_band_kernels(self.cos_kernel))
            self.register_buffer('band_sin_kernels', self._create_band_kernels(self.sin_kernel))

        elif self.model_type in ['istft_A', 'istft_B', 'istft_MB']:
            fourier_basis = torch.fft.fft(torch.eye(self.n_fft, dtype=torch.float32))
            fourier_basis = torch.vstack([
                torch.real(fourier_basis[:self.half_n_fft + 1, :]),
                torch.imag(fourier_basis[:self.half_n_fft + 1, :])
            ]).float()
            forward_basis = window * fourier_basis[:, None, :]
            inverse_basis = window * torch.linalg.pinv((fourier_basis * self.n_fft) / self.hop_len).T[:, None, :]
            
            n = self.n_fft + self.hop_len * (self.max_frames - 1)
            window_sum = torch.zeros(n, dtype=torch.float32)
            window_normalized = window / window.abs().max()
            total_pad = self.n_fft - window_normalized.shape[0]
            pad_left = total_pad // 2
            pad_right = total_pad - pad_left
            win_sq = torch.nn.functional.pad(window_normalized ** 2, (pad_left, pad_right), mode='constant', value=0)

            for i in range(self.max_frames):
                sample = i * self.hop_len
                window_sum[sample: min(n, sample + self.n_fft)] += win_sq[: max(0, min(self.n_fft, n - sample))]
                
            self.register_buffer("forward_basis", forward_basis)
            self.register_buffer("inverse_basis", inverse_basis)
            self.register_buffer("window_sum_inv", self.n_fft / (window_sum * self.hop_len))
            
            # Multi-band inverse basis
            if self.model_type == 'istft_MB':
                self.register_buffer("mb_inverse_basis", self._create_multiband_inverse_basis(inverse_basis))

    def _create_band_kernels(self, kernel):
        """Create kernels for each frequency band"""
        band_kernels = []
        for start_idx, end_idx in self.band_indices:
            band_kernel = kernel[start_idx:end_idx, :, :]
            band_kernels.append(band_kernel)
        return torch.stack(band_kernels, dim=0)
    
    def _create_multiband_inverse_basis(self, inverse_basis):
        """Create multi-band inverse basis for iSTFT"""
        mb_basis = []
        for start_idx, end_idx in self.band_indices:
            # Real and imaginary parts for this band
            real_basis = inverse_basis[start_idx:end_idx, :, :]
            imag_basis = inverse_basis[self.half_n_fft + 1 + start_idx:self.half_n_fft + 1 + end_idx, :, :]
            band_basis = torch.cat([real_basis, imag_basis], dim=0)
            mb_basis.append(band_basis)
        return torch.stack(mb_basis, dim=0)

    def forward(self, *args):
        if self.model_type == 'stft_A':
            return self.stft_A_forward(*args)
        elif self.model_type == 'stft_B':
            return self.stft_B_forward(*args)
        elif self.model_type == 'istft_A':
            return self.istft_A_forward(*args)
        elif self.model_type == 'istft_B':
            return self.istft_B_forward(*args)
        elif self.model_type == 'istft_MB':
            return self.istft_MB_forward(*args)

    def stft_A_forward(self, x):
        """STFT forward with single output"""
        if self.pad_mode == 'reflect':
            x = torch.nn.functional.pad(x, (self.half_n_fft, self.half_n_fft), mode=self.pad_mode)
        else:
            x = torch.cat((self.padding_zero, x, self.padding_zero), dim=-1)
        real_part = torch.nn.functional.conv1d(x, self.cos_kernel, stride=self.hop_len)
        return real_part

    def stft_B_forward(self, x):
        """STFT forward with real and imaginary outputs"""
        if self.pad_mode == 'reflect':
            x = torch.nn.functional.pad(x, (self.half_n_fft, self.half_n_fft), mode=self.pad_mode)
        else:
            x = torch.cat((self.padding_zero, x, self.padding_zero), dim=-1)
        real_part = torch.nn.functional.conv1d(x, self.cos_kernel, stride=self.hop_len)
        imag_part = torch.nn.functional.conv1d(x, self.sin_kernel, stride=self.hop_len)
        return real_part, imag_part

    def istft_A_forward(self, magnitude, phase):
        """Standard iSTFT forward"""
        inverse_transform = torch.nn.functional.conv_transpose1d(
            torch.cat((magnitude * torch.cos(phase), magnitude * torch.sin(phase)), dim=1),
            self.inverse_basis,
            stride=self.hop_len,
            padding=0,
        )
        output = inverse_transform[:, :, self.half_n_fft: -self.half_n_fft] * self.window_sum_inv[self.half_n_fft: inverse_transform.size(-1) - self.half_n_fft]
        return output

    def istft_B_forward(self, magnitude, real, imag):
        """iSTFT forward with separate real/imag inputs"""
        phase = torch.atan2(imag, real)
        inverse_transform = torch.nn.functional.conv_transpose1d(
            torch.cat((magnitude * torch.cos(phase), magnitude * torch.sin(phase)), dim=1),
            self.inverse_basis,
            stride=self.hop_len,
            padding=0,
        )
        output = inverse_transform[:, :, self.half_n_fft: -self.half_n_fft] * self.window_sum_inv[self.half_n_fft: inverse_transform.size(-1) - self.half_n_fft]
        return output
    
    def istft_MB_forward(self, band_magnitudes, band_phases):
        """Multi-band iSTFT forward - key for iSTFTNet2-MB"""
        # band_magnitudes: [B, num_bands, freq_per_band, T]
        # band_phases: [B, num_bands, freq_per_band, T]
        
        batch_size = band_magnitudes.size(0)
        time_frames = band_magnitudes.size(-1)
        
        # Process each band separately
        band_outputs = []
        for band_idx in range(self.num_bands):
            mag_band = band_magnitudes[:, band_idx, :, :]  # [B, freq_per_band, T]
            phase_band = band_phases[:, band_idx, :, :]    # [B, freq_per_band, T]
            
            # Get band-specific inverse basis
            band_basis = self.mb_inverse_basis[band_idx]  # [2*freq_per_band, 1, n_fft]
            
            # Convert to complex representation
            real_part = mag_band * torch.cos(phase_band)
            imag_part = mag_band * torch.sin(phase_band)
            complex_input = torch.cat([real_part, imag_part], dim=1)  # [B, 2*freq_per_band, T]
            
            # Apply inverse transform for this band
            band_output = torch.nn.functional.conv_transpose1d(
                complex_input,
                band_basis,
                stride=self.hop_len,
                padding=0,
            )
            band_outputs.append(band_output)
        
        # Sum all band outputs
        output = sum(band_outputs)
        
        # Apply window normalization
        output = output[:, :, self.half_n_fft: -self.half_n_fft] * self.window_sum_inv[self.half_n_fft: output.size(-1) - self.half_n_fft]
        return output


class TorchSTFT2D(torch.nn.Module):
    """Enhanced STFT module for iSTFTNet2-MB with ONNX support"""
    
    def __init__(self, filter_length=20, hop_length=5, win_length=20, window='hann', num_bands=4):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.num_bands = num_bands
        self.window = torch.from_numpy(get_window(window, win_length, fftbins=True).astype(np.float32))
        
        # Standard iSTFT for non-ONNX mode
        self.istft_onnx = STFTONNX2D(
            model_type='istft_A', 
            n_fft=filter_length, 
            n_mels=80, 
            hop_len=hop_length, 
            max_frames=100096, 
            window_type=torch.hann_window, 
            pad_mode='reflect'
        ).eval()
        
        # Multi-band iSTFT for ONNX mode
        self.istft_mb_onnx = STFTONNX2D(
            model_type='istft_MB', 
            n_fft=filter_length, 
            n_mels=80, 
            hop_len=hop_length, 
            max_frames=100096, 
            window_type=torch.hann_window, 
            pad_mode='reflect',
            num_bands=num_bands
        ).eval()

    def transform(self, input_data):
        """Transform audio to magnitude and phase"""
        forward_transform = torch.stft(
            input_data,
            self.filter_length, self.hop_length, self.win_length, 
            window=self.window.to(input_data.device),
            return_complex=False if torch.onnx.is_in_onnx_export() else True
        )

        if torch.onnx.is_in_onnx_export():
            real = forward_transform[..., 0]
            imag = forward_transform[..., 1]
            return torch.abs(real), torch.atan2(imag, real)
        else:
            return torch.abs(forward_transform), torch.angle(forward_transform)

    def inverse(self, magnitude, phase):
        """Standard inverse transform"""
        if torch.onnx.is_in_onnx_export():
            inverse_transform = self.istft_onnx(magnitude, phase)
        else:
            inverse_transform = torch.istft(
                magnitude * torch.exp(phase * 1j),
                self.filter_length, self.hop_length, self.win_length, 
                window=self.window.to(magnitude.device)
            )
        return inverse_transform.unsqueeze(-2)

    def inverse_multiband(self, band_magnitudes, band_phases):
        """Multi-band inverse transform for iSTFTNet2-MB"""
        if torch.onnx.is_in_onnx_export():
            # Use ONNX-compatible multi-band iSTFT
            inverse_transform = self.istft_mb_onnx(band_magnitudes, band_phases)
        else:
            # Non-ONNX mode: reconstruct full spectrum and use standard iSTFT
            batch_size, num_bands, freq_per_band, time_frames = band_magnitudes.shape
            
            # Reconstruct full magnitude and phase
            full_magnitude = torch.cat([
                band_magnitudes[:, i, :, :] for i in range(num_bands)
            ], dim=1)
            full_phase = torch.cat([
                band_phases[:, i, :, :] for i in range(num_bands)
            ], dim=1)
            
            # Standard iSTFT
            inverse_transform = torch.istft(
                full_magnitude * torch.exp(full_phase * 1j),
                self.filter_length, self.hop_length, self.win_length,
                window=self.window.to(band_magnitudes.device)
            )
        return inverse_transform.unsqueeze(-2)

    def forward(self, input_data):
        """Standard forward pass"""
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


class Conv1DTo2DONNX(nn.Module):
    """ONNX-compatible 1D to 2D conversion"""
    
    def __init__(self, in_channels, out_channels, freq_dim=16):
        super().__init__()
        self.freq_dim = freq_dim
        self.conv = weight_norm(nn.Conv1d(in_channels, out_channels * freq_dim, 1))

    def forward(self, x):
        # x: [B, C, T]
        x = self.conv(x)  # [B, C*F, T]
        B, CF, T = x.shape
        C = CF // self.freq_dim
        
        # ONNX-compatible reshape
        x = x.view(B, C, self.freq_dim, T)  # [B, C, F, T]
        return x


class Conv2DTo1DONNX(nn.Module):
    """ONNX-compatible 2D to 1D conversion"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = weight_norm(nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, x):
        # x: [B, C, F, T]
        x = self.conv(x)  # [B, C_out, F, T]
        B, C, F, T = x.shape
        
        # ONNX-compatible reshape
        x = x.view(B, C * F, T)  # [B, C*F, T]
        return x


class MultiBandSplitONNX(nn.Module):
    """ONNX-compatible multi-band splitting"""
    
    def __init__(self, channels, num_bands=4):
        super().__init__()
        self.num_bands = num_bands
        self.band_convs = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels // num_bands, 1))
            for _ in range(num_bands)
        ])

    def forward(self, x):
        # x: [B, C, T] where C = gen_istft_n_fft + 2
        band_outputs = []
        for band_conv in self.band_convs:
            band_output = band_conv(x)
            band_outputs.append(band_output)
        
        # Stack bands: [B, num_bands, C//num_bands, T]
        return torch.stack(band_outputs, dim=1)


class MultiBandCombineONNX(nn.Module):
    """ONNX-compatible multi-band combination"""
    
    def __init__(self, band_channels, output_channels, num_bands=4):
        super().__init__()
        self.num_bands = num_bands
        self.combine_conv = weight_norm(nn.Conv1d(band_channels * num_bands, output_channels, 1))

    def forward(self, band_outputs):
        # band_outputs: [B, num_bands, C_band, T]
        B, num_bands, C_band, T = band_outputs.shape
        
        # Reshape for concatenation: [B, num_bands * C_band, T]
        combined = band_outputs.view(B, num_bands * C_band, T)
        
        # Final combination
        output = self.combine_conv(combined)
        return output


def convert_istftnet2mb_to_onnx(model, onnx_path, example_inputs, opset_version=11):
    """
    Convert iSTFTNet2-MB model to ONNX format
    
    Args:
        model: iSTFTNet2MB model instance
        onnx_path: Output path for ONNX model
        example_inputs: Tuple of example inputs (x, s) for tracing
        opset_version: ONNX opset version
    """
    
    # Set model to evaluation mode
    model.eval()
    
    # Remove weight normalization for ONNX compatibility
    model.remove_weight_norm()
    
    # Define input names and output names
    input_names = ['mel_spectrogram', 'style_vector']
    output_names = ['audio_output']
    
    # Dynamic axes for variable length inputs
    dynamic_axes = {
        'mel_spectrogram': {0: 'batch_size', 2: 'time_steps'},
        'style_vector': {0: 'batch_size'},
        'audio_output': {0: 'batch_size', 2: 'audio_length'}
    }
    
    # Export to ONNX
    torch.onnx.export(
        model,
        example_inputs,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=True
    )
    
    print(f"Successfully exported iSTFTNet2-MB to ONNX: {onnx_path}")


def verify_onnx_model(onnx_path, example_inputs):
    """
    Verify the exported ONNX model
    
    Args:
        onnx_path: Path to the ONNX model
        example_inputs: Example inputs for verification
    """
    import onnx
    import onnxruntime as ort
    
    # Load and check the ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed!")
    
    # Test with ONNX Runtime
    ort_session = ort.InferenceSession(onnx_path)
    
    # Prepare inputs for ONNX Runtime
    x, s = example_inputs
    ort_inputs = {
        'mel_spectrogram': x.numpy(),
        'style_vector': s.numpy()
    }
    
    # Run inference
    ort_outputs = ort_session.run(None, ort_inputs)
    print(f"ONNX Runtime inference successful! Output shape: {ort_outputs[0].shape}")
    
    return ort_outputs


# Example usage
if __name__ == "__main__":
    # Example of how to use the ONNX conversion
    from Modules.istftnet2_mb import iSTFTNet2MB
    
    # Create model
    model = iSTFTNet2MB(
        dim_in=512,
        style_dim=128,
        num_bands=4,
        gen_istft_n_fft=20,
        gen_istft_hop_size=5
    )
    
    # Create example inputs
    batch_size = 1
    seq_len = 100
    x = torch.randn(batch_size, 512, seq_len)
    s = torch.randn(batch_size, 128)
    example_inputs = (x, s)
    
    # Convert to ONNX
    convert_istftnet2mb_to_onnx(
        model, 
        "istftnet2_mb.onnx", 
        example_inputs,
        opset_version=11
    )
    
    # Verify the ONNX model
    verify_onnx_model("istftnet2_mb.onnx", example_inputs)