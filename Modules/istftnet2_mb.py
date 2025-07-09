"""
iSTFTNet2-MB: Multi-Band iSTFTNet2 Implementation
Drop-in replacement for StyleTTS2-lite's iSTFTNet module

Based on: "iSTFTNet2: Faster and More Lightweight iSTFT-Based Neural Vocoder Using 1D-2D CNN"
Paper: https://arxiv.org/pdf/2308.07117

Key improvements over original iSTFTNet:
1. 1D-2D CNN hybrid architecture 
2. Multi-band processing for enhanced efficiency
3. Reduced temporal upsampling (8x instead of 64x)
4. Better spectrogram modeling with 2D convolutions
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm
from typing import List, Optional, Tuple


def init_weights(m, mean=0.0, std=0.01):
    """Initialize conv layer weights"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    """Calculate padding for conv layers"""
    return int((kernel_size * dilation - dilation) / 2)


class AdaIN1d(nn.Module):
    """Adaptive Instance Normalization for style conditioning"""
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdaIN2d(nn.Module):
    """Adaptive Instance Normalization for 2D convolutions"""
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class ResBlock1D(nn.Module):
    """1D Residual block with style conditioning"""
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), style_dim=128):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, 
                                dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, 
                                dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, 
                                dilation=dilation[2], padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, 
                                dilation=1, padding=get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, 
                                dilation=1, padding=get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, 
                                dilation=1, padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

        self.adain1 = nn.ModuleList([
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels)
        ])

        self.adain2 = nn.ModuleList([
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels)
        ])

    def forward(self, x, s):
        for c1, c2, n1, n2 in zip(self.convs1, self.convs2, self.adain1, self.adain2):
            xt = F.leaky_relu(n1(x, s))
            xt = c1(xt)
            xt = F.leaky_relu(n2(xt, s))
            xt = c2(xt)
            x = xt + x
        return x


class ResBlock2D(nn.Module):
    """2D Residual block for frequency upsampling"""
    def __init__(self, channels, kernel_size=(3, 3), style_dim=128):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv2d(channels, channels, kernel_size, 
                                         padding=(kernel_size[0]//2, kernel_size[1]//2)))
        self.conv2 = weight_norm(nn.Conv2d(channels, channels, kernel_size, 
                                         padding=(kernel_size[0]//2, kernel_size[1]//2)))
        self.adain1 = AdaIN2d(style_dim, channels)
        self.adain2 = AdaIN2d(style_dim, channels)

    def forward(self, x, s):
        xt = F.leaky_relu(self.adain1(x, s))
        xt = self.conv1(xt)
        xt = F.leaky_relu(self.adain2(xt, s))
        xt = self.conv2(xt)
        return xt + x


class UpSample1D(nn.Module):
    """1D Upsampling block"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, style_dim=128):
        super().__init__()
        self.conv = weight_norm(nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride,
                                                 padding=(kernel_size - stride) // 2))
        self.adain = AdaIN1d(style_dim, out_channels)

    def forward(self, x, s):
        x = self.conv(x)
        x = self.adain(x, s)
        return F.leaky_relu(x)


class UpSample2D(nn.Module):
    """2D Upsampling block for frequency dimension"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, style_dim=128):
        super().__init__()
        self.conv = weight_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                                                 padding=((kernel_size[0] - stride[0]) // 2,
                                                         (kernel_size[1] - stride[1]) // 2)))
        self.adain = AdaIN2d(style_dim, out_channels)

    def forward(self, x, s):
        x = self.conv(x)
        x = self.adain(x, s)
        return F.leaky_relu(x)


class Conv1DTo2D(nn.Module):
    """Convert 1D features to 2D representation"""
    def __init__(self, in_channels, out_channels, freq_dim=32):
        super().__init__()
        self.freq_dim = freq_dim
        self.conv = weight_norm(nn.Conv1d(in_channels, out_channels * freq_dim, 1))

    def forward(self, x):
        # x: [B, C, T]
        x = self.conv(x)  # [B, C*F, T]
        B, CF, T = x.shape
        C = CF // self.freq_dim
        x = x.view(B, C, self.freq_dim, T)  # [B, C, F, T]
        return x


class Conv2DTo1D(nn.Module):
    """Convert 2D features back to 1D"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = weight_norm(nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, x):
        # x: [B, C, F, T]
        x = self.conv(x)  # [B, C_out, F, T]
        B, C, F, T = x.shape
        x = x.view(B, C * F, T)  # [B, C*F, T]
        return x


class MultiScaleSTFT(nn.Module):
    """Multi-scale STFT for multi-band processing"""
    def __init__(self, n_fft_list=[512, 1024, 2048], hop_length_list=[128, 256, 512]):
        super().__init__()
        self.n_fft_list = n_fft_list
        self.hop_length_list = hop_length_list

    def forward(self, x):
        # Returns magnitude spectrograms at multiple scales
        spectrograms = []
        for n_fft, hop_length in zip(self.n_fft_list, self.hop_length_list):
            spec = torch.stft(x.squeeze(1), n_fft=n_fft, hop_length=hop_length,
                            window=torch.hann_window(n_fft).to(x.device),
                            return_complex=True)
            spec_mag = torch.abs(spec)
            spectrograms.append(spec_mag)
        return spectrograms


class iSTFTNet2MB(nn.Module):
    """
    iSTFTNet2 Multi-Band (iSTFTNet2-MB) Decoder
    
    Drop-in replacement for StyleTTS2-lite's iSTFTNet module with:
    - 1D-2D CNN hybrid architecture
    - Multi-band processing  
    - Reduced temporal upsampling
    - Enhanced frequency modeling with 2D convolutions
    """
    
    def __init__(self, 
                 dim_in=512,
                 style_dim=128,
                 dim_out=80,
                 resblock_kernel_sizes=[3, 7, 11],
                 upsample_rates=[8, 8, 2, 2],  # Reduced from [8, 8, 4, 2] 
                 upsample_initial_channel=512,
                 resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                 upsample_kernel_sizes=[16, 16, 4, 4],
                 gen_istft_n_fft=16,
                 gen_istft_hop_size=4,
                 num_bands=4):  # Multi-band processing
        
        super().__init__()
        
        self.num_bands = num_bands
        self.gen_istft_n_fft = gen_istft_n_fft
        self.gen_istft_hop_size = gen_istft_hop_size
        
        # Input projection
        self.input_conv = weight_norm(nn.Conv1d(dim_in, upsample_initial_channel, 7, 1, 3))
        
        # 1D upsampling layers (reduced temporal upsampling)
        self.ups_1d = nn.ModuleList()
        self.resblocks_1d = nn.ModuleList()
        
        # First stage: 1D processing with reduced upsampling
        for i, (u, k) in enumerate(zip(upsample_rates[:2], upsample_kernel_sizes[:2])):
            self.ups_1d.append(UpSample1D(upsample_initial_channel // (2**i),
                                        upsample_initial_channel // (2**(i+1)),
                                        k, u, style_dim))
            
            ch = upsample_initial_channel // (2**(i+1))
            for j, (k_r, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks_1d.append(ResBlock1D(ch, k_r, d, style_dim))
        
        # 1D to 2D conversion
        ch_1d = upsample_initial_channel // (2**len(upsample_rates[:2]))
        self.freq_dim = 32  # Frequency dimension for 2D processing
        self.conv_1d_to_2d = Conv1DTo2D(ch_1d, ch_1d // 2, self.freq_dim)
        
        # 2D processing for frequency upsampling
        self.ups_2d = nn.ModuleList()
        self.resblocks_2d = nn.ModuleList()
        
        ch_2d = ch_1d // 2
        for i, (u, k) in enumerate(zip(upsample_rates[2:], upsample_kernel_sizes[2:])):
            # Frequency upsampling
            self.ups_2d.append(UpSample2D(ch_2d // (2**i),
                                        ch_2d // (2**(i+1)),
                                        (3, k), (1, u), style_dim))
            
            ch = ch_2d // (2**(i+1))
            self.resblocks_2d.append(ResBlock2D(ch, (3, 3), style_dim))
        
        # 2D to 1D conversion  
        final_ch_2d = ch_2d // (2**len(upsample_rates[2:]))
        final_freq_dim = self.freq_dim * math.prod(upsample_rates[2:])
        self.conv_2d_to_1d = Conv2DTo1D(final_ch_2d, gen_istft_n_fft + 2)
        
        # Multi-band processing
        self.band_split = nn.ModuleList([
            weight_norm(nn.Conv1d(gen_istft_n_fft + 2, gen_istft_n_fft + 2, 1))
            for _ in range(num_bands)
        ])
        
        # Output projection for each band
        self.output_convs = nn.ModuleList([
            weight_norm(nn.Conv1d(gen_istft_n_fft + 2, gen_istft_n_fft + 2, 7, 1, 3))
            for _ in range(num_bands)
        ])
        
        # Multi-scale STFT for consistency
        self.multi_stft = MultiScaleSTFT()
        
        # Initialize weights
        self.apply(init_weights)

    def forward(self, x, s):
        """
        Args:
            x: Input features [B, dim_in, T]  
            s: Style vector [B, style_dim]
            
        Returns:
            audio: Generated audio waveform [B, 1, T*hop_size*prod(upsample_rates)]
        """
        # Input projection
        x = self.input_conv(x)
        
        # 1D upsampling stage
        for up, resblocks in zip(self.ups_1d, 
                               [self.resblocks_1d[i:i+len(resblock_kernel_sizes)] 
                                for i in range(0, len(self.resblocks_1d), len(resblock_kernel_sizes))]):
            x = up(x, s)
            xs = None
            for resblock in resblocks:
                if xs is None:
                    xs = resblock(x, s)
                else:
                    xs += resblock(x, s)
            x = xs / len(resblocks)
        
        # 1D to 2D conversion
        x = self.conv_1d_to_2d(x)  # [B, C, F, T]
        
        # 2D upsampling stage for frequency modeling
        for up, resblock in zip(self.ups_2d, self.resblocks_2d):
            x = up(x, s)
            x = resblock(x, s)
        
        # 2D to 1D conversion
        x = self.conv_2d_to_1d(x)  # [B, gen_istft_n_fft+2, T*upsampling_factor]
        
        # Multi-band processing
        band_outputs = []
        for i, (band_conv, output_conv) in enumerate(zip(self.band_split, self.output_convs)):
            band_x = band_conv(x)
            band_x = F.tanh(output_conv(band_x))
            band_outputs.append(band_x)
        
        # Combine bands
        x = sum(band_outputs) / len(band_outputs)
        
        # iSTFT processing
        spec, phase = x[:, :self.gen_istft_n_fft//2+1], x[:, self.gen_istft_n_fft//2+1:]
        
        # Convert to complex spectrogram
        spec = torch.exp(spec)
        phase = torch.sin(phase) + 1j * torch.cos(phase)
        complex_spec = spec * phase
        
        # iSTFT to audio
        audio = torch.istft(complex_spec, 
                          n_fft=self.gen_istft_n_fft,
                          hop_length=self.gen_istft_hop_size,
                          window=torch.hann_window(self.gen_istft_n_fft).to(x.device),
                          center=True)
        
        audio = audio.unsqueeze(1)  # [B, 1, T]
        
        return audio
    
    def remove_weight_norm(self):
        """Remove weight normalization for inference optimization"""
        def _remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:
                return
                
        self.apply(_remove_weight_norm)


# Alias for backward compatibility with StyleTTS2-lite
Decoder = iSTFTNet2MB


def test_istftnet2mb():
    """Test function to verify the implementation"""
    # Test parameters matching StyleTTS2-lite interface
    model = iSTFTNet2MB(
        dim_in=512,
        style_dim=128, 
        dim_out=80,
        resblock_kernel_sizes=[3, 7, 11],
        upsample_rates=[8, 8, 2, 2],
        upsample_initial_channel=512,
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_kernel_sizes=[16, 16, 4, 4],
        gen_istft_n_fft=16,
        gen_istft_hop_size=4,
        num_bands=4
    )
    
    # Test forward pass
    batch_size = 2
    seq_len = 100
    x = torch.randn(batch_size, 512, seq_len)
    s = torch.randn(batch_size, 128)
    
    with torch.no_grad():
        output = model(x, s)
    
    print(f"Input shape: {x.shape}")
    print(f"Style shape: {s.shape}")  
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


if __name__ == "__main__":
    model = test_istftnet2mb()
    print("iSTFTNet2-MB test completed successfully!")
