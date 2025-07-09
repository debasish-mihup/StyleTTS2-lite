import torch
import librosa
import yaml
import torch
import torchaudio
from munch import Munch
from model_onnx import *

class TextCleaner:
    def __init__(self, symbol_dict, debug=True):
        self.word_index_dictionary = symbol_dict
        self.debug = debug
    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError as e:
                if self.debug:
                    print("\nWARNING UNKNOWN IPA CHARACTERS/LETTERS: ", char)
                    print("To ignore set 'debug' to false in the config")
                continue
        return indexes

class Preprocess():
    def __init__(self, config_path, models_path, device=torch.device('cpu')):
        self.device = device
        config = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
        params = config['model_params']
        try:
            symbols = (
                            list(config['symbol']['pad']) +
                            list(config['symbol']['punctuation']) +
                            list(config['symbol']['letters']) +
                            list(config['symbol']['letters_ipa']) +
                            list(config['symbol']['extend'])
                        )
            symbol_dict = {}
            for i in range(len((symbols))):
                symbol_dict[symbols[i]] = i
            
            n_token = len(symbol_dict) + 1
            print("\nFound:", n_token, "symbols")
        except Exception as e:
            print(f"\nERROR: Cannot find {e} in config file!\nYour config file is likely outdated, please download updated version from the repository.")
            raise SystemExit(1)
        
        self.cleaner       = TextCleaner(symbol_dict, debug=False)
        self.style_encoder = StyleEncoder(dim_in=params['dim_in'], 
                                          style_dim=params['style_dim'], 
                                          max_conv_dim=params['hidden_dim']).to(device)# acoustic style encoder
        self.__load_models(models_path)
    def __load_models(self, models_path):
        model = {'style_encoder':self.style_encoder}

        params_whole = torch.load(models_path, map_location='cpu')
        params = params_whole['net']
        params = {key: value for key, value in params.items() if key in model.keys()}

        for key in model:
            try:
                model[key].load_state_dict(params[key])
            except:
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                model[key].load_state_dict(new_state_dict, strict=False)
        _ = [model[key].eval() for key in model]

    def __wave_preprocess(self, wave):
        to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
        mean, std = -4, 4
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
        return mel_tensor
    
    def preprocess_input(self, phonem, speaker):
        #preprocess text
        tokens = self.cleaner(phonem)
        # tokens.insert(0, 0)
        # tokens.append(0)
        tokens = torch.LongTensor(tokens)
        #preprocess audio
        wave, sr = librosa.load(speaker['path'], sr=24000)
        audio, _ = librosa.effects.trim(wave, top_db=30)
        if sr != 24000: audio = librosa.resample(audio, sr, 24000)

        mel = self.__wave_preprocess(audio).unsqueeze(1)

        return tokens, mel, speaker['speed']
    
    def get_style(self, mel):
        return self.style_encoder(mel.to(self.device))

#For ONNX inference only
class StyleTTS2(torch.nn.Module):
    def __init__(self, config_path, models_path, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.ref_s = None
        config = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
        config['model_params']['dropout'] = 0.0
        
        try:
            symbols = (
                            list(config['symbol']['pad']) +
                            list(config['symbol']['punctuation']) +
                            list(config['symbol']['letters']) +
                            list(config['symbol']['letters_ipa']) +
                            list(config['symbol']['extend'])
                        )
            symbol_dict = {}
            for i in range(len((symbols))):
                symbol_dict[symbols[i]] = i
            
            n_token = len(symbol_dict) + 1
        except Exception as e:
            print(f"\nERROR: Cannot find {e} in config file!\nYour config file is likely outdated, please download updated version from the repository.")
            raise SystemExit(1)
        
        args = self.__recursive_munch(config['model_params'])
        args['n_token'] = n_token

        assert args.decoder.type in ['istftnet2_mb', 'istftnet', 'hifigan'], 'Decoder type unknown'

        if args.decoder.type == "istftnet2_mb":
            from Modules.ONNX.istftnet2_mb_onnx import Decoder
            
            # Get all configuration values with proper defaults
            decoder_config = args.decoder
            
            decoder = Decoder(
                # Core dimensions
                dim_in=getattr(args, 'hidden_dim', 512),
                style_dim=getattr(args, 'style_dim', 128), 
                dim_out=getattr(args, 'n_mels', 80),
                
                # 1D CNN backbone configuration
                resblock_kernel_sizes=getattr(decoder_config, 'resblock_kernel_sizes', [3, 7, 11]),
                upsample_rates=getattr(decoder_config, 'upsample_rates', [10, 6, 1, 1]),
                upsample_initial_channel=getattr(decoder_config, 'upsample_initial_channel', 512),
                resblock_dilation_sizes=getattr(decoder_config, 'resblock_dilation_sizes', 
                                              [[1, 3, 5], [1, 3, 5], [1, 3, 5]]),
                upsample_kernel_sizes=getattr(decoder_config, 'upsample_kernel_sizes', [20, 12, 3, 3]),
                
                # iSTFT configuration
                gen_istft_n_fft=getattr(decoder_config, 'gen_istft_n_fft', 20),
                gen_istft_hop_size=getattr(decoder_config, 'gen_istft_hop_size', 5),
                
                # Multi-band processing
                num_bands=getattr(decoder_config, 'num_bands', 4),
                
                # 2D CNN configuration (new parameters)
                freq_dim=getattr(decoder_config, 'freq_dim', 16),
                freq_upsample_rates=getattr(decoder_config, 'freq_upsample_rates', [2, 2]),
                use_shuffle_blocks=getattr(decoder_config, 'use_shuffle_blocks', True),
                num_2d_blocks=getattr(decoder_config, 'num_2d_blocks', 3),
            )
            
            # Apply inference optimizations if specified
            if hasattr(decoder_config, 'remove_weight_norm') and decoder_config.remove_weight_norm:
                decoder.remove_weight_norm()
        
        elif args.decoder.type == "istftnet":
            from Modules.ONNX.hifigan import Decoder
            self.decoder = Decoder(dim_in=args.hidden_dim, style_dim=args.style_dim, dim_out=args.n_mels,
                    resblock_kernel_sizes = args.decoder.resblock_kernel_sizes,
                    upsample_rates = args.decoder.upsample_rates,
                    upsample_initial_channel=args.decoder.upsample_initial_channel,
                    resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
                    upsample_kernel_sizes=args.decoder.upsample_kernel_sizes, 
                    gen_istft_n_fft=args.decoder.gen_istft_n_fft, gen_istft_hop_size=args.decoder.gen_istft_hop_size) 
        elif args.decoder.type == "hifigan":
            from Modules.ONNX.hifigan import Decoder
            self.decoder = Decoder(dim_in=args.hidden_dim, style_dim=args.style_dim, dim_out=args.n_mels,
                    resblock_kernel_sizes = args.decoder.resblock_kernel_sizes,
                    upsample_rates = args.decoder.upsample_rates,
                    upsample_initial_channel=args.decoder.upsample_initial_channel,
                    resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
                    upsample_kernel_sizes=args.decoder.upsample_kernel_sizes) 
            
        self.predictor           = ProsodyPredictor(style_dim=args.style_dim, d_hid=args.hidden_dim, nlayers=args.n_layer, max_dur=args.max_dur, dropout=args.dropout)
        self.text_encoder        = TextEncoder(channels=args.hidden_dim, kernel_size=5, depth=args.n_layer, n_symbols=args.n_token)

        self.__load_models(models_path)
        self.eval()
        self.to(self.device)
    
    def __onnx_alignment(self, pred_dur, max_length):
        seq_length = pred_dur.sum().to(dtype=torch.int64)
        device = pred_dur.device
                
        clipped_dur = torch.clamp(pred_dur, 0, seq_length)
        
        zero_tensor = torch.zeros(1, device=device)
        prefix_sum = torch.cat([zero_tensor, torch.cumsum(clipped_dur[:-1], dim=0)])
        
        positions = torch.arange(seq_length, device=device).unsqueeze(0).expand(max_length, -1)
        
        start_positions = prefix_sum.unsqueeze(1)
        
        end_positions = torch.min(
            start_positions + clipped_dur.unsqueeze(1),
            start_positions.new_full(start_positions.shape, seq_length, dtype=torch.float)
        )
        
        mask = (positions >= start_positions) & (positions < end_positions)

        return mask.float()
    
    def __recursive_munch(self, d):
        if isinstance(d, dict):
            return Munch((k, self.__recursive_munch(v)) for k, v in d.items())
        elif isinstance(d, list):
            return [self.__recursive_munch(v) for v in d]
        else:
            return d
        
    def __replace_outliers_zscore(self, tensor, threshold=3.0, factor=0.95):
        mean = tensor.mean()
        std = tensor.std()
        z = (tensor - mean) / std

        # Identify outliers
        outlier_mask = torch.abs(z) > threshold
        # Compute replacement value, respecting sign
        sign = torch.sign(tensor - mean)
        replacement = mean + sign * (threshold * std * factor)

        result = tensor.clone()
        result[outlier_mask] = replacement[outlier_mask]

        return result
        
    def __load_models(self, models_path):
        model = {'decoder':self.decoder, 'predictor':self.predictor, 'text_encoder':self.text_encoder}

        params_whole = torch.load(models_path, map_location='cpu')
        params = params_whole['net']
        params = {key: value for key, value in params.items() if key in model.keys()}

        for key in model:
            try:
                model[key].load_state_dict(params[key])
            except:
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                model[key].load_state_dict(new_state_dict, strict=False)
        _ = [model[key].eval() for key in model]
        
    def forward(self, tokens, style, speed=1):
        device = self.device
        tokens = tokens.unsqueeze(0).to(device)
        
        with torch.no_grad():
            input_lengths = tokens.new_full((tokens.shape[0],), tokens.shape[1], dtype=torch.long)
            mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
            text_mask = torch.gt(mask+1, input_lengths.unsqueeze(1)).to(tokens.device)  

            # encode
            t_en = self.text_encoder(tokens, input_lengths, text_mask)
            s = style.to(device)
        
            # cal alignment
            d = self.predictor.text_encoder(t_en, s, input_lengths, text_mask)
            self.predictor.lstm.flatten_parameters()
            x, _ = self.predictor.lstm(d)
            duration = self.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            duration[:,1:-2] = self.__replace_outliers_zscore(duration[:,1:-2]) #Normalize outlier
            duration /= speed

            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            if torch.onnx.is_in_onnx_export():
                pred_aln_trg = self.__onnx_alignment(pred_dur, input_lengths[0])
            else:
                pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
                c_frame = 0
                for i in range(pred_aln_trg.size(0)):
                    pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                    c_frame += int(pred_dur[i].data)
            alignment = pred_aln_trg.unsqueeze(0).to(device)

            # encode prosody
            en = (d.transpose(2, 1) @ alignment)
            F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
            asr = (t_en @ alignment)

            out = self.decoder(asr, F0_pred, N_pred, s)
            out = out.squeeze()[4000:-4000] #Remove weird pulse and silent tokens
        return out
