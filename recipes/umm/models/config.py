from transformers.configuration_utils import PretrainedConfig


class UMMConfig(PretrainedConfig):
    model_type = "UMM"

    def __init__(
        self,
        # audio encoder
        num_channels=128,
        sample_rate=24000,
        n_fft=2048,
        win_length=400,
        hop_length=240,
        feature_encoder_kernel=5,
        feature_encoder_padding=2,
        feature_cmvn=None,
        # random quantizer
        rq_codebook_size=4096,
        rq_codebook_dim=16,
        rq_codebook_num=8,
        rq_input_layernorm=False,
        mask_hop=0.4,
        mask_prob=0.2,
        # vector quantizer
        add_vq=False,
        use_ema_vq=True,
        vq_type='EMA', # EMA, EMAEntropy
        vq_layer_idx=12,
        vq_codebook_size=32768,
        vq_codebook_dim=256,
        vq_threshold_ema_dead_code=2,
        vq_kmeans_init=True,
        vq_kmeans_iters=10,
        vq_sync_codebook=True,
        vq_proj_norm=None,
        vq_proj_noise=0,
        w_loss_vq=1,
        fix_layers=False,
        # shared encoder
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=8,
        intermediate_size=4096,
        hidden_act="gelu",
        hidden_dropout=0.1,
        activation_dropout=0.1,
        attention_dropout=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rotary_embedding_base=10000,
        max_source_positions=750,
        conv_depthwise_kernel_size=31,
        conformer_conv_dropout=0.1,
        use_bn=True,
        # vocoder
        add_vocoder=False,
        upsample_rates=[5, 4, 4, 3, 2, 2],
        upsample_kernel_sizes=[15, 12, 12, 9, 6, 4],
        upsample_initial_channel=1536,
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        vocoder_activation="snakebeta",
        snake_logscale=True,
        w_loss_mel=1,
        w_loss_chroma=1,
        # CTC
        vocab_size=30522 + 1,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        ctc_blank_id=30522,
        w_loss_ctc=1,
        add_chroma=True,
        interfere_audio=False,
        mix_prob=0.2,
        usm_config=None,
        vocoder_config=None,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )

        # audio encoder
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.feature_encoder_kernel = feature_encoder_kernel
        self.feature_encoder_padding = feature_encoder_padding
        self.feature_cmvn = feature_cmvn
        assert feature_encoder_kernel == 5 and feature_encoder_padding == 2
        self.usm_config = usm_config
        self.vocoder_config = vocoder_config

        # random quantizer
        self.rq_codebook_size = rq_codebook_size
        self.rq_codebook_dim = rq_codebook_dim
        self.rq_codebook_num = rq_codebook_num
        self.rq_input_layernorm = rq_input_layernorm
        self.mask_hop = mask_hop
        self.mask_prob = mask_prob

        # vector quantizer
        self.add_vq = add_vq
        self.use_ema_vq = use_ema_vq
        self.vq_type = vq_type
        self.vq_layer_idx = vq_layer_idx
        self.vq_codebook_size = vq_codebook_size
        self.vq_codebook_dim = vq_codebook_dim
        self.vq_threshold_ema_dead_code = vq_threshold_ema_dead_code
        self.vq_kmeans_init = vq_kmeans_init
        self.vq_kmeans_iters = vq_kmeans_iters
        self.vq_sync_codebook = vq_sync_codebook
        self.vq_proj_norm = vq_proj_norm
        self.vq_proj_noise = vq_proj_noise
        self.w_loss_vq = w_loss_vq
        self.fix_layers = fix_layers

        # shared encoder
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.max_source_positions = max_source_positions
        self.rotary_embedding_base = rotary_embedding_base
        self.conv_depthwise_kernel_size = conv_depthwise_kernel_size
        self.conformer_conv_dropout = conformer_conv_dropout
        self.use_bn = use_bn

        # vocoder
        self.add_vocoder = add_vocoder
        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.upsample_initial_channel = upsample_initial_channel
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.vocoder_activation = vocoder_activation
        self.snake_logscale = snake_logscale
        self.w_loss_mel = w_loss_mel
        self.w_loss_chroma = w_loss_chroma

        # ctc loss
        self.vocab_size = vocab_size
        self.ctc_loss_reduction = ctc_loss_reduction
        self.ctc_zero_infinity = ctc_zero_infinity
        self.ctc_blank_id = ctc_blank_id
        self.w_loss_ctc = w_loss_ctc

        self.add_chroma = add_chroma
        self.mix_prob = mix_prob
        self.interfere_audio = interfere_audio

    @property
    def rq_input_dim(self):
        return self.num_channels * pow(self.feature_encoder_kernel, 2)

    @property
    def len_masking_raw(self):
        return int(self.sample_rate * self.mask_hop)

    @property
    def len_masking_token(self):
        return int(self.len_masking_raw / self.hop_length / 4)

    def get(self, name, default):
        if hasattr(self, name):
            return self.__getattribute__(name)
        else:
            return default


class UMMConfigLFR(UMMConfig):

    def __init__(self, 
                 token_frame_rate=20,
                 hop_length = 200, # 24000/200 = 120 Hz mel feature rate to support smooth downsampling to 20,15 and 10zh
                 **kwargs):
        '''
        @hanoihantrakul 4NOV2024
        Unlike the previous config which assumed 100Hz mel features downsampled to 25hz tokens,
        This config is configured for 120Hz mel features, which can be downsampled to 20, 15, 10 Hz tokens.

        Basically, the previous implementation had 2 layers of conv with kernel size 5.
        If you try to do 10x downsampling by doing stride=2 and then stride=5 in the space of 2 layers, you will be
        jumping over too many features in the second layer when going from 100hz -> 10hz. A stride of 5 and kernel size of 5
        means there is no overlap in the features, which will make the implementation not comparable to the previous tokenizer.

        Instead I did this by making the input feature 120Hz and then downsampling by strides 2, then 2 and then 3.
        120/2/2/3 = 10hz. This ensures we still have overlapping regions with kernel 5.
        '''
        assert token_frame_rate in [20, 15, 10] # the striding patterns only support these frame rates for ablation
        assert hop_length == 200 # 24000/200 = 120 Hz mel feature rate
        self.token_frame_rate = token_frame_rate
        super().__init__(hop_length=hop_length,
                         **kwargs)

        '''
        @hanoihantrakul 4NOV2024
        The previous umm_mkii.py hardcoded this relationship as 100hz mel features -> 100/2/2 = 25hz token rate. 
        In order to support different frame rates, I have changed the input mel features to be 120Hz, which can be divided easily into 20,15 and 10Hz frame rates.

        Basically, the previous implementation had 2 layers of conv with kernel size 5.
        If you try to do 10x downsampling by doing stride=2 and then stride=5, you will be
        jumping over too many features in the second layer when going from 100hz -> 10hz. A stride of 5 and kernel size of 5
        means there is no overlap in the features, which will make the implementation not comparable to the previous tokenizer.

        Instead I did this by making the input feature 120Hz and then downsampling by strides 2, then 2 and then 3.
        120/2/2/3 = 10hz. This ensures we still have overlapping regions with kernel 5.
        '''
        self.conv_striding_patterns = {
            20 : {'conv1': 1, 'conv2': 2, 'conv3': 3}, # 120 Hz Mel Frame rate -> 120/1/2/3 = 20 Hz token rate
            15 : {'conv1': 2, 'conv2': 2, 'conv3': 2}, # 120 Hz Mel Frame rate -> 120/2/2/2 = 15 Hz token rate
            10 : {'conv1': 2, 'conv2': 2, 'conv3': 3}, # 120 Hz Mel Frame rate -> 120/2/2/3 = 10 Hz token rate
        }
    
    def get_conv_downsampling_config(self):
        """Handles the striding pattern and linear projection config to go from 120 Hz mel feature rate to 20, 15 and 10hz frame rate."""
        conv_config_dict = self.conv_striding_patterns[self.token_frame_rate]
        striding2linear_units = {20: 11264, # Why 11264? -> 128/1/2/3 = 21.33 which becomes 22 after all the conv ops on mel-128. Then 22*512 (last layer channels) gives 11264 total dims to be projected down to output_dim=1024
                                 15: 8192, # Why 8192? -> 128/2/2/2 = 16 which becomes 16 after all the conv ops on mel-128. Then 16*512 (last layer channels) gives 8096 total dims to be projected down to output_dim=1024
                                 10: 5632} # Why 5632? -> 128/2/2/3 = 10.66 which becomes 11 after all the conv ops on mel-128. Then 11*512 (last layer channels) gives 5632 total dims to be projected down to output_dim=1024
        conv_config_dict['linear'] = striding2linear_units[self.token_frame_rate]
        return conv_config_dict

    def get_conv_upsampling_config(self):
        """Handles the striding pattern and linear projection to upsample back to 120hz mel feature rate from 20, 15 and 10hz token frame rate."""
        conv_config_dict = self.conv_striding_patterns[self.token_frame_rate]
        striding2linear_units = {20: 6149, # Why 6149? -> 1024*1*2*3 = 6144 which becomes 6149 after all ConvTranspose2d padding has been accounted for. This is then projected to output_dim
                                 15: 8192, # Why 8192? -> 1024*2*2*2 = 8192 which becomes 8192 after all ConvTranspose2d padding has been accounted for. This is then projected to output_dim
                                 10: 12287} # # Why 12287? -> 1024*2*2*3 = 12288 which becomes 12287 after all ConvTranspose2d padding has been accounted for. This is then projected to output_dim
        conv_config_dict['linear'] = striding2linear_units[self.token_frame_rate]
        return conv_config_dict
    
    def get_mel_mask_len_factor(self):
        '''
        @hanoihantrakul 4NOV2024
        The mel mask in Stage1 random projection is multiples of the hop length.
        '''
        return int(self.len_masking_raw / self.hop_length) 
    
    @property
    def rq_input_dim(self):
        '''
        @hanoihantrakul 4NOV2024
        This function configures random projection quantizer in Stage1. It is related to the number of conv layers
        used in the Conv2DSubsampling implementation of AudioEncoder().
        '''
        # return self.num_channels * pow(self.feature_encoder_kernel, 2) # I left this here to indicate how umm_mkii hard coded this to 2 conv layers
        return self.num_channels * pow(self.feature_encoder_kernel, 3) # my new implementation uses 3 conv layers to achieve smooth downsampling and upsampling
    
    @property
    def len_masking_token(self):
        '''
        @hanoihantrakul 4NOV2024
        This function configures the masking vector during Stage1 training. It is related
        to the striding patterns used in the Conv2DSubsampling implementation of AudioEncoder().
        '''
        conv_dict = self.conv_striding_patterns[self.token_frame_rate]
        downsampling_factor = conv_dict['conv1'] * conv_dict['conv2'] * conv_dict['conv3'] # my implementation assumes 3 conv layers for smoothly downsampling from 120hz
        return int(self.len_masking_raw / self.hop_length / downsampling_factor) # 20hz->downsampling_factor=6, 15hz->downsampling_factor=8, 10hz->downsampling_factor=12
        # This was previously hard coded as `return int(self.len_masking_raw / self.hop_length / 4)` because 25hz->downsampling_factor=4 