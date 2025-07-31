from transformers.configuration_utils import PretrainedConfig


class UMMConfig(PretrainedConfig):
    model_type = "UMM"

    def __init__(
        self,
        # audio encoder
        num_channels=1,
        sample_rate=24000,
        frame_rate=25,
        n_fft=2048,
        win_length=2048,
        hop_length=240,
        n_mels=128,
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
        downsample_rate=4,
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
        vq_decay=0.99,
        vq_distance_type="euclidean", #["cosine", "euclidean", "dot_product"]
        w_loss_vq=1,
        fix_layers=False,
        rvq=1,
        stale_tolerance=100,
        # aux loss
        use_consistency_loss=False,
        consistency_loss_weight=0.1,
        consistency_chunk_ratio=0.2,

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
        rope_enhance_pos=15000,
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
        # for 25Hz or higher frame rates,
        upsample_strides=[1, 2, 2],
        upsample_pads=[3, 2, 2],
        downsample_strides=[2, 2],  
        # for conformer layer architecture
        window_attention=False, 
        window_left=0,
        window_right=0,
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
        head_output_norm=False, 
        ctc_downsample=False,
        ctc_downsample_rate=4,
        ctc_downsample_type="frame_concat",
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
        self.frame_rate = frame_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
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
        self.downsample_rate = downsample_rate

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
        self.vq_decay = vq_decay
        self.vq_distance_type = vq_distance_type
        self.w_loss_vq = w_loss_vq
        self.fix_layers = fix_layers
        self.rvq = rvq
        self.stale_tolerance = stale_tolerance
        self.use_consistency_loss = use_consistency_loss
        self.consistency_loss_weight = consistency_loss_weight
        self.consistency_chunk_ratio = consistency_chunk_ratio

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
        self.rope_enhance_pos = rope_enhance_pos
        self.rotary_embedding_base = rotary_embedding_base
        self.conv_depthwise_kernel_size = conv_depthwise_kernel_size
        self.conformer_conv_dropout = conformer_conv_dropout
        self.use_bn = use_bn
        # for 25Hz or higher frame rates
        self.upsample_strides = upsample_strides
        self.downsample_strides = downsample_strides
        self.upsample_pads = upsample_pads
        # for conformer layer architecture
        self.window_attention = window_attention
        self.window_left = window_left
        self.window_right = window_right

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
        self.head_output_norm = head_output_norm
        self.ctc_downsample = ctc_downsample
        self.ctc_downsample_rate = ctc_downsample_rate
        self.ctc_downsample_type = ctc_downsample_type

        self.add_chroma = add_chroma
        self.mix_prob = mix_prob
        self.interfere_audio = interfere_audio

        self.use_fused_kernel = kwargs.get("use_fused_kernel", False)
        self.use_causal_conformer = self.use_fused_kernel and \
            kwargs.get("use_causal_conformer", False)
        self.conformer_mask_topology = kwargs.get("conformer_mask_topology", \
            f'[(1000000000000,8)]*{self.num_hidden_layers}')
        self.conformer_chunk_conv = kwargs.get('conformer_chunk_conv', False)
        self.enable_dyna_chunk = kwargs.get('enable_dyna_chunk', False)
        self.dyna_chunk_range = kwargs.get('dyna_chunk_range', "1, 16")
        self.conformer_decoder_idx = kwargs.get('conformer_decoder_idx', None)

    @property
    def rq_input_dim(self):
        return self.n_mels * pow(self.feature_encoder_kernel, 2)

    @property
    def len_masking_raw(self):  
        return int(self.sample_rate * self.mask_hop)

    @property
    def len_masking_token(self):
        return int(self.len_masking_raw / self.hop_length / self.downsample_rate)

    @property
    def len_masking_mel(self):
        return int(self.len_masking_raw / self.hop_length)
        
    def get(self, name, default):
        if hasattr(self, name):
            return self.__getattribute__(name)
        else:
            return default
