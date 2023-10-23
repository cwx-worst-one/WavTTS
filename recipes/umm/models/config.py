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
        ctc_zero_infinity=False,
        ctc_blank_id=30522,
        w_loss_ctc=1,
        add_chroma=True,
        interfere_audio=False,
        mix_prob=0.2,
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
