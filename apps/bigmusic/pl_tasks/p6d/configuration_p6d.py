"""P6 Dense model configuration"""

from seed_models.utils import logging
from seed_models.utils.modeling_rope_utils import rope_config_validation
from transformers.configuration_utils import PretrainedConfig

logger = logging.get_logger(__name__)


class P6DenseConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`P6DenseModel`]. It is used to instantiate an LLaMA
    model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the P6 dense model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`P6DenseModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 14336, maps 'n_inner' in mariana):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32, maps 'n_layer' in mariana):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32, maps 'n_head' in mariana):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 8, maps 'n_head / n_shared_qhead' in mariana):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        head_dim (`int`, *optional*, defaults to hidden_size // num_attention_heads):
            The attention head dimension.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3', 'ntk'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.
        use_qk_rmsnorm (`bool`, *optional*, defaults to `False`):
            Whether to apply rmsnorm after qk proj, this is required by qwen3
        _flash_attn_kernel_implementation (`str`, *optional*, defaults to `"default"`):
            The implementation to use for the flash attention kernel. Can be one of [`"default"`, `"lego"`, `"fa3"`].
            `"default"` is the default flash attn 2 implementation.
            `"lego"` will use the in-house lego implementation.
            `"fa3"` will use the official flash-attention-v3 implementation.

    ```python
    >>> from transformers import P6DenseModel, P6DenseConfig

    >>> # Initializing a P6 configuration
    >>> configuration = P6DenseConfig()

    >>> # Initializing a model from the P6 dense configuration
    >>> model = P6DenseModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "seed_p6dense"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        query_head_scale_factor=1,
        num_key_value_heads=None,
        head_dim=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        layer_norm_eps=None,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        resid_pdrop=0.0,
        use_qk_rmsnorm=False,
        _flash_attn_kernel_implementation="default",
        _rms_norm_implementation="torch",
        _rope_implementation="torch",
        _swiglu_kernel_implementation="torch",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.query_head_scale_factor = query_head_scale_factor

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        if head_dim is None:
            if hidden_size % num_attention_heads != 0:
                raise ValueError(
                    f"hidden_size must be divisible by num_heads (got `hidden_size`: {hidden_size}"
                    f" and `num_heads`: {num_attention_heads})."
                )
            head_dim = hidden_size // num_attention_heads

        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.resid_pdrop = resid_pdrop
        self.mlp_bias = mlp_bias
        self.use_qk_rmsnorm = use_qk_rmsnorm
        self.attention_out_bias = kwargs.get("attention_out_bias", attention_bias)

        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        # flash attention kernel implementation
        self._flash_attn_kernel_implementation = _flash_attn_kernel_implementation
        # rmsnorm kernel implementation
        self._rms_norm_implementation = _rms_norm_implementation
        # rope kernel implementation
        self._rope_implementation = _rope_implementation
        # swiglu kernel implementation
        self._swiglu_kernel_implementation = _swiglu_kernel_implementation

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @classmethod
    def from_network_config(cls, network, **kwargs):
        '''Generate a model config from the network config generated by CruiseCLI.

        Introduces for compatiability purpose.

        Args:
            network(CruiseConfig): cruise Config

        Return: P6DenseConfig
        '''

        num_key_value_heads = network.n_head / network.n_shared_qhead
        assert num_key_value_heads.is_integer(), f"num_key_value_heads {num_key_value_heads} is not an integer"
        num_key_value_heads = int(num_key_value_heads)

        config = cls(
            vocab_size=network.vocab_size,
            hidden_size=network.hidden_size,
            intermediate_size=network.n_inner,
            num_hidden_layers=network.num_hidden_layers,
            num_attention_heads=network.n_head,
            query_head_scale_factor=network.query_head_scale_factor,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=network.max_position_embeddings,
            initializer_range=network.initializer_range,
            use_qk_rmsnorm=network.use_qk_rmsnorm,
            rms_norm_eps=network.rms_norm_eps,
            layer_norm_eps=network.layer_norm_eps,
            pad_token_id=network.pad_idx,
            tie_word_embeddings=network.tie_weight,
            rope_theta=network.rope_base,
            rope_scaling={
                "rope_type": network.rope_mode,
                "factor": network.rope_scale,
                "rope_cut": network.rope_cut,
                "rope_cut_head_dim": network.rope_cut_head_dim,
            },
            attention_bias=network.use_attention_bias,
            attention_dropout=network.attn_pdrop,
            attention_out_bias=network.attention_out_bias,
            resid_pdrop=network.resid_pdrop,
            hidden_act=network.activation_function,
            _flash_attn_kernel_implementation=network._flash_attn_kernel_implementation,
            _rms_norm_implementation=network._rms_norm_implementation,
            _rope_implementation=network._rope_implementation,
            mlp_bias=network.mlp_bias,
            _swiglu_kernel_implementation=network._swiglu_kernel_implementation,
            **kwargs,
        )

        return config
