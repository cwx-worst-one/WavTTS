from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

from .configuration_p6d import P6DenseConfig
from .convert_p6d_checkpoint import convert_p6_dense_to_xperf_compatible  # noqa F401
from .convert_p6d_checkpoint import convert_p6dense_config_from_cruise  # noqa F401
from .convert_p6d_checkpoint import p6_dense_convert_pt_to_hf  # noqa F401
from .convert_p6d_checkpoint import p6_dense_get_temporary_xperf_config  # noqa F401
from .modeling_p6d import (
    P6DenseForCausalLM,
    P6DenseForSequenceClassification,
    P6DenseForTokenClassification,
    P6DenseModel,
)

AutoConfig.register("seed_p6dense", P6DenseConfig)
AutoModel.register(P6DenseConfig, P6DenseModel)
AutoModelForCausalLM.register(P6DenseConfig, P6DenseForCausalLM)
AutoModelForSequenceClassification.register(P6DenseConfig, P6DenseForSequenceClassification)
AutoModelForTokenClassification.register(P6DenseConfig, P6DenseForTokenClassification)
