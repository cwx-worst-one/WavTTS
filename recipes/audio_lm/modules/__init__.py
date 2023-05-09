from .gpt import LanguageModel
from .loss import MaskedCrossEntropy

try:
    from transformers import LlamaForCausalLM
except Exception:
    print("huggingface llama disabled")
try:
    from .sparse_gpt import GPT2LMHeadSparseModel
except Exception:
    print("sparse_gpt disabled")
