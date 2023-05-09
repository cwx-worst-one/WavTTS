from .gpt import LanguageModel
from .loss import MaskedCrossEntropy
# from transformers import LlamaForCausalLM
try:
    from .sparse_gpt import GPT2LMHeadSparseModel
except:
    print("sparse_gpt disabled")
