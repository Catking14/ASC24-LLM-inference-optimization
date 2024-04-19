import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("/mlsteam/data/data/AquilaChat2-34B-Int4-GPTQ", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("/mlsteam/data/data/AquilaChat2-34B", trust_remote_code=True)
tokenizer.save_pretrained("/mlsteam/data/data/aquilachat2-34b-int4-gptq-marlin-hf.marlin.g128")