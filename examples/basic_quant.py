from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from huggingface_hub import login

login('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')
model_path = 'meta-llama/Llama-2-7b-hf'
quant_path = 'Llama-2-7b-hf'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4 }

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path, token='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')
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, token='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')

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')