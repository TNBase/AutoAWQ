from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, TextStreamer

quant_path = "/code/AutoAWQ"
quant_file = "vicuna-7b-base-4bit-AWQ.pt"

# Load model
model = AutoAWQForCausalLM.from_quantized(quant_path, quant_file, fuse_layers=True)
tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)
streamer = TextStreamer(tokenizer, skip_special_tokens=True)

# Convert prompt to tokens
prompt_template = """\
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

USER: {prompt}
ASSISTANT:"""

tokens = tokenizer(
    prompt_template.format(prompt="How are you today?"), 
    return_tensors='pt'
).input_ids.cuda()

# Generate output
generation_output = model.generate(
    tokens, 
    streamer=streamer,
    max_new_tokens=512
)
