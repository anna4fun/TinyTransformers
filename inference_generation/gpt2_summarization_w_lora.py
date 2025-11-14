from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

MODEL_NAME = "gpt2"
ADAPTER_DIR = "checkpoints/gpt2-summarization-lora"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto" if torch.cuda.is_available() else None,
)
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

INSTRUCTION_TEMPLATE = """You are a helpful assistant that summarizes news articles.

Article:
{article}

Summary:
"""

article_text = "YOUR LONG ARTICLE TEXT HERE..."

prompt = INSTRUCTION_TEMPLATE.format(article=article_text)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print("Summary:", generated)
