import tiktoken
from gpt2 import GPT2
from config import GPT2DataConfig
import torch


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"Using {device} device")

model_type = "gpt2"
max_length = 30
max_return_sequences = 5

# Port the Hugging Face GPT2 model parameters
gpt2_hf = GPT2(GPT2DataConfig()).from_pretrained(model_type)
# Move model to MPS and turn on Eval mode
gpt2_hf.to(device)
gpt2_hf.eval()

# Get the encoder
encoder = tiktoken.get_encoding("gpt2")

prompt = "Hello, I am a Language Model."
prompt_tokens = encoder.encode(prompt) # a list
prompt_idx = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
prompt_idx = prompt_idx.unsqueeze(0).repeat(max_return_sequences, 1)

generated_tokens = gpt2_hf.generate(prompt_idx, max_new_tokens=max_length, temperature=1.0, top_k=50)
for i in range(max_return_sequences):
    t = generated_tokens[i].tolist()
    decoded = encoder.decode(t)
    print(">", decoded)

