import torch
from torch.nn import functional as F
from build_my_own_transformers import get_batch, encoder, decoder

torch.manual_seed(1337)
B,T,C = 4, 8, 2 # batch size, time step, channels
# todo: is channels = n_embedd?
x = torch.randn(B,T,C)

tril = torch.tril(torch.ones(T, T))
"""
tensor([[1., 0., 0., 0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 0., 0., 0.],
        [1., 1., 1., 0., 0., 0., 0., 0.],
        [1., 1., 1., 1., 0., 0., 0., 0.],
        [1., 1., 1., 1., 1., 0., 0., 0.],
        [1., 1., 1., 1., 1., 1., 0., 0.],
        [1., 1., 1., 1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1., 1., 1., 1.]])
"""
wei = torch.zeros(T, T)
wei = wei.masked_fill(tril==0, float('-inf'))
"""
tensor([[0., -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., -inf],
        [0., 0., 0., 0., 0., 0., 0., 0.]]) 
"""
wei = torch.softmax(wei, dim=-1)
"""
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3333, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 0.0000],
        [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000, 0.0000],
        [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.0000],
        [0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250]])
"""
xbow = wei @ x # (T,T) @ (B,T,C) -> (B,T,C)
xbow.shape # torch.Size([4, 8, 2])
"""
x[0]:
tensor([[ 0.1808, -0.0700],
        [-0.3596, -0.9152],
        [ 0.6258,  0.0255],
        [ 0.9545,  0.0643],
        [ 0.3612,  1.1679],
        [-1.3499, -0.5102],
        [ 0.2360, -0.2398],
        [-0.9211,  1.5433]])
        
xbow[0]:
tensor([[ 0.1808, -0.0700],
        [-0.0894, -0.4926], # 0.5*(0.1808-0.3596) = -0.0894, 0.5*(-0.0700-0.9152) = -0.4926
        [ 0.1490, -0.3199],
        [ 0.3504, -0.2238],
        [ 0.3525,  0.0545],
        [ 0.0688, -0.0396],
        [ 0.0927, -0.0682],
        [-0.0341,  0.1332]])
"""

batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8  #256 # what is the maximum context length for predictions?
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

# Load the data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
type(text) # str
len(text)

# we are training on character level prediction, so the vocabulary contains English characters
chars = sorted(list(set(text))) # get the unique characters of the input
vocab_size = len(chars) # 65
# create a mapping between chars and integers
char2int = {ch:i for i, ch in enumerate(chars)}
int2char = {i:ch for i, ch in enumerate(chars)}

# Tokenization: using encoder
data = torch.tensor(encoder(text), dtype=torch.long).to(device)
data.shape #torch.Size([1115394])
data[:10]
# peak into what's inside
decode_data = decoder(data[:10]) # 'First Citi'
# train and validation split
# for this dataset, the proper split would be taking out chunks of data without random shuffling
n = int(0.9*len(data)) # first 90% will be trained, rest val
train_data = data[:n]
val_data = data[n:]
xb, yb = get_batch(train_data)

# Implementing on real text
weights = torch.zeros(block_size, block_size, device=device, dtype=torch.float32) # T,T # not ones, because we will softmax it
tril = torch.tril(torch.ones(block_size, block_size)).to(device)
weights = weights.masked_fill(tril == 0, float('-inf')) # where tril=0 fill that with -inf, other places of the weights stay
weights = F.softmax(weights, dim=-1).to(device) # T,T
# aggregation
attention = xb[0] @weights
# RuntimeError: MPS device does not support mm for non-float inputs
# Reason: xb[0] is torch.long (ints). @ on MPS needs float/bfloat16/half.
xb0 = xb
xb0 = xb0.to(device).to(torch.float32)
attention = weights @ xb0.T
# Why xb0.T here: because xb0 is [B,T] which misses the last dimension C, embed size; if it's [B,T,C], then we don't need the transpose
attention[:,0]  # 1st column of attention is the window of average of xb[0]
# tensor([ 1.0000, 23.5000, 33.3333, 38.5000, 39.4000, 33.0000, 35.0000, 37.1250], device='mps:0')
# which is exacly the same as I calcualte the window of averages with xb[0] = tensor([ 1, 46, 53, 54, 43,  1, 47, 52], device='mps:0')
