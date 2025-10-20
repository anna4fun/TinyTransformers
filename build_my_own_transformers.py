# This piece of code is aim to build a transformer that generate texts
# Following Andrej Karpathy's YouTube Video "Let's build GPT: from scratch, in code, spelled out."
# We start with the course's example training data, which is the Shakespear
# Then take another dataset TBD

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torch import cuda
import gc

torch.manual_seed(1337)
# Device
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

# hyperparameters
batch_size = 10 # 64 # how many independent sequences will we process in parallel?
block_size = 8  #256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 32 # 384 # don't change to 2, it will all be 0 and 1
n_head = 6
n_layer = 6
dropout = 0.2

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
char2int['?'] # 12
int2char[25] # 'M'

# encoder: a function that turns text into a list of integers using the char2int mapping
def encoder(sentence: str):
    return [char2int[ch] for ch in sentence]

powell_says = 'good afternoon'
powell_says_encoded = encoder(powell_says)

# decoder: a function turns integers back to a full text string using the int2char mapping
def decoder(int_lst):
    # allow both python list and pytorch tensor
    if isinstance(int_lst, torch.Tensor):
        indices = int_lst.detach().cpu().tolist()
    return ''.join(int2char[int(i)] for i in int_lst)

print(decoder(powell_says_encoded)) # good afternoon
print(decoder(torch.tensor(powell_says_encoded))) # good afternoon

# Tokenization: using encoder
data = torch.tensor(encoder(text), dtype=torch.long).to(device)
data.shape #torch.Size([1115394])
data[:10]
# peak into what's inside
decode_data = decoder(data[:10]) # 'First Citi'

# don't do this: embed data with one-hot-encoding
# data = F.one_hot(data, n_embd)

# train and validation split
# for this dataset, the proper split would be taking out chunks of data without random shuffling
n = int(0.9*len(data)) # first 90% will be trained, rest val
train_data = data[:n]
val_data = data[n:]

# data loading, split into batches
# todo: change to DataLoader object
def get_batch(ds):
    # ds for dataset, can be train_data or val_data
    # x : [i:i+block_size-1]; y[i+1, i+block_size], no need to exp pass block_size
    # randomly draw batch_size amount of i
    starting_index = torch.randperm(len(ds)-block_size-1)[:batch_size]
    x = torch.stack([ds[i:i+block_size] for i in starting_index])     # torch.Size([64, 256])
    y = torch.stack([ds[i+1:i+block_size+1] for i in starting_index]) # torch.Size([64, 256])
    x, y = x.to(device), y.to(device)
    return x, y

# TODO: fill out loss calculation
@torch.no_grad()
def estimate_loss():
    output = {}
    model.eval() # the eval() function is inherite from nn.Module
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y  = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        output[split] = losses.mean()
    model.train()
    return output


# sample one batch
xb, yb = get_batch(train_data)
# let's zoom in to xb[0] to do the self-attention, for simple consumption, I temporarily dialed down the block_size to 8
xb[0] # tensor([61,  1, 45, 53, 47, 52, 45,  1], device='mps:0')
print(decoder(xb[0])) # "w going "

# The Simple Bigram Model
# p(next_token|current_token): each token is predicted from the token ID before it
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # The embedding table is the target and only output of the Bigram model
        # Embedding table is a lookup table for indexing,
        # each row correspond to 1 token ID,
        # each row vector is the logits(probability distribution) for the next token
        # aka, table[i, j] = p1 means: when current token is i, the probability of the next token being j is p1
        # sum(table[i,j], j = 1 to vocab_size) = 1
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # TODO: verify idx is the index of current token that is passed into the model
        # TODO: what is the logits?
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(targets, logits)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # sample from the multinomial probability distributions of the next token of each current token
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions and loss
            logits, loss = self(idx) # todo: what does self(idx) mean?
            # focus on only the last token (where T dimention only remains 1 element)
            logits = logits[:, -1, :] # (B,1,C), aka (B,C) # todo: is C = n_embd = vocab_size here?
            # Make the logits into multinomial distribution
            # todo: this is weird, if there's (B,C), then would the softmax takes C vector and softmax by row
            prob_distribution = F.softmax(logits, dim=-1)  # (B,C)
            # generate the next token: for each row in B, sample from the distribution saved in row[b]
            idx_next = torch.multimonial(prob_distribution, num_samples=1) # (B,1)
            # append the generated next token to the running sequence idx which is (B,T) + (B,1) = (B, T+1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decoder(m.generate(context, max_new_tokens=500)[0].tolist()))



# Transformers
# now we move on to Self attention with K, Q, V
# let's use cosine similarity of k and q to represent weights, for just xb0
# weights means: how much interest do I care about other tokens (before me, here)
# Callouts: all batches share the same weights matrix (do they?)

cuda.empty_cache()
gc.collect()
