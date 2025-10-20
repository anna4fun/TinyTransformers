# Context
When I tried to implement the self attention out of un-embedded single character sentence, I realized the matrix multiplication doesn't work, because my X is (B,T), while weight matrix is (T,T), so `weight @x` doesn't make multiplication in the right orientation, it has to be `weight @x.T`, so I am very eager to make X to be (B,T,C), C = channels = n_embd, so I can make the code works more continuousely.  
I initially took the shortcut to be One Hot Encoding on the whole dataset,  but immediately found out xb[0] looks really sparse(too many zero, hardly any 1) which is very suspecious. 

# Question 1: The difference between nn.Embedding(vocab_size, n_embd) and F.one_hot(data, n_embd)
Excellent — you’re asking the *right* questions here. This gets at one of the deepest conceptual differences between **one-hot encoding** and **learned embeddings**.

Let’s unpack both parts carefully 👇

---

## 🧩 1. What `nn.Embedding(vocab_size, n_embd)` does

It’s basically a **lookup table** that maps each token ID (integer) to a learned dense vector — the *embedding vector*.

Formally:

* Input: integer token IDs of shape `(B, T)` (dtype = `long`)
* Output: dense float vectors of shape `(B, T, n_embd)`

Inside, PyTorch stores a learnable matrix:

```
E = nn.Embedding(vocab_size, n_embd)
E.weight  # shape: [vocab_size, n_embd]
```

When you call `E(x)`, PyTorch does:

```python
E(x)[b, t] = E.weight[x[b, t]]
```

That is — for each token ID, it **looks up** the corresponding row vector in that weight matrix.

### 👉 Analogy

If your vocab has 65 characters and `n_embd = 384`:

* The embedding layer starts as a random 65×384 matrix.
* Each row (one per character) is a dense 384-dimensional vector.
* During training, the network learns to *adjust these vectors* so that semantically or contextually similar tokens get similar embeddings.

---

## 🧩 2. Why your `F.one_hot(data, n_embd)` looks super sparse

Let’s look at what `one_hot` actually does.

```python
data = torch.tensor([0, 1, 4])
F.one_hot(data, num_classes=10)
```

→ gives

```
tensor([[1,0,0,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,0,0,0,0,0]])
```

✅ This is **expected** — one-hot vectors have exactly **one 1** and the rest zeros.

So if your vocab_size = 65 and n_embd = 384, you’re now creating a **65×384 = 24,960-dimensional** space where each token vector has only a **single 1** — everything else is 0.
That’s extremely sparse, and totally non-learnable — every token stays orthogonal to every other one, forever.

---

## 🧠 3. Why `nn.Embedding` is *better than* one-hot

| Aspect              | One-hot                     | nn.Embedding                                 |
| ------------------- | --------------------------- | -------------------------------------------- |
| Representation      | fixed, 1-of-N binary vector | dense, learned continuous vector             |
| Storage             | large, sparse               | compact, small matrix                        |
| Trainable           | ❌ no                        | ✅ yes                                        |
| Semantic similarity | all tokens equally distant  | can learn closeness (e.g. “a” and “b” close) |
| Efficiency          | multiplies by mostly zeros  | just looks up a row (O(1))                   |

Essentially, **`nn.Embedding` learns a continuous, information-rich replacement for the one-hot encoding**.

In deep learning practice, one-hots are rarely used beyond initial illustration — because embeddings both **compress** and **generalize** far better.

---

## 🧩 4. Small demo

```python
import torch, torch.nn.functional as F, torch.nn as nn

vocab_size = 5
n_embd = 3
x = torch.tensor([0, 2, 4])

# One-hot encoding
print(F.one_hot(x, vocab_size).float())
# -> each row has a single 1 and 4 zeros

# Learned embedding
E = nn.Embedding(vocab_size, n_embd)
print(E(x))
# -> random dense float vectors that will be learned during training
```

---

### TL;DR

* `nn.Embedding(vocab_size, n_embd)`
  → creates a learnable lookup table that maps integer tokens → dense float vectors.

* `F.one_hot()`
  → creates *fixed* sparse 0/1 vectors — huge, untrainable, inefficient.

So when you saw `xb[0]` being “mostly zeros,” that’s not a bug — that’s literally what one-hot encoding is.
To make the model learn meaningful representations, you want to use **`nn.Embedding`**, not one-hot.



# Question 2: Where does the embedding happens
So shall I do this: 
1. load the text into data; 
2. E = nn.Embedding(vocab_size, n_embd) and call data = E(data) as the new source of train and validation set;
3. split the train and valid set and start the attention calculation?

Short answer: **don’t pre-embed your whole dataset.** Keep your dataset as integer token IDs (`torch.long`). Apply the embedding **inside the model’s `forward`** on each minibatch so it stays trainable.

Here’s the right flow:

1. **Load & tokenize once (ints)**

```python
data = torch.tensor(encoder(text), dtype=torch.long)
n = int(0.9*len(data))
train_data, val_data = data[:n], data[n:]
```

2. **Batch integers**

```python
def get_batch(ds, batch_size, block_size, device):
    ix = torch.randint(len(ds) - block_size - 1, (batch_size,))
    x = torch.stack([ds[i:i+block_size] for i in ix]).to(device)        # (B,T) long
    y = torch.stack([ds[i+1:i+block_size+1] for i in ix]).to(device)    # (B,T) long
    return x, y
```

3. **Embed inside the model (per batch)**

```python
import torch.nn as nn
import torch.nn.functional as F

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)              # (V,C)
        self.pos_emb   = nn.Embedding(block_size, n_embd)              # (T,C)
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx):  # idx: (B,T) long
        B, T = idx.shape
        tok = self.token_emb(idx)                                       # (B,T,C)
        pos = self.pos_emb(torch.arange(T, device=idx.device))          # (T,C)
        x = tok + pos                                                   # (B,T,C)
        # … put your attention/feedforward blocks here on x …
        x = self.ln(x)
        logits = self.lm_head(x)                                        # (B,T,V)
        return logits
```

4. **Train normally**

```python
model = TinyTransformer(vocab_size, n_embd=384, block_size=256).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()

xb, yb = get_batch(train_data, 64, 256, device)
logits = model(xb)                           # (B,T,V)
loss = loss_fn(logits.view(-1, logits.size(-1)), yb.view(-1))
loss.backward(); opt.step(); opt.zero_grad()
```

Why not `data = E(data)` upfront?

* **Embeddings are learnable parameters**—they change every step. Precomputing them would **freeze** them (wrong) and waste memory.
* You want gradients to flow **into `nn.Embedding.weight`**, which happens when you call it **during `forward`**.

After that, add your **causal attention block(s)** between `x = tok + pos` and the `lm_head`.
