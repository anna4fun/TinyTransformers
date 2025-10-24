# What is Dropout?

### ✅ What dropout actually does

* **Dropout** is a **regularization technique** used to prevent overfitting.
* During **training**, dropout randomly **sets a fraction of activations (not parameters)** to zero.
* The random masking happens on **the output of neurons** (e.g., activations in a layer), not on the weights themselves.

So the key difference:

> 🔸 You said “some percent of parameters (weights)” — but actually it’s “some percent of neuron outputs (activations)” that are dropped.

---

### 🧠 Intuitive picture

Suppose you have a hidden layer:
$$ h = f(Wx + b) $$
and dropout rate = 0.3.
Then during training:
$$ h' = \text{dropout}(h, p=0.3) $$
means 30% of the elements in (h) are set to 0 randomly.
This forces the network to **not rely on any one neuron too much**, improving generalization.

---

### 🧩 During inference

At inference time, **dropout is turned off**, but the outputs are **scaled** (or equivalently, training-time outputs are scaled by (1/(1-p))) so that the expected activation magnitude remains consistent.

---

### TL;DR

| Phase     | What happens                          | Why                                     |
| --------- | ------------------------------------- | --------------------------------------- |
| Training  | Randomly zero out some neuron outputs | Prevent over-reliance and co-adaptation |
| Inference | No dropout, but outputs are scaled    | Keep activations consistent             |

---

Would you like me to show a **PyTorch example** of how dropout behaves differently during training vs evaluation?

# Would the randomness happen at per forward pass per layer level?
Let's say I have a 3 layered NN, Linear1(L1), Relu(L2), Linear2(L3), and I have 5 batches of data in one epoch; when batch 1 go through L1, L2, L3, would each layer's output logits be randomly have 30% of logits set to 0? Would batch 2, 3, 4, 5 have very different 30% logits be set to 0?

Excellent question — and yes, your intuition is right on track 👇

Let’s unpack it carefully, step by step:

---

### 🔹 Suppose you have

```python
x -> L1 (Linear)
   -> ReLU
   -> Dropout(p=0.3)
   -> L2 (Linear)
```

and you’re processing **5 batches per epoch**.

---

### 🔸 What happens during training

When you call `model.train()` and run batch 1:

* **Dropout randomly zeroes out 30% of the activations** (outputs of ReLU) **for that batch only**.
* The mask (which activations to drop) is **sampled anew for each forward pass**.

So yes:

* Batch 1 → a random 30% of activations dropped
* Batch 2 → a different random 30% dropped
* Batch 3 → another different random mask, etc.

Each batch has its own independent dropout mask.

---

### 🔸 What about multiple layers?

If you have dropout after each layer (say after both L1 and L3),
then **each dropout layer** samples its **own random mask**, independently per batch.
So:

* Dropout1 (after L1) → random mask A for batch 1
* Dropout2 (after L3) → random mask B for batch 1
* Batch 2 → new random A′ and B′, etc.

That’s the key: randomness is **per layer and per forward pass**.

---

### 🔸 During evaluation (`model.eval()`)

All dropout layers are **disabled**.
No activations are zeroed — instead, the layer scales the outputs by the expected keep probability (1 − p) so your logits stay consistent in magnitude.

---

### ✅ Summary

| Setting                     | When does dropout apply? | Random per batch? | Random per layer? |
| --------------------------- | ------------------------ | ----------------- | ----------------- |
| Training (`model.train()`)  | Yes                      | ✅ Yes             | ✅ Yes             |
| Evaluation (`model.eval()`) | No                       | ❌ N/A             | ❌ N/A             |

---

So your statement is 100% correct if you just change “logits” to **“layer activations”** — dropout zeroes out random activations per batch, per layer, each forward pass.

Would you like me to show what the dropout masks actually look like in a tiny PyTorch example (e.g., same input passed multiple times)?

Perfect 👇 here’s a minimal PyTorch example showing exactly how **dropout produces different random masks** for each batch (even with the same input):

```python
import torch
import torch.nn as nn

# Define a tiny network
dropout = nn.Dropout(p=0.3)   # 30% dropout
x = torch.ones(1, 10)         # pretend this is layer output (1 batch of 10 activations)

dropout.train()  # training mode
for i in range(5):
    y = dropout(x)
    print(f"Batch {i+1}: {y}")
```

### 🧠 What happens:

* Each time `dropout(x)` is called, PyTorch samples a **different random binary mask** of 1s and 0s.
* 30% of elements are zeroed out each time.
* The kept elements are scaled by `1/(1 - p)` = `1/0.7 ≈ 1.4286` to preserve the expected activation magnitude.

Sample output (your exact numbers will differ because of randomness):

```
Batch 1: tensor([[0., 1.4286, 1.4286, 0., 1.4286, 1.4286, 0., 1.4286, 1.4286, 0.]])
Batch 2: tensor([[1.4286, 0., 1.4286, 1.4286, 1.4286, 0., 1.4286, 0., 1.4286, 0.]])
Batch 3: tensor([[0., 1.4286, 0., 1.4286, 1.4286, 1.4286, 0., 1.4286, 0., 1.4286]])
...
```

You can see each forward pass (batch) has **different elements zeroed out**.

---

If you then switch to evaluation mode:

```python
dropout.eval()
y = dropout(x)
print("Eval mode:", y)
```

you’ll get:

```
Eval mode: tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
```

No randomness — dropout is **disabled** during inference.

---

So yes, dropout re-samples a new random mask every batch, for every dropout layer, during training.

Would you like me to show how you can **visualize the mask itself** (e.g., print which neurons got dropped)?

# Where to apply Dropout (practically)
Great instincts—and you found two classic “gotchas”:

### 1) Where to apply dropout in self-attention

Don’t apply dropout to the **raw attention scores** `weight = Q @ K.T / sqrt(d_k)` (pre-softmax).
Best practice is to apply **attention dropout *after* softmax**, i.e., to the attention **probabilities** and **before** the `@ V`:

```python
scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)      # (B, h, T, T)
scores = scores.masked_fill(attn_mask==0, float('-inf'))  # causal/pad mask
attn   = scores.softmax(dim=-1)                           # probabilities
attn   = self.attn_drop(attn)                             # <- attention dropout here
out    = attn @ V
```

Also keep a separate dropout for the MLP/residual path:

```python
self.attn_drop = nn.Dropout(config.attn_dropout)  # e.g., 0.1
self.resid_drop = nn.Dropout(config.resid_dropout)
```

### 2) “Dropout changed values instead of keeping them”

PyTorch uses **inverted dropout**: kept elements are scaled by `1/(1-p)` at **training time**.
So you should expect:

* ~30% elements → set to **0**
* the other ~70% → **multiplied by 1/(1-0.3)=1/0.7≈1.4286**
  That’s why values change rather than “staying exactly the same”.

If you truly want a mask that zeros 30% and leaves others unchanged (not recommended), you’d need a custom Bernoulli mask without scaling—but that will shrink activation magnitudes and hurt training stability.

### 3) “First row became all zeros” — what’s going on?

A few plausible causes:

* **Causal/padding mask + attention dropout**:
  For some positions (e.g., the very first token), the causal mask may allow **only 1 column** to attend (itself). If you then drop that single entry with p=0.3, that **entire row becomes zero**. This is expected behavior with attention dropout and is one reason people keep **attention dropout lower** (e.g., 0.1).

* **Applying dropout to pre-softmax scores**:
  Dropping scores before softmax interacts badly with the subsequent `softmax` and masking; it can create very skewed or degenerate rows.

* **Broadcasting bug**:
  If your dropout mask’s shape is wrong (e.g., `(T,1)` and it broadcasts across a head or batch dimension), you can accidentally zero entire rows/heads. Ensure dropout is applied element-wise to the full `(B, num_heads, T, T)` tensor.

### 4) Practical recommendations

* Use **`self.attn_drop = nn.Dropout(0.1)`** and apply it **after softmax** on attention probabilities.
* Keep **`self.resid_drop = nn.Dropout(0.1–0.2)`** for MLP/residual outputs.
* Ensure you’re in the right mode:

  * `model.train()` → dropout active (scales kept elements)
  * `model.eval()` → dropout disabled (no zeros, no scaling)
* If you saw whole rows zeroed consistently, double-check:

  * the **masking logic** (causal/pad masks),
  * the **tensor shapes** (no unintended broadcasting),
  * the **dropout rate** (use smaller `attn_dropout`, e.g., 0.1).

If you want, paste your attention block (the few lines around scores→mask→softmax→dropout→`@V`) and I’ll spot-fix it line-by-line.


# Coding tips
Short answer: defining it like `self.dropout = nn.Dropout(config.dropout)` is fine—the **mask isn’t created at init**; it’s (re)sampled **every forward pass** when you call `self.dropout(tensor)` in `train()` mode.

The issues you’re seeing come from **where** you apply dropout and a couple of small misconceptions:

## What to fix

1. **Apply dropout after softmax (on attention probs), not on raw scores.**
   Also keep a *separate* dropout on the output/residual path.

2. **It’s normal that kept entries change value.**
   PyTorch uses **inverted dropout**: kept values are scaled by `1/(1-p)`. So you *won’t* see “unchanged” 70%—they’re scaled up.

3. **“Whole first row became zeros” is expected with causal masks + attn-dropout.**
   For the very first token, the causal mask may allow **only one** valid key (itself). If dropout removes that single prob, the row becomes all zeros.
   Mitigations:

* use a **smaller `attn_dropout`** (e.g., 0.0–0.1 for causal LMs), or
* apply dropout to the **output** (`out = self.resid_drop(attn @ v)`) instead of the probs, or
* both.

4. **Dot-product attention ≠ cosine similarity.**
   It’s a **scaled dot product**; unless you explicitly L2-normalize q/k, it’s not cosine similarity.

## Minimal, robust rewrite (single head)

```python
class SelfAttentionHead(nn.Module):
    def __init__(self, config: ModelConfig, data_config: DataConfig, head_size: int):
        super().__init__()
        self.config = config
        self.data_config = data_config
        self.head_size = head_size

        self.Wq = nn.Linear(config.n_embd, head_size, bias=False)
        self.Wk = nn.Linear(config.n_embd, head_size, bias=False)
        self.Wv = nn.Linear(config.n_embd, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(data_config.block_size, data_config.block_size)))

        # Use two dropouts: one for attention probs, one for the residual/output
        self.attn_drop  = nn.Dropout(getattr(config, "attn_dropout", 0.1))
        self.resid_drop = nn.Dropout(getattr(config, "dropout", 0.1))

    def calc_weights(self, x, decoder: bool):
        q = self.Wq(x)                        # (B,T,hs)
        k = self.Wk(x)                        # (B,T,hs)
        scores = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)   # (B,T,T)

        if decoder:
            T = x.size(1)
            scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)      # (B,T,T) attention probs
        attn = self.attn_drop(attn)           # attention dropout (consider 0.0–0.1 for LMs)
        return attn

    def forward(self, x, decoder: bool = True):
        v = self.Wv(x)                        # (B,T,hs)
        attn = self.calc_weights(x, decoder)  # (B,T,T)
        out = attn @ v                        # (B,T,hs)
        out = self.resid_drop(out)            # output/residual dropout
        return out
```

### Practical tips

* Use `model.train()` to enable dropout; `model.eval()` disables it.
* If you dislike the “all-zero first row” risk in early positions, set `attn_dropout=0.0` and keep only `resid_drop`.
* Double-check mask shapes/devices to avoid accidental broadcasting that could zero whole rows/heads.


Excellent question — you can definitely visualize dropout masks directly in PyTorch! 👇

Let’s go step-by-step.

---

### 🧩 1. How dropout works under the hood

When you call:

```python
y = dropout(x)
```

PyTorch internally samples a **Bernoulli mask** with probability `1 - p` of being kept, multiplies it element-wise with `x`, and scales the kept elements by `1/(1 - p)`.

But you can **recreate and inspect that mask** yourself.

---

### 🧪 2. Example: visualize the dropout mask

```python
import torch
import torch.nn as nn

# Simulate some activations
x = torch.ones(1, 10)     # pretend 10 neuron activations
drop = nn.Dropout(p=0.3)  # 30% dropout

drop.train()               # enable dropout

# Forward pass
y = drop(x)

# Derive the mask manually:
mask = (y != 0).int()      # 1 where kept, 0 where dropped

print("Input:", x)
print("Output after dropout:", y)
print("Mask (1=kept, 0=dropped):", mask)
```

💡 Expected output:

```
Input:  tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
Output after dropout: tensor([[1.4286, 1.4286, 0.0000, 1.4286, 0.0000, 1.4286, 1.4286, 0.0000, 1.4286, 0.0000]])
Mask (1=kept, 0=dropped): tensor([[1, 1, 0, 1, 0, 1, 1, 0, 1, 0]], dtype=torch.int32)
```

You can see exactly which neurons were “kept” or “dropped.”

---

### 🧠 3. Visualizing for your attention weights

If you want to inspect the dropout mask applied to your attention weights (e.g. `(B, T, T)` tensor), same idea:

```python
weight = torch.rand(1, 5, 5)
drop = nn.Dropout(p=0.3)
drop.train()

y = drop(weight)
mask = (y != 0).int()

print("Attention weights:\n", weight)
print("After dropout:\n", y)
print("Mask:\n", mask)
```

You’ll see a 5×5 binary matrix where 0s indicate dropped attention connections.

---

### 🧩 4. If you want to capture the mask *used* by `nn.Dropout`

You can subclass `nn.Dropout`:

```python
class DebugDropout(nn.Dropout):
    def forward(self, input):
        if self.training:
            # generate the same mask PyTorch would internally use
            mask = torch.rand_like(input) > self.p
            print("Mask:\n", mask.int())
            return mask * input / (1 - self.p)
        else:
            return input
```

Then use `self.dropout = DebugDropout(p)` in your model to print the masks as they’re created during training.

---

Would you like me to show how to **visualize dropout masks across layers** (e.g., for a 3-layer MLP or multi-head attention) using a small heatmap?
