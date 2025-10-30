Great question — this is an important PyTorch concept, especially for things like masks and positional encodings.

---

### 🧩 What a “registered buffer” is

A **registered buffer** is a tensor that’s **part of a module’s state**, but **isn’t a trainable parameter** (i.e. it won’t show up in `model.parameters()` or get gradients).

You register it with:

```python
self.register_buffer("name", tensor)
```

That tells PyTorch:

* ✅ it **belongs to the module** (so it’s saved/loaded with `state_dict()`);
* ✅ it **moves automatically** when you call `model.to(device)` or `model.cuda()` — no need to manually do `tensor.to(device)`;
* 🚫 but it **does not require gradients** and is **not updated** by the optimizer.

---

### 💡 When to use it

Use a registered buffer for **constant tensors** that are part of the model definition but depend on device or dtype, for example:

* causal masks (`tril`)
* positional encodings (fixed)
* normalization statistics (non-learnable)
* padding masks or lookup tables

---

### ⚙️ Example

Without buffer (bug-prone):

```python
self.tril = torch.tril(torch.ones(block_size, block_size))
# later in forward
attention = attention.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
# ❌ self.tril stays on CPU; mismatch if model moved to GPU or MPS
```

With buffer (robust):

```python
self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
# automatically on correct device when you call model.to("mps")
attention = attention.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
```

---

### 🔍 You can inspect buffers like this:

```python
list(model.buffers())
```

and see them appear in:

```python
model.state_dict().keys()
```

but **not** in:

```python
model.parameters()
```

---

In short:

> 🧠 A registered buffer is a non-trainable tensor that moves with your model and is saved with its weights — perfect for masks and constants.

Would you like me to show how your `SelfAttentionHead` should define and use a registered buffer line-by-line?
