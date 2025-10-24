Great follow-up! Short answer: you *can* compute `K @ Q^T`, but you must transpose it back (or change which axis you softmax over) to keep the semantics right.

### Why we use `Q @ K^T`

* We want, for **each query token i**, a probability distribution over **keys j** in the *same sequence*.
* With `scores = Q @ K^T` (shape `(B,T,T)`), **row i** contains all scores from query *i* to every key *j*.
* Then `softmax(scores, dim=-1)` makes each row sum to 1, which matches “distribution over keys for each query”.

### What happens with `K @ Q^T`

* `K @ Q^T` is just the **transpose** of `Q @ K^T`:
  [
  KQ^\top = (QK^\top)^\top
  ]
* Now **column j** corresponds to query *j*. If you apply `softmax(..., dim=-1)` here, you’re normalizing **across queries** for each key, which is the wrong direction for standard attention.
* To fix it, either:

  1. transpose back before softmax, or
  2. keep it as-is but softmax over the **second-to-last** dim (`dim=-2`) and also adapt your causal mask axis, and finally note that when multiplying by `V`, you expect `(B,T,T) @ (B,T,d_v)` with rows indexing queries—so you’ll likely need a transpose anyway.

### Correct PyTorch patterns

**Standard (recommended):**

```python
scores = Q @ K.transpose(-2, -1)              # (B,T,T)
scores = scores * (Q.size(-1) ** -0.5)        # scale by d_k^-1/2
weights = torch.softmax(scores, dim=-1)       # row-wise
out = weights @ V                             
```

**If you start with `K @ Q^T`, transpose back:**

```python
scores = K @ Q.transpose(-2, -1)              # (B,T,T)
scores = scores.transpose(-2, -1)             # back to (B,T,T) where rows = queries
scores = scores * (K.size(-1) ** -0.5)
weights = torch.softmax(scores, dim=-1)
out = weights @ V
```

**Equivalence check:**

```python
B, T, d_k, d_v = 2, 4, 8, 16
Q = torch.randn(B, T, d_k)
K = torch.randn(B, T, d_k)

scores_qk = Q @ K.transpose(-2, -1)                        # (B,T,T)
scores_kq = (K @ Q.transpose(-2, -1)).transpose(-2, -1)    # (B,T,T)

assert torch.allclose(scores_qk, scores_kq, atol=1e-6)
```

### TL;DR

* `K @ Q^T` isn’t wrong mathematically—it’s ((QK^T)^T).
* But to keep the **row = “query i attends to keys j”** convention (and make masking and `@ V` align naturally), stick with `Q @ K^T`. If you do `K @ Q^T`, transpose/axes must be adjusted accordingly.
