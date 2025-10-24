Excellent question — this little term
[
\frac{1}{\sqrt{d_k}}
]
often looks arbitrary at first, but it’s *crucial* for the stability of attention training. Let’s unpack it carefully 👇

---

## 🧮 1. The formula

In scaled dot-product attention, the raw attention scores are computed as

[
\text{scores} = \frac{Q K^\top}{\sqrt{d_k}}
]

where:

* ( Q, K \in \mathbb{R}^{(B, T, d_k)} )
* ( d_k ) (often called **head size**) is the dimensionality of the key/query vectors.

---

## ⚖️ 2. Why scaling is needed

Each entry of ( QK^\top ) is a **dot product** between two ( d_k )-dimensional vectors:

[
\text{score}*{ij} = \sum*{t=1}^{d_k} Q_{it} K_{jt}
]

If each component of ( Q ) and ( K ) has zero mean and variance (1),
then by the Central Limit Theorem:

[
\mathrm{Var}[\text{score}_{ij}] \approx d_k
]

So as ( d_k ) grows, the variance of the dot product **increases linearly** with dimension.

This means raw scores get **larger in magnitude** for higher-dimensional embeddings.

---

## 🔥 3. What happens if we *don’t* scale

If we feed large scores into the `softmax`, we get extremely *peaked* distributions — one token gets near-1 probability, all others near 0.
That leads to:

* **Vanishing gradients** through the softmax,
* **Slow or unstable training**, especially when ( d_k ) is large (e.g. 64, 128).

---

## 🧊 4. The fix: scale down by ( \sqrt{d_k} )

By dividing by ( \sqrt{d_k} ), we rescale the dot-product variance back to roughly 1:

[
\mathrm{Var}\left[\frac{\text{score}_{ij}}{\sqrt{d_k}}\right] \approx 1
]

This keeps the softmax’s input distribution balanced and training numerically stable.

---

## 💡 5. Intuition in plain words

> The scaling term prevents attention scores from blowing up when the embedding dimension is large, keeping the softmax in a sensitive, learnable range.

---

## 🧾 6. Optional check in code

```python
import torch

B, T, d_k = 2, 4, 64
Q = torch.randn(B, T, d_k)
K = torch.randn(B, T, d_k)

raw_scores = Q @ K.transpose(-2, -1)
scaled_scores = raw_scores / (d_k ** 0.5)

print("Raw variance:", raw_scores.var().item())
print("Scaled variance:", scaled_scores.var().item())
```

Typical output:

```
Raw variance: ~64.0
Scaled variance: ~1.0
```

---

### ✅ Summary

| Concept                  | Without scaling   | With scaling         |
| ------------------------ | ----------------- | -------------------- |
| Variance of dot products | ~ (d_k)           | ~ 1                  |
| Softmax input range      | large → saturated | moderate → learnable |
| Training behavior        | unstable / slow   | stable / fast        |

---

**In short:**

> The division by ( \sqrt{d_k} ) normalizes the magnitude of attention scores so that the softmax doesn’t saturate, keeping attention learning smooth and stable regardless of head size.
