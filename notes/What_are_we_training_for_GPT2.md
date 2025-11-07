Great question. In a GPT-2–style Transformer, **almost everything that multiplies a tensor is trained**. Concretely:

### Embeddings

* **Token embedding matrix (`wte`)**: (V \times C) (vocab × model width). Learned.
* **Positional embedding matrix (`wpe`)**: (T_{\max} \times C). Learned (GPT-2 uses learned absolute position embeddings).
  *Note:* other models may use fixed sinusoidal or RoPE instead (not learned).

### Per Transformer block (repeated L times)

* **Attention projections**

  * **Q/K/V projection weights**: usually fused as one linear `c_attn: C × 3C` (+ biases). These produce Q, K, V.
  * **Output projection**: `c_proj: C × C` (+ bias). Mixes concatenated heads back to width (C).
* **MLP / Feed-Forward (FFN)**

  * **Up-projection**: `c_fc: C × 4C` (+ bias) in GPT-2 (GELU).
    *(Some variants use “gated” MLPs with two matrices; still learned.)*
  * **Down-projection**: `c_proj: 4C × C` (+ bias).
* **LayerNorms**

  * **Weights (gamma) and biases (beta)** for the pre-attn and pre-MLP norms (and a **final** LayerNorm `ln_f` at the top). All learned.

### Output head

* **LM head (`lm_head`)**: maps final hidden state to logits over vocab.
  In GPT-2, **weights are tied** to the token embedding (`lm_head.weight = wte.weight`), so there’s not an extra independent matrix—just reused parameters.

### What’s **not** trained

* **KV cache at inference**: runtime buffer, no parameters.
* **Dropout rates, mask tensors, softmax, scaling factors**: ops, not learned.
* **Optimizer states** (Adam moments, etc.): used during training but not part of the model parameters.

If you look at Hugging Face names for GPT-2 small (124M), you’ll see these as:

* `transformer.wte`, `transformer.wpe`
* For each layer `h.N`:

  * `attn.c_attn.weight/bias` (QKV), `attn.c_proj.weight/bias` (W_O)
  * `ln_1.weight/bias`, `ln_2.weight/bias`
  * `mlp.c_fc.weight/bias`, `mlp.c_proj.weight/bias`
* `transformer.ln_f.weight/bias`
* `lm_head.weight` (tied to `wte.weight`)

That’s the whole trainable set in GPT-2 terms.
