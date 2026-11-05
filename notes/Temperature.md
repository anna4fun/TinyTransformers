Temperature **doesn’t change the rank order of logits** (it’s just a positive scaling), but it **does change how you *sample* among the candidates**, which is where the “randomness” comes from.

### What temperature actually does

Given logits (z_i), decoding turns them into probabilities via softmax:

$$ p_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)} $$

* **Lower (T) (< 1)**: the gap between a higher logits and lower logits became bigger, makes the distribution *sharper* (more peaked). The top token gets even more probability → outputs become more deterministic.
* **Higher (T) (> 1)**: makes the distribution *flatter*. Lower-ranked tokens get more probability → outputs vary more.

A key identity:


$$ \frac{p_i}{p_j} = \exp\left(\frac{z_i - z_j}{T}\right) $$ 


So changing (T) changes probability *ratios* a lot, even though the **ordering** of (z_i) doesn’t change.

### “But we take top-K… shouldn’t that be stable?”

There are two different things people mean by “top-K”:

1. **Top-K *sampling*** (common):

* Keep only the top K tokens by logit (candidate set).
* Re-normalize probabilities over those K tokens.
* **Sample** one token from that distribution.

➡️ Temperature **doesn’t change which tokens are in the top-K** (rank unchanged), but it **changes the probabilities inside the top-K**, so sampling becomes more/less random.

2. **Top-K then pick the best (argmax)** (less common):

* Keep top K tokens, then always pick the highest-prob one.

➡️ This is deterministic, and **temperature won’t change the chosen token** (because rank is unchanged). In this case, temperature effectively does nothing.

### One more subtlety: top-p (nucleus) *can* change the candidate set

If you use **top-p**, the set of allowed tokens is “the smallest set whose probabilities sum to (p).”
Since temperature changes probabilities, **the top-p set can change**, which can also change outputs even before sampling.

---

You’re very close, but the key is:

**It’s not “small (z_j) becomes smaller” in an absolute sense — it’s that the *gap* between logits becomes larger, and softmax cares about those gaps.**

### Why (T<1) makes it sharper

Softmax is:

[
p_j = \frac{e^{z_j/T}}{\sum_i e^{z_i/T}}
]

When (T<1), dividing by (T) **scales logits up** (multiplies by (1/T>1)). That means:

* higher logits get **much higher** exponentials
* lower logits get **much lower** exponentials *relative to the top one*

Softmax is dominated by **differences**. A useful rewrite (also what implementations do for stability) is subtracting the max logit (m=\max_i z_i):

[
p_j = \frac{e^{(z_j-m)/T}}{\sum_i e^{(z_i-m)/T}}
]

Now notice:

* (z_j - m \le 0) for all (j)
* for non-top tokens, (z_j-m) is **negative**
* dividing a negative number by a small (T) makes it **more negative**
  [
  \frac{z_j-m}{T} \text{ is more negative when } T<1
  ]
* (e^{\text{more negative}}) becomes **closer to 0**

So yes: **non-top tokens’ contributions shrink toward 0 after softmax**, but the reason is: **their distance below the max gets amplified**.

### Tiny numeric example

Say two logits: top is (2), second is (0).

* (T=1): exponentials are (e^2) and (e^0) → ratio (e^{2}\approx 7.4)
* (T=0.5): logits become (4) and (0) → ratio (e^{4}\approx 54.6)

So the top token becomes *much* more dominant → sharper distribution.

### Important nuance

If a “small” logit is **positive**, dividing by (T<1) makes it **bigger**, not smaller. But it still usually becomes **less competitive** if it’s not the max, because the max logit also got boosted and the **gap** got amplified.

If you want, paste a specific logit vector (like 5 numbers) and a temperature, and I’ll compute the before/after probabilities so you can see the sharpening concretely.


## Common Values of Temperature
Most people keep temperature in **0 to ~1.2**.

A practical “normal” range by goal:

* **0.0** (or ~0.1 if the API disallows 0): *most deterministic* (greedy-ish), best for **factual/precise**, extraction, and code completion.
* **0.2–0.4**: still pretty stable, good for **QA**, summaries, “don’t make stuff up.”
* **0.6–0.9**: common default zone for **general chat**—natural but not too wild.
* **1.0–1.2**: more variety/creativity; higher risk of drifting or making things up.
* **>1.2**: usually **chaotic** unless you heavily constrain with **top-p/top-k**; hallucinations and incoherence rise fast.

Notes:

* Temperature matters most when you **sample** (top-k/top-p + multinomial). If you do **argmax/greedy**, temperature won’t change outputs much (rank unchanged).
* With **top-p** (e.g., 0.9), you can often use **0.7–1.0** and still stay coherent.

If you tell me your use case (factual Q&A vs creative writing vs coding), I’ll suggest a good `(temperature, top_p/top_k)` combo.
