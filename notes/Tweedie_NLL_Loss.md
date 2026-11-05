Below is a **fully worked 3-tree toy example** of LightGBM-style boosting using the **Tweedie objective** (with variance power (p=1.5), a common choice).

I’ll keep it minimal but exact:

* 1 feature `x`
* 4 training points
* Each tree is a **stump** (depth=1, 2 leaves)
* **learning_rate (\eta=1)**, **L2 (\lambda=0)**, **split penalty (\gamma=0)**
* Log-link: $(\mu=\exp(F))$ where (F) is the model’s raw score

---

## Data

|  x |  y |
| -: | -: |
|  0 |  0 |
|  1 |  1 |
|  2 |  2 |
|  3 |  4 |

---

## Tweedie objective used by LightGBM (core math)

For variance power (p\in(1,2)), a common Tweedie loss (ignoring constants/dispersion) with log-link is:

[
\ell(F) ;=; \frac{\exp((2-p)F)}{2-p} ;-; y\cdot\frac{\exp((1-p)F)}{1-p}
]
with (\mu=\exp(F)).

Gradient and Hessian (w.r.t. (F)):

[
g ;=;\frac{\partial \ell}{\partial F} ;=; \exp((2-p)F);-;y\exp((1-p)F)
]
[
h ;=;\frac{\partial^2 \ell}{\partial F^2};=;(2-p)\exp((2-p)F);-;y(1-p)\exp((1-p)F)
]

For the common case (p=1.5):

* (2-p=0.5,; 1-p=-0.5)
* (\exp(0.5F)=\sqrt{\mu}), (\exp(-0.5F)=1/\sqrt{\mu})

So:
[
g = \sqrt{\mu} - \frac{y}{\sqrt{\mu}}, \quad -g = \frac{y}{\sqrt{\mu}} - \sqrt{\mu}
]
**Notice:** (-g) is **not** (y-\mu). That’s the key difference from MSE.

---

## Initialization (base score)

LightGBM typically starts with a constant (F_0) so that (\mu_0) is the mean of (y):

[
\mu_0=\bar y = 1.75,\quad F_0=\log(1.75)=0.559616
]

So initially every point has (\mu=1.75).

---

# Tree 1

### Step 1A) Compute per-point loss/grad/hess at (F_0)

|  x |  y |        F |   mu |     loss |         g |        -g |        h |
| -: | -: | -------: | ---: | -------: | --------: | --------: | -------: |
|  0 |  0 | 0.559616 | 1.75 | 2.645751 |  1.322876 | -1.322876 | 0.661438 |
|  1 |  1 | 0.559616 | 1.75 | 4.157609 |  0.566947 | -0.566947 | 1.039402 |
|  2 |  2 | 0.559616 | 1.75 | 5.669467 | -0.188982 |  0.188982 | 1.417367 |
|  3 |  4 | 0.559616 | 1.75 | 8.693183 | -1.700840 |  1.700840 | 2.173296 |

> Compare “residual” vs Tweedie negative gradient at start (just intuition):

* residual (y-\mu) = ([-1.75,,-0.75,,+0.25,,+2.25])
* Tweedie (-g) = ([-1.3229,,-0.5669,,+0.1890,,+1.7008])

They’re **similar sign**, but **not the same quantity**.

---

### Step 1B) Choose best split by “gain” (not entropy)

LightGBM evaluates candidate thresholds (here between sorted x’s: 0.5, 1.5, 2.5).
With (\lambda=0,\gamma=0), split gain is:

[
\text{Gain}=\frac{G_L^2}{H_L}+\frac{G_R^2}{H_R}-\frac{G^2}{H}
]
where (G=\sum g_i,; H=\sum h_i).

Candidate gains (computed from the table above):

* split at **0.5**: Gain = **3.0237**
* split at **1.5**: Gain = **3.0944**  ✅ best
* split at **2.5**: Gain = **2.2588**

So Tree 1 uses: **Left leaf = {x=0,1}**, **Right leaf = {x=2,3}**.

---

### Step 1C) Compute leaf outputs (Newton step)

Leaf value formula (with (\lambda=0)):
[
w = -\frac{G}{H}
]

**Left leaf (x ≤ 1.5):**

* (G_L = 1.889822)
* (H_L = 1.700840)
* (w_L = -G_L/H_L = -1.111111)

**Right leaf (x > 1.5):**

* (G_R = -1.889822)
* (H_R = 3.590662)
* (w_R = -G_R/H_R = +0.526316)

### Step 1D) Update predictions

[
F^{(1)} = F^{(0)} + w_{\text{leaf}}
\quad,\quad
\mu^{(1)}=\exp(F^{(1)})
]

After Tree 1:

* x=0,1: (F=0.559616-1.111111=-0.551495) → (\mu=0.576088)
* x=2,3: (F=0.559616+0.526316=1.085932) → (\mu=2.962198)

Total loss drops: **21.1660 → 19.5277**

---

# Tree 2 (recompute gradients at the NEW predictions)

### Step 2A) Per-point at current (F)

|  x |  y |         F |       mu |     loss |         g |        -g |        h |
| -: | -: | --------: | -------: | -------: | --------: | --------: | -------: |
|  0 |  0 | -0.551495 | 0.576088 | 1.518009 |  0.759004 | -0.759004 | 0.379502 |
|  1 |  1 | -0.551495 | 0.576088 | 4.153040 | -0.558511 |  0.558511 | 1.038260 |
|  2 |  2 |  1.085932 | 2.962198 | 5.766297 |  0.559059 | -0.559059 | 1.441574 |
|  3 |  4 |  1.085932 | 2.962198 | 8.090387 | -0.602986 |  0.602986 | 2.022597 |

### Step 2B) Use the same split (x ≤ 1.5) and compute leaf values

(Real LightGBM would re-search splits; I’m holding it fixed so you can follow arithmetic.)

Left leaf sums:

* (G_L=0.200494), (H_L=1.417762)
* (w_L=-G_L/H_L=-0.141415)

Right leaf sums:

* (G_R=-0.043928), (H_R=3.464171)
* (w_R=+0.012681)

### Step 2C) Update

After Tree 2:

* x=0,1: (F=-0.551495-0.141415=-0.692911) → (\mu=0.500118)
* x=2,3: (F=1.085932+0.012681=1.098612) → (\mu\approx 3.000000)

Total loss drops: **19.5277 → 19.5133**

---

# Tree 3

At this point gradients are already near-balanced within each leaf. One more Newton step is tiny:

* Left: (w_L\approx -0.000236)
* Right: (w_R\approx 1.7\times 10^{-7})

Final (rounded) after Tree 3:

|  x |  y |         F |  mu |         g |
| -: | -: | --------: | --: | --------: |
|  0 |  0 | -0.693147 | 0.5 |  0.707107 |
|  1 |  1 | -0.693147 | 0.5 | -0.707107 |
|  2 |  2 |  1.098612 | 3.0 |  0.577350 |
|  3 |  4 |  1.098612 | 3.0 | -0.577350 |

Total loss: **19.5132607** (barely changes after Tree 2)

---

## What this demonstrates (the core answers)

1. **Tweedie trains trees on gradients/Hessians of the Tweedie NLL**, not on residuals (y-\hat y).
2. For (p=1.5), the “pseudo-residual” is (-g=\frac{y}{\sqrt{\mu}}-\sqrt{\mu}) — **not** (y-\mu).
3. Trees are added as **Newton-style corrections in log-mean space** (F); final (\mu) is (\exp(\sum f_m(x))) (with learning rate/regularization in real runs).

If you want, I can do the *same walk-through* but with:

* **learning_rate < 1** (more realistic),
* **nonzero (\lambda)**,
* and letting Tree 2/3 **re-pick the best split** (still with only 4 points, so it remains hand-computable).
