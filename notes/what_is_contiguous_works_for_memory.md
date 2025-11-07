In the **CausalSelfAttention**, after we calculate the contextual embedding, we implement this line of code to transform its dimension from `[B, nh, T, hs]` back to `[B, T, C]`

```
y = y.transpose(1,2).contiguous().view(B,T,C)
```

But what does `contiguous()` do? The PyTorch wiki described it as "Returns a contiguous in memory tensor containing the same data as self tensor. If self tensor is already in the specified memory format, this function returns the self tensor."

Short version: after `transpose(1, 2)`, the tensor is **not contiguous in memory** (its strides change). `.view(B, T, C)` only works on **contiguous** tensors. Calling `.contiguous()` makes a contiguous copy so `view` can safely collapse the last two dims.

What’s going on:

* Typical shapes: after attention you have `y` of shape `(B, nh, T, hs)`.
* You swap head and time: `y = y.transpose(1, 2)` → shape `(B, T, nh, hs)`, but this is **strided** (non-contiguous).
* `view(B, T, C)` needs the memory to be laid out row-major as `(B, T, nh, hs)` so it can merge `nh*hs → C`. That’s not true after a plain transpose.
* `.contiguous()` materializes a contiguous buffer in the new order, making the subsequent `view(B, T, C)` legal.

Practical options:

* Keep your pattern (explicit and fast):

  ```python
  y = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
  ```
* Or use `.reshape`, which will act like `view` when possible and copy when needed (so you can skip the explicit `.contiguous()`):

  ```python
  y = (attn @ v).transpose(1, 2).reshape(B, T, C)
  ```
* Or with einops (very readable):

  ```python
  from einops import rearrange
  y = rearrange(attn @ v, 'b h t hs -> b t (h hs)')
  ```

Rule of thumb: after any `transpose/permute`, if you’ll `view` to merge/split dims, make it contiguous first (or use `reshape`).


Great instincts—this is exactly about **memory layout**.

### What “strided” means

Every PyTorch tensor has:

* a **size** (shape), and
* a **stride**: how many memory steps you move to advance by 1 along each dimension.

Example:

```python
x = torch.arange(12).view(3, 4)   # shape (3,4)
x.stride()  # e.g. (4, 1)  -> move 4 elements to go down a row, 1 to go right a col
x.is_contiguous()  # True
```

If you **transpose/permute/slice**, PyTorch usually returns a **view** that shares the same underlying memory but with **different strides**:

```python
y = x.t()          # shape (4,3)
y.stride()         # (1, 4)  <- swapped
y.is_contiguous()  # False    <- non-contiguous (strided)
```

A **non-contiguous** (a.k.a. strided) tensor is perfectly valid; it just means the memory isn’t laid out row-major for that view.

### Why `q @ k.transpose(2, 3)` works without `.contiguous()`

Matrix multiply (`@` / `torch.matmul`) supports **strided inputs**. BLAS/GEMM kernels can consume tensors with transposed strides (or PyTorch will pass appropriate transpose flags / create temporary fast paths). So you can do:

```python
attn_logits = q @ k.transpose(2, 3)  # fine: no contiguous() needed
```

Same story for many ops (`softmax`, `masked_fill`, etc.): they accept non-contiguous tensors.

### When you *do* need `.contiguous()`

Ops like **`.view(...)` require contiguous memory** because `view` is just a metadata reshape—no copy. After a `transpose/permute`, if you then want to **merge or split dimensions** via `view`, you must make the tensor contiguous first:

```python
y = y.transpose(1, 2).contiguous().view(B, T, C)
#              ^^^^^^^^^^^ make a contiguous copy so view is legal
```

Alternatively, use `.reshape(B, T, C)`, which will behave like `view` if possible, or **copy under the hood** if not:

```python
y = y.transpose(1, 2).reshape(B, T, C)  # contiguous copy if needed
```

### Handy checks

```python
t.is_contiguous()   # bool
t.stride()          # see the per-dimension strides
```

### TL;DR

* **Strided** = tensor uses non-default strides (often from transpose/permute/slice); memory isn’t row-major for that view.
* Most compute ops (including `@`) **handle strided tensors** just fine.
* `view` **requires contiguous** → call `.contiguous()` (or use `.reshape`) after a transpose/permute before viewing.

## What does "memory" means?
Love this question — getting “stride + memory” clear unlocks a lot.

### What “memory” means here

Think of a tensor as a **view** over a 1-D block of bytes in RAM (the **storage**). The tensor view tells PyTorch:

* where that block starts (**storage_offset**),
* how many elements to read along each axis (**size**), and
* how to jump through that storage to move 1 step along each axis (**stride**).

So “memory steps” = **how many elements you skip in the underlying 1-D storage** when you increment an index along a given dimension. (PyTorch reports strides **in elements**, not bytes. The byte jump would be `stride[d] * element_size_bytes`.)

### The formula

For an `n`-D tensor with indices `(i0, i1, …, in-1)`:

```
linear_index = storage_offset + Σ (ik * stride[k])
```

PyTorch uses row-major (C-order) by default, so a freshly created contiguous tensor has:

```
stride[k] = Π(size[k+1:])
```

Example for a contiguous `(3, 4, 5)` float tensor:

```
size   = (3, 4, 5)
stride = (20, 5, 1)      # because 4*5=20, 5=5, 1=1
```

* Move +1 along last dim → jump 1 element in memory.
* Move +1 along second dim → jump 5 elements.
* Move +1 along first dim → jump 20 elements.

### How this relates to your understanding

* **Size**: how many positions along each axis (the shape).
* **Stride**: how the shape is **mapped onto the 1-D storage**—i.e., the step (in elements) to move by 1 along that axis.
* Stride does **not** “change the size”; it tells how to **interpret** storage for a given size. Multiple different views (size/stride combos) can look at the **same** storage.

### Common transformations and their strides

* **`view/reshape` (no transpose)**: if possible, it just changes size/stride metadata (no data copy). Contiguity is preserved.
* **`transpose/permute`**: changes the order of axes → strides get **rearranged** (non-contiguous, but still just a view).
* **Slicing with a step** (e.g., `x[:, ::2]`): can change strides (e.g., double them).
* **`contiguous()` / `clone()`**: makes a new storage laid out in standard row-major order; strides become the canonical contiguous ones.

### Why `transpose` often forces `.contiguous()` before `view`

After `y = y.transpose(1, 2)`, the size might be `(B, T, nh, hs)` but the strides are reordered. To **merge** `nh*hs → C` using `view(B, T, C)`, the elements must be next to each other in memory in that merged order. They aren’t after a plain transpose, so you do:

```python
y = y.transpose(1, 2).contiguous().view(B, T, C)
# or
y = y.transpose(1, 2).reshape(B, T, C)  # copies if needed
```

### Strides in practice (quick REPL)

```python
x = torch.arange(12).view(3, 4)
x.size(), x.stride(), x.is_contiguous()
# ((3, 4), (4, 1), True)

y = x.t()
y.size(), y.stride(), y.is_contiguous()
# ((4, 3), (1, 4), False)  <- same storage, different strides

z = y.contiguous()
z.size(), z.stride(), z.is_contiguous()
# ((4, 3), (3, 1), True)
```

### Bytes vs elements

If `dtype=torch.float32` (4 bytes/element) and `stride=(20,5,1)`, then the **byte** step for each axis is `(80, 20, 4)` bytes respectively. PyTorch keeps strides in **elements** so it stays dtype-agnostic.

### TL;DR

* **Size** = shape; **stride** = step (in elements) in the 1-D storage to move along each axis.
* Tensors are views over a 1-D **storage**; `(size, stride, storage_offset)` define the view.
* Many ops work fine on **strided** (non-contiguous) tensors (e.g., matmul, softmax).
* `view` needs **contiguous** layout → call `.contiguous()` (or use `.reshape`) after `transpose/permute` before merging/splitting dims.
