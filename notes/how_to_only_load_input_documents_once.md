`@lru_cache(maxsize=1)` is a Python decorator that **memoizes** (caches) the function so that:

### ✔️ **The function runs only once**

### ✔️ **Future calls with the same arguments return instantly from cache**

### ✔️ **No re-reading from disk / re-tokenizing / re-doing expensive operations**

---

# 📌 What it does, practically

When you write:

```python
@lru_cache(maxsize=1)
def load_corpus(path: str) -> str:
    return Path(path).read_text()
```

The first time you call:

```python
load_corpus("tiny.txt")
```

it actually reads the file from disk.

The second, third, fourth, hundredth time you call:

```python
load_corpus("tiny.txt")
```

it **does NOT** read the file again.
It returns the stored value from memory.

---

# 💡 Why `maxsize=1`?

Because you only care about **one file path** (Tiny Shakespeare).
If you want to cache multiple files, you can increase the cache size.

---

# ✨ Why do we use it in data-loading for LMs?

Because:

1. **Disk reads are slow**
2. **Tokenization can be expensive** (thousands to millions of tokens)
3. You often call the function multiple times:

   * Train dataloader
   * Val dataloader
   * Debug code
   * Jupyter cell reruns

Without caching, you’d keep re-tokenizing or re-reading the text.

With caching, tokenization happens once per process.

---

# 🔍 Example: caching in action

```python
@lru_cache(maxsize=1)
def f(x):
    print("running...")
    return x * 2

print(f(3))  # prints "running...", returns 6
print(f(3))  # does NOT print "running...", returns cached 6
```

Output:

```
running...
6
6
```

Second call returns immediately.

---

# 🚀 Result: faster debugging, less waste

You can re-run your training loop or dataloader creation many times without recomputing everything.

It’s a very standard trick in:

* Language model training
* Expensive data preprocessing
* File IO

---

If you want, I can also explain:

* What happens if arguments differ
* How to clear the cache
* Thread-safety and multiprocessing considerations (important for DataLoader workers)
