# What is TensorFloat32
TensorFloat32 (TF32) is a **specialized 19-bit floating-point format** designed by NVIDIA for accelerating AI workloads, distinct from both 32-bit single-precision (FP32) and 16-bit half-precision (FP16/BF16) formats. Here’s a breakdown of the key differences:

### 1. **TF32 Format Specifications**
TF32 uses:
- **1 sign bit**,
- **8 exponent bits** (same as FP32, enabling the same dynamic range as FP32: ~10⁻³⁸ to 10³⁸),
- **10 mantissa bits** (far fewer than FP32’s 23 bits, but more than FP16’s 10 bits for mantissa *plus* exponent differences).  

In total, TF32 is a 19-bit format (though stored as 32 bits for compatibility with FP32 memory layouts).

### 2. **Comparison with FP16/BF16 (Half Precision)**
| Aspect                | TF32                  | FP16 (Half Precision) | BF16 (Brain Float 16) |
|-----------------------|-----------------------|-----------------------|-----------------------|
| **Total Bits**        | 19 (stored as 32)     | 16                    | 16                    |
| **Exponent Bits**     | 8 (same as FP32)      | 5                     | 8 (same as FP32)      |
| **Mantissa Bits**     | 10                    | 10                    | 7                     |
| **Dynamic Range**     | Same as FP32          | Narrow (10⁻⁴ to 65504)| Same as FP32          |
| **Precision**         | ~3 decimal digits     | ~3 decimal digits     | ~2 decimal digits     |
| **Hardware Support**  | NVIDIA Ampere/Hopper  | Most modern GPUs      | NVIDIA Turing/Ampere, AMD, Intel |

### 3. **Why TF32 Is Not Half Precision**
- **Dynamic Range**: FP16 has a much narrower dynamic range than TF32 (FP16 cannot represent very small or very large numbers that TF32/FP32 can). TF32 inherits FP32’s full dynamic range via its 8 exponent bits, which is a critical distinction from FP16.  
- **Memory Storage**: TF32 operations are *computed* in 19 bits but use FP32’s 32-bit memory layout (so no memory savings vs. FP32, unlike FP16/BF16 which cut memory usage in half).  
- **Design Goal**: TF32 is optimized for **speed of FP32 matrix operations** (e.g., `torch.matmul`, convolutions) by reducing mantissa precision *without sacrificing dynamic range*, whereas half-precision (FP16/BF16) is optimized for **memory efficiency and speed** (with tradeoffs in dynamic range/precision).  

### 4. **Practical Implications**
- TF32 is a **drop-in acceleration for FP32 workloads**: You can run FP32 code on Ampere/Hopper GPUs with TF32 enabled, and it will automatically speed up matrix operations (no need to convert data to half precision).  
- Half-precision (FP16/BF16) requires **explicit data type conversion** (e.g., `tensor.half()`) and may need techniques like mixed-precision training (e.g., `torch.cuda.amp`) to avoid underflow/overflow due to limited dynamic range.  

In short, TF32 is a **precision-reduced variant of FP32** (not half precision) designed for fast matrix math, while FP16/BF16 are true 16-bit half-precision formats optimized for memory efficiency.

# How to enable TF32
There are two sets of code snippets relate to controlling floating-point precision in CUDA matrix operations but differ in **scope, abstraction level, supported hardware, and precision behavior**. Below is a detailed breakdown of their differences:

### 1. **`torch.set_float32_matmul_precision('high')`**  
Introduced in **PyTorch 1.12+**, this is a **high-level, unified API** for configuring the precision of FP32 matrix multiplications (matmul) on CUDA devices. It abstracts low-level hardware-specific settings (e.g., TF32, FP32) and is designed to simplify precision management across different GPU architectures.  

#### Key Details:  
- **Parameter Options**:  
  - `'high'`: Default for most GPUs. Prioritizes speed while maintaining "high" precision (varies by architecture):  
    - **Ampere/Hopper GPUs (A100, H100, RTX 3090/4090)**: Uses **TF32** for matmul (if supported) or FP16/BF16 (if TF32 is unavailable, e.g., RTX 4090).  
    - **Pre-Ampere GPUs (V100, GTX 1080)**: Uses pure **FP32** (no TF32 support).  
  - `'medium'`: Balances speed and precision (e.g., uses BF16 on Ampere/Hopper, FP16 on older GPUs).  
  - `'highest'`: Enforces pure **FP32** (disables TF32/BF16) for maximum precision (slowest).  

- **Scope**:  
  Applies **only to CUDA matrix multiplication operations** (e.g., `torch.matmul`, `nn.Linear`, convolution layers that rely on matmul). It does **not** affect other cuDNN operations (e.g., pooling, activation functions) or non-matmul CUDA kernels.  

- **Abstraction**:  
  Hides hardware-specific details (e.g., whether TF32 is supported). For example, on an RTX 4090 (no TF32), `'high'` will fall back to FP16/BF16 instead of TF32, whereas on an A100, it will use TF32.  

### 2. **`torch.backends.cuda.matmul.allow_tf32 = True` + `torch.backends.cudnn.allow_tf32 = True`**  
These are **low-level, hardware-specific flags** introduced in PyTorch 1.7 (alongside NVIDIA’s Ampere architecture) to explicitly enable/disable **TF32 (TensorFloat32)** for CUDA matmul and cuDNN operations.  

#### Key Details:  
- **`torch.backends.cuda.matmul.allow_tf32`**:  
  Enables TF32 **only for CUDA matrix multiplication kernels** (e.g., `torch.matmul`, `nn.Linear`). If `True`, Ampere/Hopper GPUs will use TF32 for FP32 matmul; pre-Ampere GPUs ignore this flag (no TF32 support).  

- **`torch.backends.cudnn.allow_tf32`**:  
  Enables TF32 **for cuDNN-accelerated operations** (e.g., convolutions, pooling, batch normalization). This is relevant for computer vision models (e.g., CNNs) and some NLP models (e.g., transformers with convolutional layers).  

- **Scope**:  
  - Explicitly targets **TF32** (only supported on Ampere/Hopper GPUs). These flags have no effect on GPUs without TF32 (e.g., RTX 4090, V100, GTX 1080).  
  - `allow_tf32 = True` does **not** enforce TF32; it allows PyTorch to use TF32 *when beneficial* (e.g., for FP32 inputs). Pure FP32 is still used if `allow_tf32 = False`.  

- **Hardware Dependency**:  
  These flags are **meaningless on non-Ampere/Hopper GPUs** (e.g., RTX 4090, V100) because TF32 is a hardware feature exclusive to Ampere (2020) and Hopper (2022) architectures.  

### Core Differences Summary  
| Aspect                  | `torch.set_float32_matmul_precision('high')` | `torch.backends.cuda.matmul/cudnn.allow_tf32 = True` |  
|-------------------------|-----------------------------------------------|-------------------------------------------------------|  
| **Abstraction Level**   | High-level (architecture-agnostic)            | Low-level (hardware-specific to TF32)                 |  
| **Supported Hardware**  | All CUDA GPUs (falls back to relevant precision) | Only Ampere/Hopper GPUs (TF32 support required)       |  
| **Scope**               | Only CUDA matmul operations                   | `cuda.matmul`: CUDA matmul; `cudnn`: cuDNN ops        |  
| **Precision Control**   | Chooses precision *category* (high/medium/highest) | Explicitly enables/disables TF32 (binary flag)        |  
| **Fallback Behavior**   | Uses FP16/BF16/FP32 if TF32 is unavailable     | Ignored if TF32 is unavailable (no fallback)          |  

### Practical Usage Recommendations  
- **For most users (Ampere/Hopper GPUs)**:  
  `torch.set_float32_matmul_precision('high')` is preferred because it is simpler and architecture-agnostic. It will automatically enable TF32 (on supported GPUs) and handle fallbacks on other GPUs.  

- **For fine-grained control (e.g., CNNs/NLP mixed with cuDNN ops)**:  
  Use `allow_tf32 = True` for both `cuda.matmul` and `cudnn` if you need TF32 for cuDNN operations (e.g., convolutions) *in addition to* matmul. Note that `set_float32_matmul_precision` does **not** affect cuDNN ops, so you may need both settings if using CNNs on Ampere/Hopper GPUs.  

- **For non-Ampere/Hopper GPUs (e.g., RTX 4090, V100)**:  
  `allow_tf32` flags have no effect, so use `set_float32_matmul_precision` to control precision (e.g., `'high'` will use FP16/BF16 on RTX 4090).  

### Example: Combined Usage (Ampere/Hopper GPUs)  
If you want TF32 for both matmul and cuDNN ops *and* use the high-level precision setting:  
```python
import torch

# High-level precision for matmul (enables TF32 on Ampere/Hopper)
torch.set_float32_matmul_precision('high')

# Explicitly enable TF32 for cuDNN ops (e.g., convolutions)
torch.backends.cudnn.allow_tf32 = True
```  
This is common for transformer models (matmul-heavy) with convolutional layers (cuDNN-heavy) on A100/H100 GPUs.  

In summary, the high-level API (`set_float32_matmul_precision`) is more user-friendly and portable, while the low-level flags (`allow_tf32`) offer explicit control over TF32 for specific operations on supported hardware.

# Hardware that supports TF32

### 1. **RTX 4090 (Ada Lovelace architecture)**  
The RTX 4090 is based on NVIDIA’s Ada Lovelace architecture and **does NOT support TF32** (TensorFloat32). TF32 is a feature primarily designed for NVIDIA’s Ampere architecture (e.g., A100, RTX 3090) and Hopper architecture (e.g., H100) GPUs, optimized for AI training/inference workloads.  

The RTX 4090 supports **FP16**, **BF16**, and **FP8** (for AI tasks), as well as NVIDIA’s **FP32** and **Tensor Cores optimized for FP16/BF16**, but it lacks hardware support for TF32.  

### 2. **A100/A30/A10 (Ampere architecture)**  
NVIDIA’s Ampere architecture GPUs (e.g., A100, A30, A10, RTX 3090/3080) **natively support TF32**. TF32 was first introduced with the Ampere architecture in 2020, designed to accelerate matrix operations (e.g., CUDA Core Tensor Cores) for AI workloads by combining the speed of FP16 with the dynamic range of FP32 (19 bits of precision, 8 bits of exponent).  

### 3. **H100 (Hopper architecture)**  
The H100 (Hopper architecture) further enhances TF32 with **TF32 v2** (improved precision) and also supports FP8, which is more efficient for large-scale AI training.  

### How to enable TF32 (for supported GPUs)  
If you are using an Ampere/Hopper architecture GPU (e.g., A100, RTX 3090, H100), you can enable TF32 in frameworks like PyTorch/TensorFlow:  
- **PyTorch**:  
  ```python
  import torch
  torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for matrix multiplication
  torch.backends.cudnn.allow_tf32 = True        # Enable TF32 for cuDNN operations
  ```  
- **TensorFlow**:  
  ```python
  import tensorflow as tf
  tf.config.optimizer.set_experimental_options({'tensor_float_32_execution': True})
  ```  

### Summary  
- If you meant **RTX 4090**: It does **not** support TF32.  
- If you meant **A100/A30/RTX 3090** (Ampere architecture): They **do** support TF32.  

If you require TF32 acceleration for language model training, it is recommended to choose Ampere (A100/A30/RTX 3090) or Hopper (H100) architecture GPUs. The RTX 4090, while powerful for consumer AI tasks, relies on FP16/BF16 instead of TF32.