# Running Computations in Parallel: DDP, FSDP and MapReduce

## Distributed Data Parallel
### 1. Is DDP an implementation of MapReduce?
To answer your question clearly: **DDP (Distributed Data Parallel) does leverage core ideas of MapReduce, but it is not a strict implementation of MapReduce—instead, it adapts and extends these ideas to fit the iterative, gradient-based nature of deep learning training across multiple GPUs/processes.**

Let’s break down the connection, differences, and core mechanics step by step:

---

### 2. Core Parallels Between DDP and MapReduce
MapReduce has two stages:  
- **Map**: Split data, process subsets in parallel.  
- **Reduce**: Aggregate results from parallel workers to produce a final output.  

DDP aligns with this high-level pattern for gradient computation and model synchronization:

#### Step 1: "Map" Stage in DDP
- **Data Splitting**: The training dataset is partitioned (via `DistributedSampler` in PyTorch) into disjoint subsets, with each GPU/process handling one subset (no overlap).  
- **Local Computation**: Each GPU independently:  
  - Feeds its data subset through the local copy of the model to compute forward passes (loss).  
  - Runs backward passes to calculate gradients of the loss with respect to model parameters (local gradients).  

This is the "Map" step: parallel processing of data subsets to generate intermediate results (gradients).

#### Step 2: "Reduce" Stage in DDP
- **Gradient Aggregation**: DDP uses **AllReduce** (a collective communication primitive) to aggregate (sum/average) local gradients across all GPUs. Unlike classic MapReduce (which uses a single "reducer" node), AllReduce is a symmetric operation: every GPU sends its local gradients and receives the aggregated (global) gradients.  
- **Parameter Update**: All GPUs then use the identical aggregated gradients to update their local model parameters (via the optimizer). Since all GPUs start with the same model weights and aggregate gradients identically, their parameters remain synchronized after each step.  

This is the "Reduce" step: aggregating parallel intermediate results (gradients) to produce a global result (synced gradients for parameter updates).

---

### 3. Key Differences (DDP ≠ Strict MapReduce)
While the high-level "split-process-aggregate" flow matches MapReduce, DDP diverges in critical ways due to the iterative nature of deep learning:

| Aspect                | Classic MapReduce               | DDP                              |
|-----------------------|----------------------------------|----------------------------------|
| **Execution Model**   | One-shot (batch processing)      | Iterative (epoch/batch loop)     |
| **Communication**     | Asymmetric (Map → Shuffle → Reduce to single node) | Symmetric (AllReduce: all nodes send/receive) |
| **State Persistence** | No shared state between workers  | All workers maintain identical copies of the model (state sync at every step) |
| **Reduction Goal**    | Final output (e.g., sum/count)   | Synced gradients for parameter updates (intermediate step, not final output) |
| **Fault Tolerance**   | Resilient to worker failures (recompute) | Less tolerant (requires all workers to progress in lockstep) |

---

### 4. Why DDP Uses AllReduce Instead of "Classic Reduce"
Classic MapReduce’s "reduce-to-single-node" would be inefficient for deep learning:  
- If gradients were sent to a single GPU (reducer) to aggregate and broadcast back, this would create a **bottleneck** (the reducer GPU becomes a single point of failure and latency).  
- AllReduce (e.g., via NCCL for GPUs) is optimized for multi-GPU clusters: it uses ring-based communication to aggregate gradients in O(log N) time (N = number of GPUs) with minimal latency, and all GPUs end up with the aggregated result simultaneously.

---

### 5. Simplified DDP Workflow (MapReduce Analogy)
```
# Map phase (per GPU)
1. Split dataset → GPU 0: batch 0-3, GPU 1: batch 4-7, ... (no overlap)
2. Local forward pass → compute loss for local batch
3. Local backward pass → compute local gradients

# Reduce phase (all GPUs)
4. AllReduce: sum local gradients across all GPUs → global gradients
5. All GPUs update model parameters with global gradients (sync maintained)

# Repeat for next batch (iterative MapReduce)
```

---

### 6. Conclusion
- **DDP is conceptually rooted in MapReduce**: it splits data (Map) across GPUs, computes local gradients, aggregates them (Reduce), and updates the model.  
- **DDP is not classic MapReduce**: it is iterative (not one-shot), uses symmetric AllReduce (not asymmetric reduce-to-single-node), and maintains synchronized model state across all GPUs—critical adaptations for low-latency, scalable deep learning training.

In short: *DDP is a specialized, iterative MapReduce variant optimized for multi-GPU gradient synchronization in deep learning.*

# The AllReduce step that synchronize the gradients across GPUs
### Ring-Based AllReduce: How GPUs Synchronize Gradients  
Ring-based AllReduce is the **de facto standard** for gradient synchronization in multi-GPU DDP (implemented via libraries like NCCL for NVIDIA GPUs). It solves the key problem of aggregating gradients across GPUs with minimal latency and bandwidth usage—far more efficiently than naive "all-to-one-then-one-to-all" approaches.  

Let’s break this down step by step, with concrete examples, terminology, and a play-by-play of how gradients flow between GPUs.


## Key Pre-Requisites (Before AllReduce Starts)
First, let’s set the stage for the AllReduce step:  
1. **Model Initialization**: All GPUs start with identical copies of the model (weights, biases, etc.)—this is enforced by DDP during setup (e.g., broadcasting the root GPU’s model to all others).  
2. **Local Gradients**: Each GPU computes its own local gradients for the batch it processed (e.g., GPU 0 has gradients `g0`, GPU 1 has `g1`, ..., GPU N-1 has `gN-1`).  
3. **Gradient Partitioning**: The full gradient tensor (e.g., for a model with 100M parameters) is split into **equal, non-overlapping chunks** (let’s call each chunk a "slice") across the GPUs. For N GPUs, we split the gradient into N slices: `S0, S1, ..., SN-1`.  

   - Example: 4 GPUs → gradient split into 4 slices (S0, S1, S2, S3). Each GPU is assigned a "home slice" (GPU 0 owns S0, GPU 1 owns S1, etc.)—this is critical for the ring protocol.  
4. **Ring Topology**: GPUs are arranged in a logical ring (e.g., GPU 0 → GPU 1 → GPU 2 → GPU 3 → GPU 0). Each GPU has exactly one **successor** (the next GPU in the ring) and one **predecessor** (the previous GPU).  

   <img src="https://i.imgur.com/8Z7X7a0.png" width="400">  
   *Logical ring for 4 GPUs: 0 → 1 → 2 → 3 → 0*


## Two Core Phases of Ring-Based AllReduce
Ring AllReduce has **two sequential phases** (both executed in parallel across the ring) to aggregate gradients:  
### Phase 1: Scatter-Reduce (Sum the Gradient Slices)  
### Phase 2: AllGather (Broadcast the Aggregated Slices)  

By splitting the work into these two phases, we avoid bottlenecks and minimize total data transfer (each GPU only sends/receives N-1 chunks, vs. N chunks in naive approaches).

---

### Let’s Use a Concrete Example: 4 GPUs (0,1,2,3)
We’ll track 4 gradient slices (S0, S1, S2, S3), where:  
- GPU 0 has local gradients for S0: `g0(S0)`; S1: `g0(S1)`; S2: `g0(S2)`; S3: `g0(S3)`  
- GPU 1 has local gradients for S0: `g1(S0)`; S1: `g1(S1)`; S2: `g1(S2)`; S3: `g1(S3)`  
- GPU 2 has local gradients for S0: `g2(S0)`; S1: `g2(S1)`; S2: `g2(S2)`; S3: `g2(S3)`  
- GPU 3 has local gradients for S0: `g3(S0)`; S1: `g3(S1)`; S2: `g3(S2)`; S3: `g3(S3)`  

Our goal: Every GPU ends up with the **sum of all local gradients** for every slice (e.g., `S0_total = g0(S0) + g1(S0) + g2(S0) + g3(S0)`).

---

### Phase 1: Scatter-Reduce (Summing Slices Across the Ring)
In this phase, GPUs pass gradient slices around the ring and **sum them in place**—each GPU only processes its assigned slices, and no slice is duplicated.  

The phase runs for **N-1 steps** (3 steps for 4 GPUs), with each step involving one send/receive between adjacent GPUs:  

#### Step 1 of Scatter-Reduce:
- GPU 0 sends its slice S0 to GPU 1; GPU 1 receives S0 and sums it with its own local S0 (`g1(S0) += g0(S0)`).  
- GPU 1 sends its slice S1 to GPU 2; GPU 2 receives S1 and sums it with its own local S1 (`g2(S1) += g1(S1)`).  
- GPU 2 sends its slice S2 to GPU 3; GPU 3 receives S2 and sums it with its own local S2 (`g3(S2) += g2(S2)`).  
- GPU 3 sends its slice S3 to GPU 0; GPU 0 receives S3 and sums it with its own local S3 (`g0(S3) += g3(S3)`).  

#### Step 2 of Scatter-Reduce:
- GPU 0 (now holding summed S3) sends S3 to GPU 1; GPU 1 sums it with its local S3 (`g1(S3) += g0(S3)`).  
- GPU 1 (now holding summed S0) sends S0 to GPU 2; GPU 2 sums it with its local S0 (`g2(S0) += g1(S0)`).  
- GPU 2 (now holding summed S1) sends S1 to GPU 3; GPU 3 sums it with its local S1 (`g3(S1) += g2(S1)`).  
- GPU 3 (now holding summed S2) sends S2 to GPU 0; GPU 0 sums it with its local S2 (`g0(S2) += g3(S2)`).  

#### Step 3 of Scatter-Reduce:
- GPU 0 (now holding summed S2) sends S2 to GPU 1; GPU 1 sums it with its local S2 (`g1(S2) += g0(S2)`).  
- GPU 1 (now holding summed S3) sends S3 to GPU 2; GPU 2 sums it with its local S3 (`g2(S3) += g1(S3)`).  
- GPU 2 (now holding summed S0) sends S0 to GPU 3; GPU 3 sums it with its local S0 (`g3(S0) += g2(S0)`).  
- GPU 3 (now holding summed S1) sends S1 to GPU 0; GPU 0 sums it with its local S1 (`g0(S1) += g3(S1)`).  

#### Result of Scatter-Reduce:
After N-1 steps, **each GPU now holds the FULL SUM of exactly one slice** (its "home slice"):  
- GPU 0 has `S0_total = g0(S0)+g1(S0)+g2(S0)+g3(S0)`  
- GPU 1 has `S1_total = g0(S1)+g1(S1)+g2(S1)+g3(S1)`  
- GPU 2 has `S2_total = g0(S2)+g1(S2)+g2(S2)+g3(S2)`  
- GPU 3 has `S3_total = g0(S3)+g1(S3)+g2(S3)+g3(S3)`  

All other slices on each GPU are still partial sums—we fix this in Phase 2.

---

### Phase 2: AllGather (Broadcasting Aggregated Slices)
In this phase, GPUs pass the **fully summed home slices** around the ring so that every GPU ends up with all summed slices.  

Like Scatter-Reduce, this phase runs for **N-1 steps** (3 steps for 4 GPUs):  

#### Step 1 of AllGather:
- GPU 0 sends S0_total to GPU 1; GPU 1 now has S0_total + S1_total.  
- GPU 1 sends S1_total to GPU 2; GPU 2 now has S1_total + S2_total.  
- GPU 2 sends S2_total to GPU 3; GPU 3 now has S2_total + S3_total.  
- GPU 3 sends S3_total to GPU 0; GPU 0 now has S3_total + S0_total.  

#### Step 2 of AllGather:
- GPU 0 sends S3_total (from Step 1) to GPU 1; GPU 1 now has S0_total + S1_total + S3_total.  
- GPU 1 sends S0_total (from Step 1) to GPU 2; GPU 2 now has S1_total + S2_total + S0_total.  
- GPU 2 sends S1_total (from Step 1) to GPU 3; GPU 3 now has S2_total + S3_total + S1_total.  
- GPU 3 sends S2_total (from Step 1) to GPU 0; GPU 0 now has S3_total + S0_total + S2_total.  

#### Step 3 of AllGather:
- GPU 0 sends S2_total (from Step 2) to GPU 1; GPU 1 now has all summed slices (S0_total, S1_total, S2_total, S3_total).  
- GPU 1 sends S3_total (from Step 2) to GPU 2; GPU 2 now has all summed slices.  
- GPU 2 sends S0_total (from Step 2) to GPU 3; GPU 3 now has all summed slices.  
- GPU 3 sends S1_total (from Step 2) to GPU 0; GPU 0 now has all summed slices.  

#### Final Result of AllReduce:
Every GPU now has the **complete, summed gradient tensor** (all slices S0_total to S3_total). All GPUs have identical gradients—this is the "sync" we need for DDP!


## Why Ring-Based AllReduce Is So Efficient
### 1. Minimal Bandwidth Usage
- Total data transferred per GPU: `2*(N-1)*G/N`, where G = total gradient size.  
  For N=8 GPUs, this is ~1.75G (vs. 7G for naive "all-to-one" reduce).  
- No GPU is a bottleneck (unlike a central reducer)—all GPUs send/receive in parallel.  

### 2. Low Latency
- The ring protocol runs in **O(N)** time (linear in the number of GPUs), but with a small constant factor (since communication is pipelined).  
- NCCL optimizes this further with hardware-level optimizations (e.g., direct GPU-to-GPU links via NVLink, overlapping compute/communication).  

### 3. Scalability
- Works seamlessly for 2, 4, 8, 16, or even 100+ GPUs (common in large-scale training clusters).  
- The ring topology is flexible (can be logical—doesn’t require physical GPU ordering).


## How This Integrates with DDP
1. After AllReduce completes, every GPU has the **global sum of gradients** (or average, if normalized by batch size).  
2. Each GPU runs the optimizer step (e.g., Adam, SGD) using the global gradients—since all GPUs have identical gradients and start with identical model weights, their model parameters remain perfectly synchronized after the update.  
3. This entire process repeats for every batch (iterative sync) throughout training.


## Edge Cases & Optimizations
- **Uneven Gradient Slices**: If the gradient size isn’t divisible by N, NCCL pads small slices to avoid fragmentation.  
- **Hybrid Rings**: For multi-node clusters (e.g., 8 GPUs per node × 4 nodes = 32 GPUs), NCCL uses a "hierarchical ring" (ring within nodes first, then ring across nodes) to minimize cross-node network traffic.  
- **Overlap with Compute**: Modern frameworks (PyTorch DDP) overlap the AllReduce communication with the next batch’s forward pass (when possible) to hide latency.


## Summary of GPU Sync in Ring AllReduce
1. **Partition**: Split gradients into N slices (one per GPU).  
2. **Scatter-Reduce**: Pass slices around the ring, summing them in N-1 steps—each GPU ends up with the full sum of its home slice.  
3. **AllGather**: Pass the summed home slices around the ring in N-1 steps—every GPU collects all summed slices.  
4. **Sync**: All GPUs now have identical global gradients, so parameter updates stay in lockstep.  

This is why DDP with ring-based AllReduce is the gold standard for multi-GPU training: it’s fast, scalable, and ensures perfect synchronization of gradients across all GPUs.

# FSDP
Yes, **FSDP (Fully Sharded Data Parallel)** is closely related to DDP (Distributed Data Parallel)—both are PyTorch-native frameworks for multi-GPU/multi-node distributed training, and both follow the "data parallel" paradigm (splitting data across workers, synchronizing model state). However, FSDP is a **more advanced, memory-efficient evolution of DDP** designed to scale to very large models (e.g., 10B+ parameters) that would not fit in the memory of a single GPU.

Let’s break down their relationship, key similarities, critical differences, and when to use each:

---

## Core Relationship: FSDP Builds on DDP’s Foundation
FSDP inherits DDP’s core data parallel logic (split data → local forward/backward → sync gradients/parameters) but reimagines **how model parameters/gradients/optimizers are stored and synchronized** to reduce memory usage. 

At a high level:
- DDP: Each GPU holds a **full copy of the model, gradients, and optimizer state** (only gradients are sharded/synced via AllReduce).
- FSDP: Each GPU holds only a **shard (slice) of the model, gradients, and optimizer state** (all three are sharded across GPUs, with minimal replication).

---

## Key Similarities Between DDP and FSDP
| Aspect                | DDP                              | FSDP                              |
|-----------------------|----------------------------------|----------------------------------|
| **Parallelism Type**  | Data Parallel (split data across workers) | Data Parallel (same data-splitting logic as DDP) |
| **Synchronization Primitive** | Uses AllReduce (ring-based) for gradient sync | Uses AllGather/ReduceScatter (extensions of AllReduce) for sharded parameter/gradient sync |
| **Iterative Training** | Iterative batch processing (forward → backward → sync → update) | Identical iterative training loop (same DDP workflow, but with sharded state) |
| **Compatibility**     | Works with most PyTorch models/optimizers | Works with the same models/optimizers (with minor configs for sharding) |

---

## Critical Differences (Why FSDP Exists)
The biggest pain point DDP solves is **computational parallelism**, but it fails for large models (e.g., GPT-3, Llama 2) because:
- A single 7B-parameter model requires ~28GB of memory (FP32) or ~14GB (FP16)—too large for a single GPU (even with 24GB VRAM).
- DDP requires every GPU to store the full model, so 8 GPUs would use 8× the model memory (wasting resources).

FSDP fixes this with **full sharding** (sometimes called "ZeRO Stage 3"—inspired by Microsoft’s ZeRO optimizer):

| Aspect                | DDP                              | FSDP                              |
|-----------------------|----------------------------------|----------------------------------|
| **Model Storage**     | Full model copy per GPU          | Only a shard (1/N of model) per GPU (N = number of GPUs) |
| **Gradient Storage**  | Full gradients per GPU (summed via AllReduce) | Only a shard of gradients per GPU (summed via ReduceScatter) |
| **Optimizer State**   | Full optimizer state (e.g., Adam moments) per GPU | Only a shard of optimizer state per GPU |
| **Memory Footprint**  | O(G) per GPU (G = full model size) | O(G/N) per GPU (linear reduction with number of GPUs) |
| **Sync Primitives**   | AllReduce (gradients only)       | 1. AllGather (unshard parameters for forward pass)<br>2. ReduceScatter (shard/sum gradients for backward pass)<br>3. AllGather (unshard parameters for optimizer step) |
| **Overhead**          | Low (simple AllReduce)           | Slightly higher (extra AllGather/ReduceScatter), but offset by memory savings |
| **Use Case**          | Small/medium models (fits on 1 GPU) | Large models (too big for 1 GPU—7B+ parameters) |

### Visualizing the Difference
| DDP (Per GPU)               | FSDP (Per GPU)                   |
|------------------------------|----------------------------------|
| [Full Model] + [Full Gradients] + [Full Optimizer] | [1/8 Model] + [1/8 Gradients] + [1/8 Optimizer] |

---

## How FSDP Works (Simplified Workflow)
FSDP extends DDP’s logic with sharding/unsharding steps to fit large models:

### 1. Initialization
- The model is split into **shards** (e.g., 8 GPUs → 8 shards) and distributed across workers (each GPU gets 1 shard).
- No GPU holds the full model—only its shard.

### 2. Forward Pass (Per Batch)
- **AllGather**: GPUs collectively unshard the model (each GPU sends its shard to others) so every GPU has the full model for the forward pass.
- Local forward pass: Each GPU processes its data subset (same as DDP) to compute loss.
- **Shard**: After forward pass, the full model is discarded—only the local shard is kept (saves memory).

### 3. Backward Pass (Per Batch)
- **AllGather**: Unshard the model again for backward pass (to compute gradients for all parameters).
- Local backward pass: Compute gradients for the local shard (only 1/N of total gradients).
- **ReduceScatter**: Instead of AllReduce (sum full gradients), FSDP scatters gradient shards across GPUs and sums them in place (each GPU gets the summed shard of gradients).
- **Shard**: Discard full gradients—keep only the summed shard.

### 4. Optimizer Step
- **AllGather**: Unshard the model one last time to apply the summed gradient shards to the corresponding model shards.
- Local optimizer step: Update only the local model shard (1/N of parameters).
- **Shard**: Discard full model—keep updated shard for next iteration.

### Key Optimization: "Activation Checkpointing"
FSDP often pairs with activation checkpointing (discarding intermediate activations during forward pass) to further reduce memory usage—critical for 100B+ parameter models.

---

## When to Use DDP vs. FSDP
| Scenario                          | Choose DDP                      | Choose FSDP                      |
|-----------------------------------|---------------------------------|----------------------------------|
| Model size                        | Fits on a single GPU (e.g., ResNet, BERT-base) | Too large for a single GPU (e.g., Llama 2 7B, GPT-3) |
| Number of GPUs                    | Small clusters (2-8 GPUs)       | Large clusters (8+ GPUs)         |
| Priority                          | Speed (minimal overhead)        | Memory efficiency (fit large models) |
| Overhead Tolerance                | Low overhead preferred          | Tolerate minor overhead for memory savings |

---

## Practical Implementation Notes
- Both DDP and FSDP use PyTorch’s `torch.distributed` backend (NCCL for GPUs, Gloo for CPUs).
- FSDP requires minimal code changes from DDP (replace `DDP` with `FSDP` and add sharding configs):
  ```python
  # DDP Example
  model = DDP(model, device_ids=[local_rank])

  # FSDP Example
  model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)
  ```
- FSDP supports "mixed sharding" (e.g., shard model but not optimizer) for flexibility (ZeRO Stage 1/2).

---

## Summary
- **FSDP is a superset of DDP**: It retains DDP’s data parallel core but adds sharding of model/gradients/optimizer to enable training of large models.
- **DDP is simpler/faster** for small/medium models (fits on 1 GPU) — no sharding/unsharding overhead.
- **FSDP is necessary** for large models (7B+ parameters) — it reduces memory usage from O(G) to O(G/N) per GPU, making training feasible.
- Both use collective communication (AllReduce/AllGather/ReduceScatter) for synchronization—FSDP just uses more advanced primitives to handle sharding.

In short: DDP is for "scaling out" small models across GPUs, while FSDP is for "scaling up" large models to fit across GPUs.
