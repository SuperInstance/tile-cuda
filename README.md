# tile-cuda — GPU-Native CUDA Kernels for Tile Field Operations

High-performance CUDA kernels for hash-based tile field operations, built from the silicon up for NVIDIA GPUs. Targets Ada Lovelace (sm_89) with multi-arch support from Turing through Hopper.

## What It Does

Five hand-optimized CUDA kernels form a complete tile processing pipeline:

1. **Batch Hash** — BLAKE2b hashing of 10K+ states in parallel, one thread per state
2. **Batch Embed** — Position-aware token embedding into 64-dim float vectors using shared memory accumulation
3. **Cosine Search** — Similarity search with warp-level `__shfl_down_sync` reductions and top-K selection
4. **Batch Evolve** — Atomic counter-based score updates with learning rate and clamping
5. **Batch SVD** — Power-iteration Jacobi SVD for small matrices

## Architecture

Each kernel follows the same pattern: a `__global__` function + a `launch_*` wrapper that configures grid/block dimensions and shared memory.

| File | Kernel | Technique |
|------|--------|-----------|
| `src/hash_kernel.cu` | BLAKE2b batch hash | Per-thread 12-round compression, constant memory for IV/sigma |
| `src/embed_kernel.cu` | Position-aware embedding | One block per intent, shared memory accumulation, `atomicAdd` |
| `src/search_kernel.cu` | Cosine similarity search | Warp-level `__shfl_down_sync` max-reduction, shared-memory top-K |
| `src/evolve_kernel.cu` | Score evolution | `atomicAdd` for win/total counters, `fmaf` for score update |
| `src/svd_kernel.cu` | Jacobi SVD | Power iteration with convergence check |

### Hand-Written PTX

Lower-level PTX snippets for critical inner loops:

| File | Purpose |
|------|---------|
| `ptx/hash.ptx` | BLAKE2b G-function using `.reg .u64`, `shf.l.wrap` rotation |
| `ptx/dot_product.ptx` | Warp dot product with `shfl.sync`, `fma.rn.f32` |
| `ptx/softmax.ptx` | Warp softmax with `ex2.approx`, `shfl.down` reduction |

## Key Types (C API)

```c
// Configuration constants (tile_cuda.h)
#define TILE_HASH_LEN    64    // BLAKE2b digest size (bytes)
#define TILE_EMBED_DIM   64    // Embedding dimensionality

// All functions return cudaError_t
cudaError_t tile_batch_hash(const void *states, void *hashes,
    size_t state_stride, int count, cudaStream_t stream);

cudaError_t tile_batch_embed(const int *token_ids, float *vectors,
    int max_tokens, int count, cudaStream_t stream);

cudaError_t tile_batch_cosine_search(const float *queries, const float *db,
    int *out_indices, float *out_scores,
    int dim, int db_size, int n_queries, int top_k, cudaStream_t stream);

cudaError_t tile_batch_evolve(float *scores, const float *rewards,
    int *win_counts, int *total, int count,
    float lr, float clamp_min, float clamp_max, cudaStream_t stream);

cudaError_t tile_batch_svd(const float *matrices, float *U, float *S, float *Vt,
    int rows, int cols, int batch, int max_iters, float tol, cudaStream_t stream);
```

## Installation

### Requirements

- CUDA Toolkit 12.0+
- NVIDIA GPU with compute capability ≥ 7.5 (Turing / Ampere / Ada / Hopper)
- Primary target: RTX 4050 (sm_89, Ada Lovelace)

### Build

```bash
make              # Build static library (build/libtile_cuda.a)
make test         # Build and run CUDA test suite
make bench        # Build and run benchmarks
make clean        # Clean build artifacts
```

Multi-arch compilation generates PTX for sm_75, sm_80, sm_89, and sm_90.

## Usage

```c
#include "tile_cuda.h"

// Hash 10K states
CUDA_CHECK(tile_batch_hash(d_states, d_hashes, 128, 10000, 0));

// Embed 1K intents
CUDA_CHECK(tile_batch_embed(d_tokens, d_vectors, 16, 1000, 0));

// Search: 1 query vs 100K database, top-10
CUDA_CHECK(tile_batch_cosine_search(d_query, d_db, d_indices, d_scores,
    64, 100000, 1, 10, 0));

// Evolve 100K tile scores
CUDA_CHECK(tile_batch_evolve(d_scores, d_rewards, d_wins, d_totals,
    100000, 0.01f, 0.0f, 1.0f, 0));

// Batch SVD: 1K matrices (10x10)
CUDA_CHECK(tile_batch_svd(d_matrices, d_U, d_S, d_Vt,
    10, 10, 1000, 100, 1e-6f, 0));
```

### Error Handling

Every CUDA API call is checked via the `CUDA_CHECK()` macro. Last error string: `tile_cuda_last_error()`.

## Target Throughput (RTX 4050)

| Kernel | Target |
|--------|--------|
| Batch hash | 10M hashes/sec |
| Batch embed | 1M embeds/sec |
| Cosine search | 10B comparisons/sec |
| Batch evolve | 100M tiles/sec |
| Batch SVD | 1K SVDs/sec |

## Project Structure

```
tile-cuda/
├── include/tile_cuda.h      — C API header (all functions + constants)
├── src/
│   ├── hash_kernel.cu       — BLAKE2b batch hashing
│   ├── embed_kernel.cu      — Position-aware embedding
│   ├── search_kernel.cu     — Cosine similarity with warp reduction
│   ├── evolve_kernel.cu     — Score evolution with atomic counters
│   ├── svd_kernel.cu        — Jacobi SVD
│   └── tile_cuda.cu         — Entry point / error tracking
├── ptx/                     — Hand-written PTX snippets
├── tests/                   — CUDA test suite + cross-language tests
├── benches/                 — Benchmark suite
├── docs/                    — Documentation
└── Makefile                 — Multi-arch build system
```

## License

MIT
