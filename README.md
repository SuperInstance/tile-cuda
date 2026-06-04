# tile-cuda — GPU-native CUDA kernels for tile field operations

High-performance CUDA kernels for hash-based tile field operations, built from the silicon up for NVIDIA GPUs (Ada Lovelace / sm_89 primary target).

## Kernels

| Kernel | Operation | Target throughput |
|--------|-----------|-------------------|
| `batch_hash` | BLAKE2b batch hashing (10K states in parallel) | 10M hashes/sec |
| `batch_embed` | Position-aware embedding (intents → 64-dim vectors) | 1M embeds/sec |
| `batch_cosine_search` | Cosine similarity search (1 query vs 100K DB) | 10B comparisons/sec |
| `batch_evolve` | Batch score update with atomic counters | 100M tiles/sec |
| `batch_svd` | Power-iteration SVD (Jacobi for small matrices) | 1K SVDs/sec |

## Hand-optimized PTX

| File | Purpose |
|------|---------|
| `ptx/hash.ptx` | BLAKE2b G-function using `.reg .u64`, `shf.l.wrap` rotation |
| `ptx/dot_product.ptx` | Warp-level dot product with `shfl.sync`, `fma.rn.f32` |
| `ptx/softmax.ptx` | Warp-level softmax with `ex2.approx`, `shfl.down` reduction |

## Requirements

- CUDA Toolkit 12.0+
- NVIDIA GPU with compute capability ≥ 7.5 (Turing / Ampere / Ada / Hopper)
- Primary target: RTX 4050 (sm_89, Ada Lovelace)

## Build

```bash
make              # Build library
make test         # Build and run tests
make bench        # Build and run benchmarks
make clean        # Clean build artifacts
```

## Architecture Support

Compiles for multiple architectures:
- **sm_75** — Turing (RTX 20-series, GTX 16-series)
- **sm_80** — Ampere (RTX 30-series, A100)
- **sm_89** — Ada Lovelace (RTX 40-series) — primary
- **sm_90** — Hopper (H100)

## Usage

```c
#include "tile_cuda.h"

// Hash 10K states
tile_batch_hash(d_states, d_hashes, 128, 10000, 0);

// Embed intents
tile_batch_embed(d_tokens, d_vectors, 16, 1000, 0);

// Search
tile_batch_cosine_search(d_query, d_db, d_indices, d_scores,
                         64, 100000, 1, 10, 0);

// Evolve
tile_batch_evolve(d_scores, d_rewards, d_wins, d_totals,
                  100000, 0.01f, 0.0f, 1.0f, 0);

// SVD
tile_batch_svd(d_matrices, d_U, d_S, d_Vt,
               10000, 10, 1000, 100, 1e-6f, 0);
```

## Error Handling

Every CUDA API call is checked. Use `CUDA_CHECK()` macro:

```c
CUDA_CHECK(tile_batch_hash(...));
```

Get last error string: `tile_cuda_last_error()`.

## License

MIT
