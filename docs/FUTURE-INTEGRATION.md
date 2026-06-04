# Future Integration: tile-cuda

## Current State
GPU-native CUDA kernels for tile field operations. Hand-optimized PTX for BLAKE2b hashing, warp-level dot products, warp-level softmax, and batch score updates. Targets Ada Lovelace (sm_89). Batch operations: 10M hashes/sec, 1M embeds/sec, 10B cosine comparisons/sec, 100M tiles/sec.

## Integration Opportunities

### With forgemaster GPU simulation
tile-cuda's kernels accelerate the Forgemaster's GPU simulation pipeline. Hash kernels for cell ID hashing, dot product for surprise computation, softmax for energy redistribution, and batch evolve for population updates. The Forgemaster dispatches these kernels across available GPUs.

### With ternary-cell GPU acceleration
ternary-cell's CellGrid is embarrassingly parallel. tile-cuda provides the GPU kernels for each tick phase: predict (dot product of prediction vs neighbor states), perceive (hash for state lookup), surprise (softmax for energy normalization), vibe (batch evolve for strategy updates), gc (vector search for nearest-neighbor pruning), conservation (batch score verification).

### With room-as-codespace GPU rooms
When a Codespace has GPU access (e.g., running on a DGX), tile-cuda's kernels accelerate room computation. Rooms without GPU fall back to CPU (lever-runner-carapace). The same room, different acceleration — GPU when available, CPU when not.

## Dormant Ideas Now Unlockable
The CUDA kernels were standalone benchmarks. Now ternary-cell provides the application and the Forgemaster provides the orchestration. The kernels go from benchmarks to production GPU code.

## Potential in Mature Systems
Every GPU in the fleet runs tile-cuda kernels. The Forgemaster selects the optimal kernel configuration per GPU (RTX 4050 vs Jetson vs DGX). Room computation is GPU-accelerated wherever possible, falling back gracefully to CPU.

## Cross-Pollination Ideas
- **tile-neon**: ARM NEON version for Jetson edge GPU rooms
- **tile-opencl**: OpenCL version for non-NVIDIA GPUs
- **ptx-bench**: Benchmarks validate tile-cuda's performance claims

## Dependencies for Next Steps
- Integration with ternary-cell tick cycle as GPU kernels
- Forgemaster kernel dispatch and scheduling
- Fallback to CPU kernels when GPU unavailable
