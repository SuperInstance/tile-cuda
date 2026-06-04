#ifndef TILE_CUDA_H
#define TILE_CUDA_H

/*
 * tile_cuda.h — C API for tile field GPU operations.
 *
 * All functions return cudaError_t.  Callers should check with
 * CUDA_CHECK() (see macro below) or tile_cuda_last_error().
 */

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  Error handling                                                     */
/* ------------------------------------------------------------------ */

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t _err = (call);                                      \
        if (_err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error %s:%d: %s (%s)\n",             \
                    __FILE__, __LINE__,                                 \
                    cudaGetErrorString(_err),                           \
                    cudaGetErrorName(_err));                            \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

/* Returns the last CUDA error (thread-local). */
const char *tile_cuda_last_error(void);

/* ------------------------------------------------------------------ */
/*  Configuration                                                      */
/* ------------------------------------------------------------------ */

/* Batch sizes supported by the API. */
#define TILE_HASH_BATCH_MAX     10000000
#define TILE_EMBED_BATCH_MAX    1000000
#define TILE_SEARCH_DB_MAX      100000
#define TILE_EVOLVE_BATCH_MAX   1000000
#define TILE_SVD_M_MAX          10000
#define TILE_SVD_N_MAX          10

/* Embedding dimensions. */
#define TILE_EMBED_DIM          64

/* BLAKE2b digest length (bytes). */
#define TILE_HASH_LEN           64

/* ------------------------------------------------------------------ */
/*  1. batch_hash_kernel — parallel BLAKE2b hashing                    */
/* ------------------------------------------------------------------ */

/*
 * Hash `count` state strings in parallel on the GPU.
 *
 * d_states    : device pointer, `count * state_stride` bytes (raw bytes)
 * d_hashes    : device pointer, `count * TILE_HASH_LEN` bytes (output)
 * state_stride: bytes per state string (padded to 64-byte boundary recommended)
 * count       : number of items to hash
 * stream      : CUDA stream (0 for default)
 */
cudaError_t tile_batch_hash(const void       *d_states,
                            void             *d_hashes,
                            size_t            state_stride,
                            int               count,
                            cudaStream_t      stream);

/* ------------------------------------------------------------------ */
/*  2. batch_embed_kernel — position-aware embedding                   */
/* ------------------------------------------------------------------ */

/*
 * Embed `count` intent token-ids into TILE_EMBED_DIM-dim float vectors.
 *
 * d_token_ids : device int32, shape [count * max_tokens]
 * d_vectors   : device float32, shape [count * TILE_EMBED_DIM]
 * max_tokens  : max tokens per intent (padded)
 * count       : number of intents
 * stream      : CUDA stream
 */
cudaError_t tile_batch_embed(const int       *d_token_ids,
                             float           *d_vectors,
                             int              max_tokens,
                             int              count,
                             cudaStream_t     stream);

/* ------------------------------------------------------------------ */
/*  3. batch_cosine_search_kernel — similarity search                  */
/* ------------------------------------------------------------------ */

/*
 * Search `n_queries` query vectors against a database of `db_size` vectors.
 *
 * d_queries   : device float32, [n_queries * dim]
 * d_db        : device float32, [db_size * dim]
 * d_out_indices: device int32, [n_queries * top_k]
 * d_out_scores : device float32, [n_queries * top_k]
 * dim         : vector dimensionality (must be TILE_EMBED_DIM)
 * db_size     : number of database vectors
 * n_queries   : number of query vectors
 * top_k       : number of results per query
 * stream      : CUDA stream
 */
cudaError_t tile_batch_cosine_search(const float  *d_queries,
                                     const float  *d_db,
                                     int          *d_out_indices,
                                     float        *d_out_scores,
                                     int           dim,
                                     int           db_size,
                                     int           n_queries,
                                     int           top_k,
                                     cudaStream_t  stream);

/* ------------------------------------------------------------------ */
/*  4. batch_evolve_kernel — score update                              */
/* ------------------------------------------------------------------ */

/*
 * Update tile scores in batch.
 *
 * d_scores    : device float32, [count] — current scores (overwritten)
 * d_rewards   : device float32, [count] — reward signals
 * d_win_counts: device int32, [count]   — win counts (atomically updated)
 * d_total     : device int32, [count]   — total games (atomically updated)
 * count       : number of tiles
 * learning_rate: float
 * clamp_min   : minimum score
 * clamp_max   : maximum score
 * stream      : CUDA stream
 */
cudaError_t tile_batch_evolve(float          *d_scores,
                              const float    *d_rewards,
                              int            *d_win_counts,
                              int            *d_total,
                              int             count,
                              float           learning_rate,
                              float           clamp_min,
                              float           clamp_max,
                              cudaStream_t    stream);

/* ------------------------------------------------------------------ */
/*  5. batch_svd_kernel — power-iteration / Jacobi SVD                 */
/* ------------------------------------------------------------------ */

/*
 * Batch SVD for tall-thin matrices (rows × cols, cols ≤ TILE_SVD_N_MAX).
 * Uses Jacobi rotations for small matrices, cuBLAS for large.
 *
 * d_matrices  : device float32, [batch * rows * cols] (row-major)
 * d_U         : device float32, [batch * rows * cols] (output left singular)
 * d_S         : device float32, [batch * cols]        (output singular values)
 * d_Vt        : device float32, [batch * cols * cols]  (output right singular)
 * rows        : number of rows per matrix
 * cols        : number of columns per matrix
 * batch       : number of matrices
 * max_iters   : max power-iteration steps
 * tol         : convergence tolerance
 * stream      : CUDA stream
 */
cudaError_t tile_batch_svd(const float    *d_matrices,
                           float          *d_U,
                           float          *d_S,
                           float          *d_Vt,
                           int             rows,
                           int             cols,
                           int             batch,
                           int             max_iters,
                           float           tol,
                           cudaStream_t    stream);

#ifdef __cplusplus
}
#endif

#endif /* TILE_CUDA_H */
