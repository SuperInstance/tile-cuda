/*
 * tile_cuda.cu — Main CUDA implementation for tile field operations.
 *
 * Provides host-side launch wrappers for all kernels and common
 * utilities (error tracking, device property queries).
 */

#include "tile_cuda.h"
#include <stdio.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Error tracking                                                     */
/* ------------------------------------------------------------------ */

static __thread cudaError_t g_last_error = cudaSuccess;

const char *tile_cuda_last_error(void)
{
    if (g_last_error != cudaSuccess)
        return cudaGetErrorString(g_last_error);
    return "no error";
}

static void set_last_error(cudaError_t err)
{
    if (err != cudaSuccess) g_last_error = err;
}

/* ------------------------------------------------------------------ */
/*  Device helpers                                                     */
/* ------------------------------------------------------------------ */

static int current_sm(void)
{
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    return prop.major * 10 + prop.minor;
}

/* ------------------------------------------------------------------ */
/*  Kernel launch helpers                                              */
/* ------------------------------------------------------------------ */

/*
 * We forward-declare the __global__ entry points so this translation
 * unit can launch them.  The actual kernel bodies live in their
 * respective .cu files and get linked at compile time.
 */

/* hash_kernel.cu */
extern cudaError_t launch_hash_kernel(const void *d_states,
                                      void *d_hashes,
                                      size_t state_stride,
                                      int count,
                                      cudaStream_t stream);

/* embed_kernel.cu */
extern cudaError_t launch_embed_kernel(const int *d_token_ids,
                                       float *d_vectors,
                                       int max_tokens,
                                       int count,
                                       cudaStream_t stream);

/* search_kernel.cu */
extern cudaError_t launch_search_kernel(const float *d_queries,
                                        const float *d_db,
                                        int *d_out_indices,
                                        float *d_out_scores,
                                        int dim,
                                        int db_size,
                                        int n_queries,
                                        int top_k,
                                        cudaStream_t stream);

/* evolve_kernel.cu */
extern cudaError_t launch_evolve_kernel(float *d_scores,
                                        const float *d_rewards,
                                        int *d_win_counts,
                                        int *d_total,
                                        int count,
                                        float learning_rate,
                                        float clamp_min,
                                        float clamp_max,
                                        cudaStream_t stream);

/* svd_kernel.cu */
extern cudaError_t launch_svd_kernel(const float *d_matrices,
                                     float *d_U,
                                     float *d_S,
                                     float *d_Vt,
                                     int rows,
                                     int cols,
                                     int batch,
                                     int max_iters,
                                     float tol,
                                     cudaStream_t stream);

/* ------------------------------------------------------------------ */
/*  Public API — thin wrappers                                         */
/* ------------------------------------------------------------------ */

cudaError_t tile_batch_hash(const void   *d_states,
                            void         *d_hashes,
                            size_t        state_stride,
                            int           count,
                            cudaStream_t  stream)
{
    cudaError_t err = launch_hash_kernel(d_states, d_hashes,
                                         state_stride, count, stream);
    set_last_error(err);
    return err;
}

cudaError_t tile_batch_embed(const int   *d_token_ids,
                             float       *d_vectors,
                             int          max_tokens,
                             int          count,
                             cudaStream_t stream)
{
    cudaError_t err = launch_embed_kernel(d_token_ids, d_vectors,
                                          max_tokens, count, stream);
    set_last_error(err);
    return err;
}

cudaError_t tile_batch_cosine_search(const float  *d_queries,
                                     const float  *d_db,
                                     int          *d_out_indices,
                                     float        *d_out_scores,
                                     int           dim,
                                     int           db_size,
                                     int           n_queries,
                                     int           top_k,
                                     cudaStream_t  stream)
{
    cudaError_t err = launch_search_kernel(d_queries, d_db,
                                           d_out_indices, d_out_scores,
                                           dim, db_size, n_queries, top_k,
                                           stream);
    set_last_error(err);
    return err;
}

cudaError_t tile_batch_evolve(float        *d_scores,
                              const float  *d_rewards,
                              int          *d_win_counts,
                              int          *d_total,
                              int           count,
                              float         learning_rate,
                              float         clamp_min,
                              float         clamp_max,
                              cudaStream_t  stream)
{
    cudaError_t err = launch_evolve_kernel(d_scores, d_rewards,
                                           d_win_counts, d_total, count,
                                           learning_rate, clamp_min, clamp_max,
                                           stream);
    set_last_error(err);
    return err;
}

cudaError_t tile_batch_svd(const float  *d_matrices,
                           float        *d_U,
                           float        *d_S,
                           float        *d_Vt,
                           int           rows,
                           int           cols,
                           int           batch,
                           int           max_iters,
                           float         tol,
                           cudaStream_t  stream)
{
    cudaError_t err = launch_svd_kernel(d_matrices, d_U, d_S, d_Vt,
                                        rows, cols, batch,
                                        max_iters, tol, stream);
    set_last_error(err);
    return err;
}
