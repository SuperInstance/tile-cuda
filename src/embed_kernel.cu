/*
 * embed_kernel.cu — Position-aware embedding on GPU.
 *
 * Maps token IDs → 64-dim float vectors where position modulates
 * which dimensions are set.  Uses shared memory for intermediate
 * accumulation.
 *
 * Target: 1M embeds/sec on RTX 4050.
 */

#include "tile_cuda.h"
#include <math.h>

/* ------------------------------------------------------------------ */
/*  Simple hash for token → dimension mapping                          */
/* ------------------------------------------------------------------ */

static __device__ __forceinline__
unsigned int token_hash(int token_id, int seed)
{
    unsigned int h = (unsigned int)token_id ^ (unsigned int)seed;
    h = ((h >> 16) ^ h) * 0x45d9f3b;
    h = ((h >> 16) ^ h) * 0x45d9f3b;
    h = (h >> 16) ^ h;
    return h;
}

/* ------------------------------------------------------------------ */
/*  Embedding kernel                                                   */
/* ------------------------------------------------------------------ */

/*
 * Each block processes one intent (up to max_tokens tokens).
 * Each thread handles one or more output dimensions.
 *
 * Shared memory layout: float[TILE_EMBED_DIM] for accumulation.
 */

__global__ void embed_kernel(const int  * __restrict__ token_ids,  /* [count * max_tokens] */
                             float      * __restrict__ vectors,    /* [count * TILE_EMBED_DIM] */
                             int max_tokens,
                             int count)
{
    /* One intent per block */
    int intent_idx = blockIdx.x;
    if (intent_idx >= count) return;

    int tid = threadIdx.x;

    /* Shared accumulator for this intent */
    extern __shared__ float s_vec[];

    /* Initialize shared memory */
    for (int d = tid; d < TILE_EMBED_DIM; d += blockDim.x)
        s_vec[d] = 0.0f;

    __syncthreads();

    /* Each thread processes some tokens and accumulates into shared mem */
    for (int pos = tid; pos < max_tokens; pos += blockDim.x) {
        int token = token_ids[intent_idx * max_tokens + pos];
        if (token == 0) continue; /* padding */

        /* Position-aware: position determines base dimension offset */
        int base_dim = (pos * 7) % TILE_EMBED_DIM; /* prime stride */

        /* Hash token to select 4 dimensions */
        unsigned int h = token_hash(token, pos);
        for (int k = 0; k < 4; k++) {
            int dim = (base_dim + (int)((h >> (k * 4)) & 0xF)) % TILE_EMBED_DIM;
            float value = __int_as_float(0x3f800000 | ((h >> (k * 8)) & 0x007FFFFF));
            /* Keep it in [0, 1] */
            value = (value - 1.0f);
            atomicAdd(&s_vec[dim], value / (float)max_tokens);
        }
    }

    __syncthreads();

    /* Write out */
    for (int d = tid; d < TILE_EMBED_DIM; d += blockDim.x)
        vectors[intent_idx * TILE_EMBED_DIM + d] = s_vec[d];
}

/* ------------------------------------------------------------------ */
/*  Launch wrapper                                                     */
/* ------------------------------------------------------------------ */

cudaError_t launch_embed_kernel(const int   *d_token_ids,
                                float       *d_vectors,
                                int          max_tokens,
                                int          count,
                                cudaStream_t stream)
{
    if (count <= 0) return cudaSuccess;

    /* Use TILE_EMBED_DIM threads per block (one intent per block) */
    int block = TILE_EMBED_DIM; /* 64 */
    int grid  = count;
    size_t shmem = TILE_EMBED_DIM * sizeof(float);

    embed_kernel<<<grid, block, shmem, stream>>>(
        d_token_ids, d_vectors, max_tokens, count);

    return cudaGetLastError();
}
