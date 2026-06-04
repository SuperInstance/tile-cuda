/*
 * evolve_kernel.cu — Batch tile evolution (score update).
 *
 * Each thread updates one tile: compute new score from win rate,
 * apply learning rate, clamp.
 *
 * Target: 100M tiles/sec on RTX 4050.
 */

#include "tile_cuda.h"
#include <math.h>

/* ------------------------------------------------------------------ */
/*  Evolve kernel                                                      */
/* ------------------------------------------------------------------ */

__global__ void evolve_kernel(float       * __restrict__ scores,
                              const float * __restrict__ rewards,
                              int         * __restrict__ win_counts,
                              int         * __restrict__ total_counts,
                              int    count,
                              float  learning_rate,
                              float  clamp_min,
                              float  clamp_max)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    float reward = rewards[idx];

    /* Update win/total atomically */
    if (reward > 0.0f) {
        atomicAdd(&win_counts[idx], 1);
    }
    atomicAdd(&total_counts[idx], 1);

    /* Read updated counts */
    int wins  = win_counts[idx];
    int total = total_counts[idx];

    /* Compute win rate */
    float win_rate = (total > 0) ? (float)wins / (float)total : 0.0f;

    /* Apply learning rate update */
    float score = scores[idx];
    score += learning_rate * (reward * win_rate - score * 0.01f);

    /* Clamp */
    score = fmaxf(clamp_min, fminf(clamp_max, score));

    scores[idx] = score;
}

/* ------------------------------------------------------------------ */
/*  Launch wrapper                                                     */
/* ------------------------------------------------------------------ */

cudaError_t launch_evolve_kernel(float        *d_scores,
                                 const float  *d_rewards,
                                 int          *d_win_counts,
                                 int          *d_total,
                                 int           count,
                                 float         learning_rate,
                                 float         clamp_min,
                                 float         clamp_max,
                                 cudaStream_t  stream)
{
    if (count <= 0) return cudaSuccess;

    int block = 256;
    int grid  = (count + block - 1) / block;

    evolve_kernel<<<grid, block, 0, stream>>>(
        d_scores, d_rewards, d_win_counts, d_total,
        count, learning_rate, clamp_min, clamp_max);

    return cudaGetLastError();
}
