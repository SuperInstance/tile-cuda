/*
 * bench_cuda.cu — CUDA event-timing benchmarks for tile operations.
 *
 * Compares GPU vs CPU performance at [100, 1K, 10K, 100K, 1M] scales.
 * Uses cudaEventRecord for accurate GPU timing.
 *
 * Build: make bench
 * Run:   ./bench_cuda
 */

#include "tile_cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>

/* ------------------------------------------------------------------ */
/*  CPU reference implementations (for comparison)                     */
/* ------------------------------------------------------------------ */

/* Simple CPU BLAKE2b placeholder — just a basic hash for timing */
static void cpu_hash_reference(const uint8_t *input, size_t len,
                               uint8_t *output)
{
    /* XOR-shift hash as a stand-in for BLAKE2b timing baseline */
    uint64_t h[8];
    for (int i = 0; i < 8; i++) h[i] = 0x1234567890ABCDEFULL ^ i;
    for (size_t i = 0; i < len; i++) {
        h[i % 8] ^= (uint64_t)input[i] << ((i * 7) % 56);
        h[i % 8] = h[i % 8] * 0x517cc1b727220a95ULL + 0x6a09e667f3bcc908ULL;
    }
    memcpy(output, h, 64);
}

static void cpu_embed_reference(const int *tokens, float *vec,
                                int max_tokens, int dim)
{
    memset(vec, 0, dim * sizeof(float));
    for (int pos = 0; pos < max_tokens; pos++) {
        int token = tokens[pos];
        if (token == 0) continue;
        unsigned int h = (unsigned)token ^ (unsigned)pos;
        h = ((h >> 16) ^ h) * 0x45d9f3b;
        h = ((h >> 16) ^ h) * 0x45d9f3b;
        h = (h >> 16) ^ h;
        int base = (pos * 7) % dim;
        for (int k = 0; k < 4; k++) {
            int d = (base + (int)((h >> (k*4)) & 0xF)) % dim;
            vec[d] += (float)(h & 0xFF) / (255.0f * max_tokens);
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Timing helpers                                                     */
/* ------------------------------------------------------------------ */

#define GPU_BENCH_BEGIN()                       \
    cudaEvent_t start, stop;                    \
    cudaEventCreate(&start);                    \
    cudaEventCreate(&stop);                     \
    cudaEventRecord(start, stream);

#define GPU_BENCH_END(ms)                       \
    cudaEventRecord(stop, stream);              \
    cudaEventSynchronize(stop);                 \
    cudaEventElapsedTime(&ms, start, stop);     \
    cudaEventDestroy(start);                    \
    cudaEventDestroy(stop);

/* ------------------------------------------------------------------ */
/*  Benchmark: Hash                                                    */
/* ------------------------------------------------------------------ */

static void bench_hash(cudaStream_t stream)
{
    printf("\n=== Hash Benchmark ===\n");
    printf("%-10s %12s %12s %8s\n", "Count", "GPU(ms)", "CPU(ms)", "Speedup");
    printf("--------------------------------------------------\n");

    int scales[] = {100, 1000, 10000, 100000, 1000000};
    size_t state_stride = 128; /* 128 bytes per state */

    for (int s = 0; s < 5; s++) {
        int count = scales[s];

        /* Allocate */
        uint8_t *d_states, *d_hashes;
        uint8_t *h_states = (uint8_t *)malloc(count * state_stride);
        uint8_t *h_hashes = (uint8_t *)malloc(count * 64);

        /* Fill with test data */
        for (int i = 0; i < count * (int)state_stride; i++)
            h_states[i] = (uint8_t)(i * 7 + 13);

        CUDA_CHECK(cudaMalloc(&d_states, count * state_stride));
        CUDA_CHECK(cudaMalloc(&d_hashes, count * 64));
        CUDA_CHECK(cudaMemcpy(d_states, h_states,
                               count * state_stride, cudaMemcpyHostToDevice));

        /* GPU timing */
        float gpu_ms;
        GPU_BENCH_BEGIN();
        CUDA_CHECK(tile_batch_hash(d_states, d_hashes, state_stride,
                                   count, stream));
        GPU_BENCH_END(gpu_ms);

        /* CPU timing */
        auto cpu_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < count; i++)
            cpu_hash_reference(h_states + i * state_stride, state_stride,
                               h_hashes + i * 64);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        float cpu_ms = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();

        printf("%-10d %12.3f %12.3f %8.2fx\n", count, gpu_ms, cpu_ms,
               cpu_ms / (gpu_ms > 0.001f ? gpu_ms : 0.001f));

        cudaFree(d_states);
        cudaFree(d_hashes);
        free(h_states);
        free(h_hashes);
    }
}

/* ------------------------------------------------------------------ */
/*  Benchmark: Embed                                                   */
/* ------------------------------------------------------------------ */

static void bench_embed(cudaStream_t stream)
{
    printf("\n=== Embed Benchmark ===\n");
    printf("%-10s %12s %12s %8s\n", "Count", "GPU(ms)", "CPU(ms)", "Speedup");
    printf("--------------------------------------------------\n");

    int scales[] = {100, 1000, 10000, 100000, 1000000};
    int max_tokens = 16;

    for (int s = 0; s < 5; s++) {
        int count = scales[s];

        int *d_tokens;
        float *d_vectors;
        int *h_tokens = (int *)malloc(count * max_tokens * sizeof(int));
        float *h_vecs = (float *)malloc(count * TILE_EMBED_DIM * sizeof(float));

        for (int i = 0; i < count * max_tokens; i++)
            h_tokens[i] = (i % 1000) + 1;

        CUDA_CHECK(cudaMalloc(&d_tokens, count * max_tokens * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_vectors, count * TILE_EMBED_DIM * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens,
                               count * max_tokens * sizeof(int),
                               cudaMemcpyHostToDevice));

        float gpu_ms;
        GPU_BENCH_BEGIN();
        CUDA_CHECK(tile_batch_embed(d_tokens, d_vectors, max_tokens,
                                    count, stream));
        GPU_BENCH_END(gpu_ms);

        auto cpu_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < count; i++)
            cpu_embed_reference(h_tokens + i * max_tokens,
                               h_vecs + i * TILE_EMBED_DIM,
                               max_tokens, TILE_EMBED_DIM);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        float cpu_ms = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();

        printf("%-10d %12.3f %12.3f %8.2fx\n", count, gpu_ms, cpu_ms,
               cpu_ms / (gpu_ms > 0.001f ? gpu_ms : 0.001f));

        cudaFree(d_tokens);
        cudaFree(d_vectors);
        free(h_tokens);
        free(h_vecs);
    }
}

/* ------------------------------------------------------------------ */
/*  Benchmark: Search                                                  */
/* ------------------------------------------------------------------ */

static void bench_search(cudaStream_t stream)
{
    printf("\n=== Search Benchmark (1 query vs N db vectors) ===\n");
    printf("%-10s %12s %12s %8s\n", "DB Size", "GPU(ms)", "CPU(ms)", "Speedup");
    printf("--------------------------------------------------\n");

    int db_sizes[] = {100, 1000, 10000, 100000};

    for (int s = 0; s < 4; s++) {
        int db_size = db_sizes[s];
        int top_k = 5;

        /* Allocate query + db */
        float *d_query, *d_db, *d_scores;
        int *d_indices;
        float *h_query = (float *)malloc(TILE_EMBED_DIM * sizeof(float));
        float *h_db = (float *)malloc(db_size * TILE_EMBED_DIM * sizeof(float));
        float *h_scores = (float *)malloc(top_k * sizeof(float));
        int *h_indices = (int *)malloc(top_k * sizeof(int));

        for (int i = 0; i < TILE_EMBED_DIM; i++)
            h_query[i] = (float)sin(i * 0.1) * 0.5f + 0.5f;
        for (int i = 0; i < db_size * TILE_EMBED_DIM; i++)
            h_db[i] = (float)cos(i * 0.07) * 0.5f + 0.5f;

        CUDA_CHECK(cudaMalloc(&d_query, TILE_EMBED_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_db, db_size * TILE_EMBED_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_scores, top_k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_indices, top_k * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_query, h_query, TILE_EMBED_DIM * sizeof(float),
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_db, h_db,
                               db_size * TILE_EMBED_DIM * sizeof(float),
                               cudaMemcpyHostToDevice));

        float gpu_ms;
        GPU_BENCH_BEGIN();
        CUDA_CHECK(tile_batch_cosine_search(d_query, d_db, d_indices, d_scores,
                                            TILE_EMBED_DIM, db_size, 1, top_k,
                                            stream));
        GPU_BENCH_END(gpu_ms);

        /* CPU timing */
        auto cpu_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < db_size; i++) {
            float dot = 0;
            for (int d = 0; d < TILE_EMBED_DIM; d++)
                dot += h_query[d] * h_db[i * TILE_EMBED_DIM + d];
        }
        auto cpu_end = std::chrono::high_resolution_clock::now();
        float cpu_ms = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();

        printf("%-10d %12.3f %12.3f %8.2fx\n", db_size, gpu_ms, cpu_ms,
               cpu_ms / (gpu_ms > 0.001f ? gpu_ms : 0.001f));

        cudaFree(d_query); cudaFree(d_db);
        cudaFree(d_scores); cudaFree(d_indices);
        free(h_query); free(h_db); free(h_scores); free(h_indices);
    }
}

/* ------------------------------------------------------------------ */
/*  Benchmark: Evolve                                                  */
/* ------------------------------------------------------------------ */

static void bench_evolve(cudaStream_t stream)
{
    printf("\n=== Evolve Benchmark ===\n");
    printf("%-10s %12s\n", "Count", "GPU(ms)");
    printf("------------------------\n");

    int scales[] = {100, 1000, 10000, 100000, 1000000};

    for (int s = 0; s < 5; s++) {
        int count = scales[s];

        float *d_scores, *d_rewards;
        int *d_wins, *d_total;

        CUDA_CHECK(cudaMalloc(&d_scores, count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rewards, count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_wins, count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_total, count * sizeof(int)));

        float gpu_ms;
        GPU_BENCH_BEGIN();
        CUDA_CHECK(tile_batch_evolve(d_scores, d_rewards, d_wins, d_total,
                                     count, 0.01f, 0.0f, 1.0f, stream));
        GPU_BENCH_END(gpu_ms);

        printf("%-10d %12.3f\n", count, gpu_ms);

        cudaFree(d_scores); cudaFree(d_rewards);
        cudaFree(d_wins); cudaFree(d_total);
    }
}

/* ------------------------------------------------------------------ */
/*  Benchmark: SVD                                                     */
/* ------------------------------------------------------------------ */

static void bench_svd(cudaStream_t stream)
{
    printf("\n=== SVD Benchmark (rows×10 matrices) ===\n");
    printf("%-10s %12s %12s\n", "Batch", "Rows", "GPU(ms)");
    printf("---------------------------------------\n");

    int batches[] = {1, 10, 100, 1000};
    int rows = 10000;
    int cols = 10;

    for (int s = 0; s < 4; s++) {
        int batch = batches[s];
        int total = batch * rows * cols;

        float *d_mat, *d_U, *d_S, *d_Vt;
        float *h_mat = (float *)malloc(total * sizeof(float));
        for (int i = 0; i < total; i++)
            h_mat[i] = (float)sin(i * 0.01) * 0.5f;

        CUDA_CHECK(cudaMalloc(&d_mat, total * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_U, total * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_S, batch * cols * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Vt, batch * cols * cols * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_mat, h_mat, total * sizeof(float),
                               cudaMemcpyHostToDevice));

        float gpu_ms;
        GPU_BENCH_BEGIN();
        CUDA_CHECK(tile_batch_svd(d_mat, d_U, d_S, d_Vt,
                                  rows, cols, batch, 100, 1e-6f, stream));
        GPU_BENCH_END(gpu_ms);

        printf("%-10d %12d %12.3f\n", batch, rows, gpu_ms);

        cudaFree(d_mat); cudaFree(d_U); cudaFree(d_S); cudaFree(d_Vt);
        free(h_mat);
    }
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

int main(void)
{
    /* Print device info */
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("GPU: %s (SM %d.%d, %d SMs, %.1f GB)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           prop.totalGlobalMem / 1e9);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    bench_hash(stream);
    bench_embed(stream);
    bench_search(stream);
    bench_evolve(stream);
    bench_svd(stream);

    CUDA_CHECK(cudaStreamDestroy(stream));
    printf("\nAll benchmarks complete.\n");
    return 0;
}
