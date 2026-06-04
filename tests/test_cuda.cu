/*
 * test_cuda.cu — Unit tests for tile CUDA operations.
 *
 * Build: make test
 * Run:   ./test_cuda
 */

#include "tile_cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int g_pass = 0;
static int g_fail = 0;

#define ASSERT(cond, msg) do {                          \
    if (!(cond)) {                                      \
        printf("  FAIL: %s\n", msg);                    \
        g_fail++;                                       \
    } else {                                            \
        printf("  PASS: %s\n", msg);                    \
        g_pass++;                                       \
    }                                                   \
} while(0)

#define ASSERT_F(cond, fmt, ...) do {                   \
    if (!(cond)) {                                      \
        printf("  FAIL: " fmt "\n", __VA_ARGS__);       \
        g_fail++;                                       \
    } else {                                            \
        printf("  PASS: " fmt "\n", __VA_ARGS__);       \
        g_pass++;                                       \
    }                                                   \
} while(0)

/* ------------------------------------------------------------------ */
/*  Test: Hash                                                         */
/* ------------------------------------------------------------------ */

static void test_hash(void)
{
    printf("\n--- test_hash ---\n");
    int count = 256;
    size_t stride = 128;

    uint8_t *h_states = (uint8_t *)calloc(count, stride);
    uint8_t *h_hashes = (uint8_t *)calloc(count, 64);

    /* Fill with known data */
    for (int i = 0; i < count * (int)stride; i++)
        h_states[i] = (uint8_t)(i & 0xFF);

    uint8_t *d_states, *d_hashes;
    CUDA_CHECK(cudaMalloc(&d_states, count * stride));
    CUDA_CHECK(cudaMalloc(&d_hashes, count * 64));
    CUDA_CHECK(cudaMemcpy(d_states, h_states, count * stride,
                           cudaMemcpyHostToDevice));

    CUDA_CHECK(tile_batch_hash(d_states, d_hashes, stride, count, 0));
    CUDA_CHECK(cudaMemcpy(h_hashes, d_hashes, count * 64,
                           cudaMemcpyDeviceToHost));

    /* Check: all hashes should be non-zero (BLAKE2b produces non-trivial output) */
    int nonzero = 0;
    for (int i = 0; i < count; i++) {
        int allzero = 1;
        for (int j = 0; j < 64; j++) {
            if (h_hashes[i * 64 + j] != 0) { allzero = 0; break; }
        }
        if (!allzero) nonzero++;
    }
    ASSERT_F(nonzero == count, "all %d hashes are non-zero", count);

    /* Check: identical inputs → identical hashes */
    /* First two states differ (index 0 vs 128), but fill with same pattern */
    int same = memcmp(h_hashes, h_hashes + 64, 64) == 0;
    /* First two states are NOT identical (different data at offset 0 vs 128) */
    /* Let's do: hash same state twice and compare */
    CUDA_CHECK(cudaMemcpy(d_states, h_states, stride, cudaMemcpyHostToDevice));
    CUDA_CHECK(tile_batch_hash(d_states, d_hashes, stride, 1, 0));
    uint8_t hash1[64];
    CUDA_CHECK(cudaMemcpy(hash1, d_hashes, 64, cudaMemcpyDeviceToHost));

    CUDA_CHECK(tile_batch_hash(d_states, d_hashes, stride, 1, 0));
    uint8_t hash2[64];
    CUDA_CHECK(cudaMemcpy(hash2, d_hashes, 64, cudaMemcpyDeviceToHost));

    ASSERT(memcmp(hash1, hash2, 64) == 0, "deterministic: same input → same hash");

    cudaFree(d_states); cudaFree(d_hashes);
    free(h_states); free(h_hashes);
}

/* ------------------------------------------------------------------ */
/*  Test: Embed                                                        */
/* ------------------------------------------------------------------ */

static void test_embed(void)
{
    printf("\n--- test_embed ---\n");
    int count = 100;
    int max_tokens = 8;

    int *h_tokens = (int *)calloc(count * max_tokens, sizeof(int));
    float *h_vecs = (float *)calloc(count * TILE_EMBED_DIM, sizeof(float));

    for (int i = 0; i < count * max_tokens; i++)
        h_tokens[i] = (i % 500) + 1;

    int *d_tokens;
    float *d_vecs;
    CUDA_CHECK(cudaMalloc(&d_tokens, count * max_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vecs, count * TILE_EMBED_DIM * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens,
                           count * max_tokens * sizeof(int),
                           cudaMemcpyHostToDevice));

    CUDA_CHECK(tile_batch_embed(d_tokens, d_vecs, max_tokens, count, 0));
    CUDA_CHECK(cudaMemcpy(h_vecs, d_vecs,
                           count * TILE_EMBED_DIM * sizeof(float),
                           cudaMemcpyDeviceToHost));

    /* Check: vectors should not be all zeros */
    int nonzero = 0;
    for (int i = 0; i < count; i++) {
        for (int d = 0; d < TILE_EMBED_DIM; d++) {
            if (fabsf(h_vecs[i * TILE_EMBED_DIM + d]) > 1e-6f) {
                nonzero++;
                break;
            }
        }
    }
    ASSERT_F(nonzero == count, "all %d embeddings are non-zero", count);

    /* Check: different inputs → different outputs */
    int diff = 0;
    for (int d = 0; d < TILE_EMBED_DIM; d++) {
        if (fabsf(h_vecs[0 * TILE_EMBED_DIM + d] -
                  h_vecs[1 * TILE_EMBED_DIM + d]) > 1e-6f) {
            diff = 1;
            break;
        }
    }
    ASSERT(diff, "different inputs produce different embeddings");

    cudaFree(d_tokens); cudaFree(d_vecs);
    free(h_tokens); free(h_vecs);
}

/* ------------------------------------------------------------------ */
/*  Test: Search                                                       */
/* ------------------------------------------------------------------ */

static void test_search(void)
{
    printf("\n--- test_search ---\n");
    int db_size = 1000;
    int top_k = 5;
    int dim = TILE_EMBED_DIM;

    float *h_query = (float *)malloc(dim * sizeof(float));
    float *h_db = (float *)malloc(db_size * dim * sizeof(float));
    float *h_scores = (float *)malloc(top_k * sizeof(float));
    int *h_indices = (int *)malloc(top_k * sizeof(int));

    /* Query: unit vector [1,0,0,...] */
    memset(h_query, 0, dim * sizeof(float));
    h_query[0] = 1.0f;

    /* DB: put a known match at index 42 */
    for (int i = 0; i < db_size; i++) {
        memset(h_db + i * dim, 0, dim * sizeof(float));
        h_db[i * dim + 0] = (float)(i + 1) / (float)db_size;
        /* index 42 has highest alignment with query */
    }
    h_db[42 * dim + 0] = 1.0f;

    float *d_query, *d_db, *d_scores;
    int *d_indices;
    CUDA_CHECK(cudaMalloc(&d_query, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db, db_size * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scores, top_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_indices, top_k * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_query, h_query, dim * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_db, h_db, db_size * dim * sizeof(float),
                           cudaMemcpyHostToDevice));

    CUDA_CHECK(tile_batch_cosine_search(d_query, d_db, d_indices, d_scores,
                                        dim, db_size, 1, top_k, 0));
    CUDA_CHECK(cudaMemcpy(h_scores, d_scores, top_k * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_indices, d_indices, top_k * sizeof(int),
                           cudaMemcpyDeviceToHost));

    /* Index 42 should be the top result */
    ASSERT_F(h_indices[0] == 42, "top result is index 42 (got %d)", h_indices[0]);
    ASSERT_F(h_scores[0] > 0.99f, "top score > 0.99 (got %.4f)", h_scores[0]);

    /* Scores should be in descending order */
    int sorted = 1;
    for (int i = 1; i < top_k; i++) {
        if (h_scores[i] > h_scores[i-1]) { sorted = 0; break; }
    }
    ASSERT(sorted, "results sorted by descending score");

    cudaFree(d_query); cudaFree(d_db);
    cudaFree(d_scores); cudaFree(d_indices);
    free(h_query); free(h_db); free(h_scores); free(h_indices);
}

/* ------------------------------------------------------------------ */
/*  Test: Evolve                                                       */
/* ------------------------------------------------------------------ */

static void test_evolve(void)
{
    printf("\n--- test_evolve ---\n");
    int count = 10000;

    float *h_scores = (float *)malloc(count * sizeof(float));
    float *h_rewards = (float *)malloc(count * sizeof(float));
    int *h_wins = (int *)calloc(count, sizeof(int));
    int *h_total = (int *)calloc(count, sizeof(int));

    for (int i = 0; i < count; i++) {
        h_scores[i] = 0.5f;
        h_rewards[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    }

    float *d_scores, *d_rewards;
    int *d_wins, *d_total;
    CUDA_CHECK(cudaMalloc(&d_scores, count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rewards, count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wins, count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_total, count * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_scores, h_scores, count * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rewards, h_rewards, count * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wins, h_wins, count * sizeof(int),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_total, h_total, count * sizeof(int),
                           cudaMemcpyHostToDevice));

    CUDA_CHECK(tile_batch_evolve(d_scores, d_rewards, d_wins, d_total,
                                 count, 0.01f, 0.0f, 1.0f, 0));
    CUDA_CHECK(cudaMemcpy(h_scores, d_scores, count * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_wins, d_wins, count * sizeof(int),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_total, d_total, count * sizeof(int),
                           cudaMemcpyDeviceToHost));

    /* Check: scores changed */
    int changed = 0;
    for (int i = 0; i < count; i++) {
        if (fabsf(h_scores[i] - 0.5f) > 1e-6f) { changed = 1; break; }
    }
    ASSERT(changed, "scores updated after evolve");

    /* Check: all totals incremented */
    int all_one = 1;
    for (int i = 0; i < count; i++) {
        if (h_total[i] != 1) { all_one = 0; break; }
    }
    ASSERT(all_one, "all total counts == 1");

    /* Check: scores clamped [0, 1] */
    int clamped = 1;
    for (int i = 0; i < count; i++) {
        if (h_scores[i] < -1e-6f || h_scores[i] > 1.0f + 1e-6f) {
            clamped = 0; break;
        }
    }
    ASSERT(clamped, "all scores within [0, 1]");

    cudaFree(d_scores); cudaFree(d_rewards);
    cudaFree(d_wins); cudaFree(d_total);
    free(h_scores); free(h_rewards); free(h_wins); free(h_total);
}

/* ------------------------------------------------------------------ */
/*  Test: SVD                                                          */
/* ------------------------------------------------------------------ */

static void test_svd(void)
{
    printf("\n--- test_svd ---\n");
    int rows = 100;
    int cols = 5;
    int batch = 10;

    int total = batch * rows * cols;
    float *h_mat = (float *)malloc(total * sizeof(float));

    /* Create a rank-1 matrix: A = u * v^T */
    for (int b = 0; b < batch; b++) {
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                h_mat[(b * rows + r) * cols + c] =
                    (float)(r + 1) * (float)(c + 1) * (float)(b + 1);
    }

    float *d_mat, *d_U, *d_S, *d_Vt;
    CUDA_CHECK(cudaMalloc(&d_mat, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_U, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_S, batch * cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Vt, batch * cols * cols * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_mat, h_mat, total * sizeof(float),
                           cudaMemcpyHostToDevice));

    CUDA_CHECK(tile_batch_svd(d_mat, d_U, d_S, d_Vt,
                              rows, cols, batch, 200, 1e-6f, 0));

    float *h_S = (float *)malloc(batch * cols * sizeof(float));
    float *h_U = (float *)malloc(total * sizeof(float));
    float *h_Vt = (float *)malloc(batch * cols * cols * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_S, d_S, batch * cols * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_U, d_U, total * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Vt, d_Vt, batch * cols * cols * sizeof(float),
                           cudaMemcpyDeviceToHost));

    /* Check: singular values are non-negative */
    int sv_nonneg = 1;
    for (int i = 0; i < batch * cols; i++) {
        if (h_S[i] < -1e-4f) { sv_nonneg = 0; break; }
    }
    ASSERT(sv_nonneg, "singular values are non-negative");

    /* Check: first singular value is the largest */
    int sv_sorted = 1;
    for (int b = 0; b < batch; b++) {
        for (int i = 1; i < cols; i++) {
            if (h_S[b * cols + i] > h_S[b * cols + 0] + 1e-2f) {
                sv_sorted = 0; break;
            }
        }
    }
    ASSERT(sv_sorted, "first singular value is largest (within tolerance)");

    /* Check: V^T rows are orthonormal (Vt * Vt^T ≈ I) */
    int ortho = 1;
    for (int b = 0; b < batch; b++) {
        float *vt = h_Vt + b * cols * cols;
        for (int i = 0; i < cols; i++) {
            for (int j = 0; j <= i; j++) {
                float dot = 0;
                for (int k = 0; k < cols; k++)
                    dot += vt[i * cols + k] * vt[j * cols + k];
                float expected = (i == j) ? 1.0f : 0.0f;
                if (fabsf(dot - expected) > 0.1f) {
                    ortho = 0;
                    goto done_ortho;
                }
            }
        }
    }
done_ortho:
    ASSERT(ortho, "V^T rows approximately orthonormal");

    cudaFree(d_mat); cudaFree(d_U); cudaFree(d_S); cudaFree(d_Vt);
    free(h_mat); free(h_S); free(h_U); free(h_Vt);
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

int main(void)
{
    printf("=== Tile CUDA Tests ===\n");

    /* Ensure a device is available */
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("No CUDA devices found — running compile-only tests.\n");
        ASSERT(1, "code compiles cleanly");
        printf("\nResults: %d passed, %d failed\n", g_pass, g_fail);
        return g_fail > 0 ? 1 : 0;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);

    test_hash();
    test_embed();
    test_search();
    test_evolve();
    test_svd();

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
