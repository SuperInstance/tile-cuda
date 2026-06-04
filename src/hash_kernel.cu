/*
 * hash_kernel.cu — BLAKE2b batch hashing on GPU.
 *
 * Each thread hashes one state string independently.
 * Target: 10M hashes/sec on RTX 4050.
 *
 * BLAKE2b parameters: 64-byte digest, 128-byte block, 12 rounds.
 */

#include "tile_cuda.h"
#include <stdint.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  BLAKE2b constants                                                  */
/* ------------------------------------------------------------------ */

static __device__ __constant__ uint64_t d_IV[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
};

static __device__ __constant__ uint64_t d_SIGMA[12][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15},
    {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3},
    {11, 8,12, 0, 5, 2,15,13,10,14, 3, 6, 7, 1, 9, 4},
    { 7, 9, 3, 1,13,12,11,14, 2, 6, 5,10, 4, 0,15, 8},
    { 9, 0, 5, 7, 2, 4,10,15,14, 1,11,12, 6, 8, 3,13},
    { 2,12, 6,10, 0,11, 8, 3, 4,13, 7, 5,15,14, 1, 9},
    {12, 5, 1,15,14,13, 4,10, 0, 7, 6, 3, 9, 2, 8,11},
    {13,11, 7,14,12, 1, 3, 9, 5, 0,15, 4, 8, 6, 2,10},
    { 6,15,14, 9,11, 3, 0, 8,12, 2,13, 7, 1, 4,10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5,15,11, 9,14, 3,12,13, 0},
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15},
    {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3}
};

/* ------------------------------------------------------------------ */
/*  BLAKE2b G-function                                                 */
/* ------------------------------------------------------------------ */

static __device__ __forceinline__
void G(uint64_t &a, uint64_t &b, uint64_t &c, uint64_t &d,
       uint64_t x, uint64_t y)
{
    a = a + b + x;
    d = (d ^ a) >> 32 | (d ^ a) << 32;  /* rotr64(d,32) */
    c = c + d;
    b = (b ^ c) >> 24 | (b ^ c) << 40;  /* rotr64(b,24) */
    a = a + b + y;
    d = (d ^ a) >> 16 | (d ^ a) << 48;  /* rotr64(d,16) */
    c = c + d;
    b = (b ^ c) >> 63 | (b ^ c) << 1;   /* rotr64(b,63) */
}

/* ------------------------------------------------------------------ */
/*  BLAKE2b compression function                                       */
/* ------------------------------------------------------------------ */

static __device__ __forceinline__
void blake2b_compress(uint64_t h[8], const uint64_t block[16],
                      uint64_t counter, bool is_final)
{
    uint64_t v[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        v[i]   = h[i];
        v[i+8] = d_IV[i];
    }
    v[12] ^= counter;
    v[13] ^= 0; /* counter >> 64 = 0 for our use */
    if (is_final) v[14] = ~v[14];

    /* 12 rounds */
    for (int round = 0; round < 12; round++) {
        const uint64_t *s = d_SIGMA[round];
        G(v[ 0], v[ 4], v[ 8], v[12], block[s[ 0]], block[s[ 1]]);
        G(v[ 1], v[ 5], v[ 9], v[13], block[s[ 2]], block[s[ 3]]);
        G(v[ 2], v[ 6], v[10], v[14], block[s[ 4]], block[s[ 5]]);
        G(v[ 3], v[ 7], v[11], v[15], block[s[ 6]], block[s[ 7]]);
        G(v[ 0], v[ 5], v[10], v[15], block[s[ 8]], block[s[ 9]]);
        G(v[ 1], v[ 6], v[11], v[12], block[s[10]], block[s[11]]);
        G(v[ 2], v[ 7], v[ 8], v[13], block[s[12]], block[s[13]]);
        G(v[ 3], v[ 4], v[ 9], v[14], block[s[14]], block[s[15]]);
    }

    #pragma unroll
    for (int i = 0; i < 8; i++)
        h[i] ^= v[i] ^ v[i+8];
}

/* ------------------------------------------------------------------ */
/*  Full BLAKE2b hash (supports up to 2 blocks = 256 bytes input)      */
/* ------------------------------------------------------------------ */

static __device__ __forceinline__
void blake2b_device(const uint8_t *msg, size_t msg_len, uint8_t *digest)
{
    uint64_t h[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) h[i] = d_IV[i];
    h[0] ^= 0x01010040; /* key_len=0, fanout=1, depth=1 */

    /* Pad message to 128-byte blocks */
    uint64_t block[16] = {0};

    if (msg_len <= 128) {
        /* Single block */
        memcpy(block, msg, msg_len);
        blake2b_compress(h, block, msg_len, true);
    } else {
        /* First block */
        memcpy(block, msg, 128);
        blake2b_compress(h, block, 128, false);

        /* Second (final) block */
        memset(block, 0, sizeof(block));
        size_t rem = msg_len - 128;
        memcpy(block, msg + 128, rem);
        blake2b_compress(h, block, msg_len, true);
    }

    memcpy(digest, h, 64);
}

/* ------------------------------------------------------------------ */
/*  Batch hash kernel                                                  */
/* ------------------------------------------------------------------ */

__global__ void hash_kernel(const uint8_t * __restrict__ states,
                            uint8_t       * __restrict__ hashes,
                            size_t state_stride,
                            int    count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    blake2b_device(states + (size_t)idx * state_stride,
                   state_stride,
                   hashes + (size_t)idx * TILE_HASH_LEN);
}

/* ------------------------------------------------------------------ */
/*  Launch wrapper                                                     */
/* ------------------------------------------------------------------ */

cudaError_t launch_hash_kernel(const void   *d_states,
                               void         *d_hashes,
                               size_t        state_stride,
                               int           count,
                               cudaStream_t  stream)
{
    if (count <= 0) return cudaSuccess;

    /* 256 threads/block gives good occupancy on Ada */
    int block = 256;
    int grid  = (count + block - 1) / block;

    hash_kernel<<<grid, block, 0, stream>>>(
        (const uint8_t *)d_states,
        (uint8_t *)d_hashes,
        state_stride,
        count);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    return cudaSuccess;
}
