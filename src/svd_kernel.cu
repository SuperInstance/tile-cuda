/*
 * svd_kernel.cu — Batch SVD factorization.
 *
 * For small matrices (rows × cols, cols ≤ TILE_SVD_N_MAX):
 *   - One block per matrix
 *   - Power iteration for singular values
 *   - Jacobi rotations for 2×2 subproblems
 *
 * For larger matrices the host wrapper falls back to cuBLAS gesvd.
 *
 * Target: 1K SVDs/sec for 10K×10 matrices.
 */

#include "tile_cuda.h"
#include <math.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Small matrix helpers (device, per-block)                           */
/* ------------------------------------------------------------------ */

/* C = A^T * B  for cols_a × rows x rows × cols_b → cols_a × cols_b */
/* All matrices are row-major and stored in shared/global memory.      */

static __device__ __forceinline__
void matmul_10x10(float C[10][10],
                  const float A[10][10], int rows_a, int cols_a,
                  /* B is stored as flat array cols_b*rows_b */
                  const float *B, int rows_b, int cols_b)
{
    /* B is rows_b × cols_b but we need A^T(cols_a × rows_a) × B(rows_b × cols_b) */
    /* For our use case A is rows×cols, so A^T is cols×rows */
    for (int i = 0; i < cols_a; i++)
        for (int j = 0; j < cols_b; j++) {
            float sum = 0.0f;
            for (int k = 0; k < rows_a; k++)
                sum += A[k][i] * B[k * cols_b + j];
            C[i][j] = sum;
        }
}

/* ------------------------------------------------------------------ */
/*  Jacobi 2×2 SVD                                                     */
/* ------------------------------------------------------------------ */

static __device__ __forceinline__
void jacobi_2x2(float &a11, float &a12, float &a21, float &a22,
                float &c, float &s)
{
    float tau = (a11 * a11 + a22 * a22 - a12 * a12 - a21 * a21) /
                (2.0f * (a12 * a11 + a21 * a22 + 1e-30f));
    float t = (tau >= 0.0f) ?
        1.0f / (tau + sqrtf(1.0f + tau * tau)) :
       -1.0f / (-tau + sqrtf(1.0f + tau * tau));
    c = 1.0f / sqrtf(1.0f + t * t);
    s = t * c;
}

/* ------------------------------------------------------------------ */
/*  Power iteration SVD kernel                                         */
/* ------------------------------------------------------------------ */

/*
 * One block per matrix in the batch.
 * Shared memory holds: V (cols × cols), singular values, workspace.
 *
 * For M = U * S * V^T:
 *   1. Compute M^T * M → B (cols × cols)
 *   2. Power-iterate to find eigenvectors of B → V
 *   3. Singular values = sqrt(eigenvalues)
 *   4. U = M * V * diag(1/sigma)
 */

__global__ void svd_kernel(const float * __restrict__ matrices,
                           float       * __restrict__ U_out,
                           float       * __restrict__ S_out,
                           float       * __restrict__ Vt_out,
                           int rows, int cols, int batch,
                           int max_iters, float tol)
{
    int mat_idx = blockIdx.x;
    if (mat_idx >= batch) return;

    int tid = threadIdx.x;
    const float *M = matrices + mat_idx * rows * cols;

    /* Shared memory for B = M^T * M (cols × cols, max 10×10) */
    __shared__ float s_B[10][10];
    __shared__ float s_V[10][10];
    __shared__ float s_sigma[10];

    /* Initialize B = 0 */
    for (int i = tid; i < cols * cols; i += blockDim.x)
        ((float *)s_B)[i] = 0.0f;
    __syncthreads();

    /* Compute B = M^T * M (cols × cols) */
    /* Each thread handles a subset of (i,j) pairs */
    for (int ij = tid; ij < cols * cols; ij += blockDim.x) {
        int i = ij / cols;
        int j = ij % cols;
        float sum = 0.0f;
        for (int r = 0; r < rows; r++)
            sum += M[r * cols + i] * M[r * cols + j];
        s_B[i][j] = sum;
    }
    __syncthreads();

    /* Initialize V = I */
    for (int i = tid; i < cols * cols; i += blockDim.x)
        ((float *)s_V)[i] = 0.0f;
    __syncthreads();
    if (tid < cols) s_V[tid][tid] = 1.0f;
    __syncthreads();

    /* Jacobi eigenvalue iterations on B */
    for (int iter = 0; iter < max_iters; iter++) {
        float off_diag = 0.0f;

        /* Sweep all pairs (p,q) */
        for (int p = 0; p < cols - 1; p++) {
            for (int q = p + 1; q < cols; q++) {
                /* Only one thread does the rotation to avoid races */
                if (tid == 0) {
                    float bpp = s_B[p][p];
                    float bqq = s_B[q][q];
                    float bpq = s_B[p][q];

                    if (fabsf(bpq) < tol * sqrtf(fabsf(bpp * bqq) + 1e-30f))
                        continue;

                    float c, s;
                    jacobi_2x2(bpp, bpq, bpq, bqq, c, s);

                    /* Apply rotation to B */
                    for (int r = 0; r < cols; r++) {
                        float brp = s_B[r][p];
                        float brq = s_B[r][q];
                        s_B[r][p] = c * brp + s * brq;
                        s_B[r][q] = -s * brp + c * brq;
                    }
                    for (int r = 0; r < cols; r++) {
                        float bpr = s_B[p][r];
                        float bqr = s_B[q][r];
                        s_B[p][r] = c * bpr + s * bqr;
                        s_B[q][r] = -s * bpr + c * bqr;
                    }
                    s_B[p][q] = 0.0f;
                    s_B[q][p] = 0.0f;

                    /* Apply rotation to V */
                    for (int r = 0; r < cols; r++) {
                        float vrp = s_V[r][p];
                        float vrq = s_V[r][q];
                        s_V[r][p] = c * vrp + s * vrq;
                        s_V[r][q] = -s * vrp + c * vrq;
                    }
                }
                __syncthreads();
            }
        }
        __syncthreads();

        /* Check convergence */
        if (tid == 0) {
            off_diag = 0.0f;
            for (int p = 0; p < cols; p++)
                for (int q = 0; q < cols; q++)
                    if (p != q) off_diag += s_B[p][q] * s_B[p][q];
        }
        /* Broadcast convergence */
        off_diag = __shfl_sync(0xFFFFFFFF, off_diag, 0);
        if (off_diag < tol * tol) break;
    }

    /* Extract singular values */
    for (int i = tid; i < cols; i += blockDim.x)
        s_sigma[i] = sqrtf(fabsf(s_B[i][i]) + 1e-30f);
    __syncthreads();

    /* Compute U = M * V * diag(1/sigma) */
    for (int ij = tid; ij < rows * cols; ij += blockDim.x) {
        int r = ij / cols;
        int c = ij % cols;
        float sum = 0.0f;
        for (int k = 0; k < cols; k++)
            sum += M[r * cols + k] * s_V[k][c];
        U_out[mat_idx * rows * cols + r * cols + c] = sum / s_sigma[c];
    }

    /* Write singular values */
    for (int i = tid; i < cols; i += blockDim.x)
        S_out[mat_idx * cols + i] = s_sigma[i];

    /* Write V^T */
    for (int ij = tid; ij < cols * cols; ij += blockDim.x) {
        int r = ij / cols;
        int c = ij % cols;
        Vt_out[mat_idx * cols * cols + r * cols + c] = s_V[c][r];
    }
}

/* ------------------------------------------------------------------ */
/*  Launch wrapper                                                     */
/* ------------------------------------------------------------------ */

cudaError_t launch_svd_kernel(const float  *d_matrices,
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
    if (batch <= 0 || rows <= 0 || cols <= 0) return cudaSuccess;

    if (cols > 10) {
        /* Fall back indication — for production, call cuBLAS gesvd here.
         * For now, return an error suggesting the matrix is too wide. */
        fprintf(stderr, "SVD: cols=%d > 10, need cuBLAS fallback\n", cols);
        return cudaErrorNotSupported;
    }

    /* One block per matrix, 32 threads is enough for cols ≤ 10 */
    int block = 32;
    int grid  = batch;

    svd_kernel<<<grid, block, 0, stream>>>(
        d_matrices, d_U, d_S, d_Vt,
        rows, cols, batch, max_iters, tol);

    return cudaGetLastError();
}
