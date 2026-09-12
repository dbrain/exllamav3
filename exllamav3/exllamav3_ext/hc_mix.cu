#include <cstdint>
#include <cstdlib>
#include <cuda_fp16.h>
#include "hc_mix.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "graph.cuh"

/*

Fused mHC HyperConnection mix() kernel

mix(streams (R, H, D) fp32) -> post (R, H), comb (R, H, H), collapsed (R, D):

  flat = rmsnorm_unweighted(streams.flatten)          (row of H * D values)
  mixv = flat @ fn.T                                  (M = 2H + H^2 outputs)
  pre  = sigmoid(mixv[0:H]  * s0 + base[0:H]) + eps
  post = 2 sigmoid(mixv[H:2H] * s1 + base[H:2H])
  comb = sinkhorn(softmax(mixv[2H:] * s2 + base[2H:]))   (iters alternating row/col norms)
  collapsed = sum_h pre[h] * streams[h]

Two launches, no grid-wide sync, deterministic (fixed reduction order, no atomics):
  K1 partials: grid (chunksA, R); each block reduces its column chunk to M + 1 partials.
  K2 finalize: grid (chunksC, R); EVERY block re-reduces the tiny partial matrix and
    derives rmr + pre redundantly (removes the cross-block dependency), then streams its
    chunk of collapsed; the chunk-0 block also runs the sinkhorn on H^2 lanes of warp 0
    (row sums: shfl_xor 1|2, col sums: shfl_xor 4|8 for H = 4) and writes post/comb.

*/

// Partials blocks are small: at R = 1 the grid is the only parallelism, so favor many
// blocks (row_len / (4 * 64) chunks) over wide ones; the M + 1 block reduce also shrinks
#define NUM_THREADS 256
#define NUM_THREADS_A 64

__device__ __forceinline__ float sigmoidf_(float x)
{
    return 1.0f / (1.0f + __expf(-x));
}

template <int H, int M_, typename FN_T>
__global__ __launch_bounds__(NUM_THREADS_A)
void hc_mix_partials_kernel
(
    const float* __restrict__ streams,   // (R, H * D)
    const FN_T* __restrict__ fn,         // (M, H * D) float, or half (opt-in, halves traffic)
    float* __restrict__ partials,        // (R, chunksA, M + 1)
    const int row_len,
    const int chunk_cols                 // multiple of 4 * NUM_THREADS_A
)
{
    constexpr int M = M_;
    const int r = blockIdx.y;
    const int c0 = blockIdx.x * chunk_cols;
    const int c1 = min(c0 + chunk_cols, row_len);

    const float4* s4 = (const float4*) (streams + (size_t) r * row_len);
    const int row_len4 = row_len / 4;

    float acc[M + 1];
    #pragma unroll
    for (int k = 0; k <= M; ++k) acc[k] = 0.0f;

    for (int c = c0 / 4 + threadIdx.x; c < c1 / 4; c += NUM_THREADS_A)
    {
        float4 s = s4[c];
        acc[M] = fmaf(s.x, s.x, acc[M]);
        acc[M] = fmaf(s.y, s.y, acc[M]);
        acc[M] = fmaf(s.z, s.z, acc[M]);
        acc[M] = fmaf(s.w, s.w, acc[M]);
        #pragma unroll
        for (int j = 0; j < M; ++j)
        {
            float4 w;
            if constexpr (std::is_same_v<FN_T, half>)
            {
                // Single vectorized 8-byte load (two half2 loads would double the LDGs)
                int2 pk = ((const int2*) fn)[(size_t) j * row_len4 + c];
                float2 lo = __half22float2(*(const half2*) &pk.x);
                float2 hi = __half22float2(*(const half2*) &pk.y);
                w = make_float4(lo.x, lo.y, hi.x, hi.y);
            }
            else
                w = ((const float4*) fn)[(size_t) j * row_len4 + c];
            float d = fmaf(s.x, w.x, fmaf(s.y, w.y, fmaf(s.z, w.z, s.w * w.w)));
            acc[j] += d;
        }
    }

    // Block reduce M + 1 lanes' accumulators
    __shared__ float red[NUM_THREADS_A / 32][M + 1];
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    #pragma unroll
    for (int k = 0; k <= M; ++k)
    {
        float v = acc[k];
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, offset);
        if (lane == 0) red[warp][k] = v;
    }
    __syncthreads();

    if (warp == 0)
    {
        float* out = partials + ((size_t) r * gridDim.x + blockIdx.x) * (M + 1);
        for (int k = lane; k <= M; k += 32)
        {
            float v = 0.0f;
            #pragma unroll
            for (int w = 0; w < NUM_THREADS_A / 32; ++w)
                v += red[w][k];
            out[k] = v;
        }
    }
}

template <int H, int M_, bool HEAD, bool HALF_OUT>
__global__ __launch_bounds__(NUM_THREADS)
void hc_mix_finalize_kernel
(
    const float* __restrict__ streams,   // (R, H * D)
    const float* __restrict__ partials,  // (R, chunksA, M + 1)
    const float* __restrict__ base,      // (M)
    const float* __restrict__ scale,     // (3)
    float* __restrict__ post,            // (R, H)
    float* __restrict__ comb,            // (R, H, H)
    void* __restrict__ collapsed,        // (R, D) float or half
    const int D,
    const int chunksA,
    const int chunk_cols_c,              // multiple of 4
    const float rms_eps,
    const float hc_eps,
    const int sinkhorn_iters
)
{
    constexpr int M = M_;
    const int r = blockIdx.y;
    const int row_len = H * D;

    // Re-reduce this row's partials (tiny, L2-resident). Every block does this to avoid
    // cross-block dependencies. Warp-split over the chunk axis: the serial loop sits
    // at the head of the kernel's critical path, so with many partials chunks (small
    // NUM_THREADS_A blocks) a single-thread-per-quantity loop is too long
    __shared__ float mix_s[M + 1];
    __shared__ float pre_s[H];
    __shared__ float red_s[NUM_THREADS / 32][M + 1];
    {
        const int lane = threadIdx.x % 32;
        const int warp = threadIdx.x / 32;
        if (lane <= M)
        {
            const float* p = partials + (size_t) r * chunksA * (M + 1) + lane;
            float v = 0.0f;
            for (int i = warp; i < chunksA; i += NUM_THREADS / 32)
                v += p[(size_t) i * (M + 1)];
            red_s[warp][lane] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x <= M)
    {
        float v = 0.0f;
        #pragma unroll
        for (int w = 0; w < NUM_THREADS / 32; ++w)
            v += red_s[w][threadIdx.x];
        mix_s[threadIdx.x] = v;
    }
    __syncthreads();
    float rmr = rsqrtf(mix_s[M] / (float) row_len + rms_eps);
    if (threadIdx.x < H)
        pre_s[threadIdx.x] = sigmoidf_(fmaf(mix_s[threadIdx.x] * rmr, scale[0], base[threadIdx.x])) + hc_eps;
    __syncthreads();

    // Sinkhorn + post/comb writes: chunk-0 block only, one lane per comb element. Runs in
    // a dedicated warp, concurrent with the other warps' phase C: the ~20-iteration
    // normalization is a serial shfl/div latency chain that only needs the M + 1 reduced
    // scalars, so at small R it sets the kernel's critical path; don't stack phase C
    // work in front of it.
    //
    // Lane l = i * H + j; row sums reduce over j (xor 1..H/2), col sums over i (xor H..).
    const bool sink_warp = !HEAD && blockIdx.x == 0 && threadIdx.x < 32;
    if (sink_warp)
    {
        if (threadIdx.x < H)
            post[(size_t) r * H + threadIdx.x] =
                2.0f * sigmoidf_(fmaf(mix_s[H + threadIdx.x] * rmr, scale[1], base[H + threadIdx.x]));

        if (threadIdx.x < H * H)
        {
            const unsigned mask = (H * H == 32) ? 0xffffffffu : ((1u << (H * H)) - 1u);
            float v = fmaf(mix_s[2 * H + threadIdx.x] * rmr, scale[2], base[2 * H + threadIdx.x]);

            // softmax over rows
            float m = v;
            #pragma unroll
            for (int o = 1; o < H; o <<= 1) m = fmaxf(m, __shfl_xor_sync(mask, m, o));
            v = __expf(v - m);
            float s = v;
            #pragma unroll
            for (int o = 1; o < H; o <<= 1) s += __shfl_xor_sync(mask, s, o);
            v = __fdividef(v, s) + hc_eps;

            // column normalize, then (iters - 1) x (row, column)
            float cs = v;
            #pragma unroll
            for (int o = H; o < H * H; o <<= 1) cs += __shfl_xor_sync(mask, cs, o);
            v = __fdividef(v, cs + hc_eps);
            for (int it = 0; it < sinkhorn_iters - 1; ++it)
            {
                float rs = v;
                #pragma unroll
                for (int o = 1; o < H; o <<= 1) rs += __shfl_xor_sync(mask, rs, o);
                v = __fdividef(v, rs + hc_eps);
                cs = v;
                #pragma unroll
                for (int o = H; o < H * H; o <<= 1) cs += __shfl_xor_sync(mask, cs, o);
                v = __fdividef(v, cs + hc_eps);
            }
            comb[(size_t) r * H * H + threadIdx.x] = v;
        }
        return;
    }

    // Phase C: collapsed chunk, weighted sum over the H stream rows. In the sinkhorn
    // block the first warp is excluded, so the remaining threads re-cover its lanes
    float pre_r[H];
    #pragma unroll
    for (int h = 0; h < H; ++h) pre_r[h] = pre_s[h];

    const bool shrunk = !HEAD && blockIdx.x == 0;
    const int tid = shrunk ? threadIdx.x - 32 : threadIdx.x;
    const int nth = shrunk ? NUM_THREADS - 32 : NUM_THREADS;
    const int c0 = blockIdx.x * chunk_cols_c;
    const int c1 = min(c0 + chunk_cols_c, D);
    const float4* s4 = (const float4*) (streams + (size_t) r * row_len);
    const int D4 = D / 4;
    for (int c = c0 / 4 + tid; c < c1 / 4; c += nth)
    {
        float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            float4 s = s4[(size_t) h * D4 + c];
            o.x = fmaf(pre_r[h], s.x, o.x);
            o.y = fmaf(pre_r[h], s.y, o.y);
            o.z = fmaf(pre_r[h], s.z, o.z);
            o.w = fmaf(pre_r[h], s.w, o.w);
        }
        if (HALF_OUT)
        {
            half2* out2 = (half2*) ((half*) collapsed + (size_t) r * D);
            out2[c * 2] = __floats2half2_rn(o.x, o.y);
            out2[c * 2 + 1] = __floats2half2_rn(o.z, o.w);
        }
        else
            ((float4*) ((float*) collapsed + (size_t) r * D))[c] = o;
    }
}


/*

GatedResidual (Qwen4Exp, the low-rank elementwise cousin of mHC) fused mix, decode form:

  mix(streams (R, H, D) fp32) -> post (R, H), mixed (R, D):
    normed[h] = rmsnorm(streams[h]) * w[h]           (per-STREAM norm, weighted, w incl +1)
    dots      = cat(down, inject) @ normed.flatten   (M = LR + H outputs, LR = low rank ~320)
    t         = silu(dots[:LR] / H)
    post      = 2 sigmoid(dots[LR:] / H)
    g[h, d]   = sigmoid(up[h * D + d, :] @ t)        (per-CHANNEL gate through the low rank)
    mixed[d]  = mean_h g[h, d] * normed[h, d]

  Same two-launch, no-grid-sync, deterministic shape as hc_mix, restructured for the wide low
  rank (LR >> mHC's M = 24, so per-thread accumulator arrays don't fit):
    K1 gr_dots: one block per (fn row | sum-of-squares), computing per-STREAM partial dots
      against the RAW streams -- by linearity the per-stream rms scale applies in K2, and the
      norm WEIGHT is folded into the fn rows at load time.
    K2 gr_finalize: every block re-derives rmr / t redundantly from the K1 output (the mHC
      finalize pattern), then streams its chunk of mixed, evaluating the per-channel up-gate
      inline -- upT is laid out (LR, H * D) so the serial rank loop reads coalesced and stays
      L2-resident at decode R. This fp16 pair stays untiled, so it is NOT for large R (the
      per-row up/down reads defeat the L2); the int8 pair further down tiles r inside the
      block instead. Either way the python side runs a plain half-GEMM path for prefill.

  apply_ is hc_apply with no comb: x[h] += post[h] * y.

*/

#define GR_THREADS_A 128

template <int H>
__global__ __launch_bounds__(GR_THREADS_A)
void gr_dots_kernel
(
    const float* __restrict__ streams,   // (R, H, D)
    const half* __restrict__ fn,         // (M, H * D) half, norm weight folded in
    float* __restrict__ dots,            // (R, M + 1, H): per-stream dots, row M = sum sq
    const int M,
    const int D
)
{
    const int r = blockIdx.y;
    const int j = blockIdx.x;            // fn row, or M for the sum-of-squares row
    const int D4 = D / 4;
    const float4* s4 = (const float4*) (streams + (size_t) r * H * D);

    __shared__ float red[H][GR_THREADS_A / 32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;

    #pragma unroll
    for (int h = 0; h < H; ++h)
    {
        float a = 0.0f;
        if (j < M)
        {
            // 16-byte fn loads (8 halves) against two 16-byte stream quads
            const int4* f8 = (const int4*) (fn + ((size_t) j * H + h) * D);
            for (int c = threadIdx.x; c < D4 / 2; c += GR_THREADS_A)
            {
                float4 s0 = s4[(size_t) h * D4 + 2 * c];
                float4 s1 = s4[(size_t) h * D4 + 2 * c + 1];
                int4 pk = f8[c];
                half4 w0 = *(half4*) &pk.x;
                half4 w1 = *(half4*) &pk.z;
                a = fmaf(s0.x, LOW_TO_FLOAT(w0.x), a);
                a = fmaf(s0.y, HIGH_TO_FLOAT(w0.x), a);
                a = fmaf(s0.z, LOW_TO_FLOAT(w0.y), a);
                a = fmaf(s0.w, HIGH_TO_FLOAT(w0.y), a);
                a = fmaf(s1.x, LOW_TO_FLOAT(w1.x), a);
                a = fmaf(s1.y, HIGH_TO_FLOAT(w1.x), a);
                a = fmaf(s1.z, LOW_TO_FLOAT(w1.y), a);
                a = fmaf(s1.w, HIGH_TO_FLOAT(w1.y), a);
            }
        }
        else
        {
            for (int c = threadIdx.x; c < D4; c += GR_THREADS_A)
            {
                float4 s = s4[(size_t) h * D4 + c];
                a = fmaf(s.x, s.x, fmaf(s.y, s.y, fmaf(s.z, s.z, fmaf(s.w, s.w, a))));
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            a += __shfl_down_sync(0xffffffffu, a, offset);
        if (lane == 0) red[h][warp] = a;
    }
    __syncthreads();
    if (threadIdx.x < H)
    {
        float v = 0.0f;
        #pragma unroll
        for (int w = 0; w < GR_THREADS_A / 32; ++w)
            v += red[threadIdx.x][w];
        dots[((size_t) r * (M + 1) + j) * H + threadIdx.x] = v;
    }
}

template <int H, bool HALF_OUT>
__global__ __launch_bounds__(NUM_THREADS)
void gr_finalize_kernel
(
    const float* __restrict__ streams,   // (R, H, D)
    const float* __restrict__ dots,      // (R, M + 1, H) from gr_dots
    const half* __restrict__ upt,        // (H, D / 4, LR, 4) half: up repacked lane-contiguous
    const half* __restrict__ w,          // (H * D) half norm weight (incl +1)
    float* __restrict__ post,            // (R, H) or nullptr (final-mixer form)
    void* __restrict__ mixed,            // (R, D) half or float
    const int D,
    const int LR,                        // low rank; M = LR + (post ? H : 0)
    const int chunk_cols,                // multiple of 4
    const float rms_eps
)
{
    const int r = blockIdx.y;
    const int M = LR + (post ? H : 0);
    const float* dr = dots + (size_t) r * (M + 1) * H;

    // Redundant per-block head derivation (no cross-block dependency): rmr from the sumsq
    // row, then the silu'd low-rank activations into shared memory
    __shared__ float rmr_s[H];
    extern __shared__ float t_s[];
    if (threadIdx.x < H)
        rmr_s[threadIdx.x] = rsqrtf(dr[(size_t) M * H + threadIdx.x] / (float) D + rms_eps);
    __syncthreads();
    const float inv_h = 1.0f / (float) H;
    for (int i = threadIdx.x; i < LR; i += NUM_THREADS)
    {
        float v = 0.0f;
        #pragma unroll
        for (int h = 0; h < H; ++h)
            v = fmaf(rmr_s[h], dr[(size_t) i * H + h], v);
        v *= inv_h;
        t_s[i] = v * sigmoidf_(v);
    }
    if (post && blockIdx.x == 0 && threadIdx.x < H)
    {
        float v = 0.0f;
        #pragma unroll
        for (int h = 0; h < H; ++h)
            v = fmaf(rmr_s[h], dr[(size_t) (LR + threadIdx.x) * H + h], v);
        post[(size_t) r * H + threadIdx.x] = 2.0f * sigmoidf_(v * inv_h);
    }
    __syncthreads();

    // Streamed mixed chunk: one WARP per column quad, the LR-long up-gate dots split across
    // the lanes (lane l covers ranks l, l+32, ...) and shfl-reduced -- at decode R the total
    // column count is small (R * D / 4), so per-thread columns would leave the GPU nearly
    // idle with each thread serializing the rank loop
    const int c0 = blockIdx.x * chunk_cols;
    const int c1 = min(c0 + chunk_cols, D);
    const int D4 = D / 4;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const float4* s4 = (const float4*) (streams + (size_t) r * H * D);
    for (int c = c0 / 4 + warp; c < c1 / 4; c += NUM_THREADS / 32)
    {
        float4 g[H];
        #pragma unroll
        for (int h = 0; h < H; ++h) g[h] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int i = lane; i < LR; i += 32)
        {
            float ti = t_s[i];
            #pragma unroll
            for (int h = 0; h < H; ++h)
            {
                // upx layout (H, D / 4, LR, 4): consecutive lanes (consecutive i) read
                // consecutive 8-byte quads, so the rank loop is fully coalesced
                half4 u = *(const half4*) (upt + ((((size_t) h * (D / 4) + c) * LR + i) * 4));
                g[h].x = fmaf(ti, LOW_TO_FLOAT(u.x), g[h].x);
                g[h].y = fmaf(ti, HIGH_TO_FLOAT(u.x), g[h].y);
                g[h].z = fmaf(ti, LOW_TO_FLOAT(u.y), g[h].z);
                g[h].w = fmaf(ti, HIGH_TO_FLOAT(u.y), g[h].w);
            }
        }
        #pragma unroll
        for (int h = 0; h < H; ++h)
            for (int offset = 16; offset > 0; offset >>= 1)
            {
                g[h].x += __shfl_xor_sync(0xffffffffu, g[h].x, offset);
                g[h].y += __shfl_xor_sync(0xffffffffu, g[h].y, offset);
                g[h].z += __shfl_xor_sync(0xffffffffu, g[h].z, offset);
                g[h].w += __shfl_xor_sync(0xffffffffu, g[h].w, offset);
            }
        if (lane != 0) continue;
        float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            float4 s = s4[(size_t) h * D4 + c];
            half4 wq = *(const half4*) (w + (size_t) h * D + 4 * c);
            float coef = rmr_s[h] * inv_h;
            o.x = fmaf(sigmoidf_(g[h].x) * coef * LOW_TO_FLOAT(wq.x),  s.x, o.x);
            o.y = fmaf(sigmoidf_(g[h].y) * coef * HIGH_TO_FLOAT(wq.x), s.y, o.y);
            o.z = fmaf(sigmoidf_(g[h].z) * coef * LOW_TO_FLOAT(wq.y),  s.z, o.z);
            o.w = fmaf(sigmoidf_(g[h].w) * coef * HIGH_TO_FLOAT(wq.y), s.w, o.w);
        }
        if (HALF_OUT)
        {
            half2* out2 = (half2*) ((half*) mixed + (size_t) r * D);
            out2[c * 2] = __floats2half2_rn(o.x, o.y);
            out2[c * 2 + 1] = __floats2half2_rn(o.z, o.w);
        }
        else
            ((float4*) ((float*) mixed + (size_t) r * D))[c] = o;
    }
}


/*

int8 variants of the gr_mix pair (EXL3_HC_QUANT=int8, see modules/hyperconnections.py).

fn (M, H * D) and up (H, D / 4, LR, 4) are stored symmetric-int8 with an fp16 scale along the
NON-reduction axis of their matmul: per fn row, and per output channel (h, d) for up. Both dots
are linear in the weight, so the scale is one multiply on the finished accumulator -- zero
per-element ALU beyond the int-to-float convert -- and the up scales are laid out exactly like
the norm weight w, so the finalize kernel folds them into its existing half4 quad read.

*/

template <int H>
__global__ __launch_bounds__(GR_THREADS_A)
void gr_dots_q8_kernel
(
    const float* __restrict__ streams,   // (R, H, D)
    const int8_t* __restrict__ fn,       // (M, H * D) int8
    const half* __restrict__ fn_scale,   // (M, H)
    float* __restrict__ dots,            // (R, M + 1, H)
    const int M,
    const int D
)
{
    const int r = blockIdx.y;
    const int j = blockIdx.x;
    const int D4 = D / 4;
    const float4* s4 = (const float4*) (streams + (size_t) r * H * D);

    __shared__ float red[H][GR_THREADS_A / 32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;

    #pragma unroll
    for (int h = 0; h < H; ++h)
    {
        float a = 0.0f;
        if (j < M)
        {
            // 8-byte fn loads (8 int8) against two 16-byte stream quads, mirroring the fp16 path
            const int2* f8 = (const int2*) (fn + ((size_t) j * H + h) * D);
            for (int c = threadIdx.x; c < D4 / 2; c += GR_THREADS_A)
            {
                float4 s0 = s4[(size_t) h * D4 + 2 * c];
                float4 s1 = s4[(size_t) h * D4 + 2 * c + 1];
                int2 pk = f8[c];
                char4 q0 = *(char4*) &pk.x;
                char4 q1 = *(char4*) &pk.y;
                a = fmaf(s0.x, (float) q0.x, a);
                a = fmaf(s0.y, (float) q0.y, a);
                a = fmaf(s0.z, (float) q0.z, a);
                a = fmaf(s0.w, (float) q0.w, a);
                a = fmaf(s1.x, (float) q1.x, a);
                a = fmaf(s1.y, (float) q1.y, a);
                a = fmaf(s1.z, (float) q1.z, a);
                a = fmaf(s1.w, (float) q1.w, a);
            }
        }
        else
        {
            for (int c = threadIdx.x; c < D4; c += GR_THREADS_A)
            {
                float4 s = s4[(size_t) h * D4 + c];
                a = fmaf(s.x, s.x, fmaf(s.y, s.y, fmaf(s.z, s.z, fmaf(s.w, s.w, a))));
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            a += __shfl_down_sync(0xffffffffu, a, offset);
        if (lane == 0) red[h][warp] = a;
    }
    __syncthreads();
    if (threadIdx.x < H)
    {
        float v = 0.0f;
        #pragma unroll
        for (int w = 0; w < GR_THREADS_A / 32; ++w)
            v += red[threadIdx.x][w];
        // Row M is the sum of squares of the raw streams, not a weight dot: unscaled
        if (j < M) v *= __half2float(fn_scale[(size_t) j * H + threadIdx.x]);
        dots[((size_t) r * (M + 1) + j) * H + threadIdx.x] = v;
    }
}

template <int H, bool HALF_OUT>
__global__ __launch_bounds__(NUM_THREADS)
void gr_finalize_q8_kernel
(
    const float* __restrict__ streams,   // (R, H, D)
    const float* __restrict__ dots,      // (R, M + 1, H) from gr_dots_q8
    const int8_t* __restrict__ upt,      // (H, D / 4, LR, 4) int8
    const half* __restrict__ upt_scale,  // (H * D) per output channel, same layout as w
    const half* __restrict__ w,          // (H * D) half norm weight (incl +1)
    float* __restrict__ post,            // (R, H) or nullptr (final-mixer form)
    void* __restrict__ mixed,            // (R, D) half or float
    const int D,
    const int LR,
    const int chunk_cols,
    const float rms_eps
)
{
    const int r = blockIdx.y;
    const int M = LR + (post ? H : 0);
    const float* dr = dots + (size_t) r * (M + 1) * H;

    __shared__ float rmr_s[H];
    extern __shared__ float t_s[];
    if (threadIdx.x < H)
        rmr_s[threadIdx.x] = rsqrtf(dr[(size_t) M * H + threadIdx.x] / (float) D + rms_eps);
    __syncthreads();
    const float inv_h = 1.0f / (float) H;
    for (int i = threadIdx.x; i < LR; i += NUM_THREADS)
    {
        float v = 0.0f;
        #pragma unroll
        for (int h = 0; h < H; ++h)
            v = fmaf(rmr_s[h], dr[(size_t) i * H + h], v);
        v *= inv_h;
        t_s[i] = v * sigmoidf_(v);
    }
    if (post && blockIdx.x == 0 && threadIdx.x < H)
    {
        float v = 0.0f;
        #pragma unroll
        for (int h = 0; h < H; ++h)
            v = fmaf(rmr_s[h], dr[(size_t) (LR + threadIdx.x) * H + h], v);
        post[(size_t) r * H + threadIdx.x] = 2.0f * sigmoidf_(v * inv_h);
    }
    __syncthreads();

    const int c0 = blockIdx.x * chunk_cols;
    const int c1 = min(c0 + chunk_cols, D);
    const int D4 = D / 4;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const float4* s4 = (const float4*) (streams + (size_t) r * H * D);
    for (int c = c0 / 4 + warp; c < c1 / 4; c += NUM_THREADS / 32)
    {
        float4 g[H];
        #pragma unroll
        for (int h = 0; h < H; ++h) g[h] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int i = lane; i < LR; i += 32)
        {
            float ti = t_s[i];
            #pragma unroll
            for (int h = 0; h < H; ++h)
            {
                char4 u = *(const char4*) (upt + ((((size_t) h * (D / 4) + c) * LR + i) * 4));
                g[h].x = fmaf(ti, (float) u.x, g[h].x);
                g[h].y = fmaf(ti, (float) u.y, g[h].y);
                g[h].z = fmaf(ti, (float) u.z, g[h].z);
                g[h].w = fmaf(ti, (float) u.w, g[h].w);
            }
        }
        #pragma unroll
        for (int h = 0; h < H; ++h)
            for (int offset = 16; offset > 0; offset >>= 1)
            {
                g[h].x += __shfl_xor_sync(0xffffffffu, g[h].x, offset);
                g[h].y += __shfl_xor_sync(0xffffffffu, g[h].y, offset);
                g[h].z += __shfl_xor_sync(0xffffffffu, g[h].z, offset);
                g[h].w += __shfl_xor_sync(0xffffffffu, g[h].w, offset);
            }
        if (lane != 0) continue;
        float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            float4 s = s4[(size_t) h * D4 + c];
            half4 wq = *(const half4*) (w + (size_t) h * D + 4 * c);
            half4 uq = *(const half4*) (upt_scale + (size_t) h * D + 4 * c);
            float coef = rmr_s[h] * inv_h;
            o.x = fmaf(sigmoidf_(g[h].x * LOW_TO_FLOAT(uq.x))  * coef * LOW_TO_FLOAT(wq.x),  s.x, o.x);
            o.y = fmaf(sigmoidf_(g[h].y * HIGH_TO_FLOAT(uq.x)) * coef * HIGH_TO_FLOAT(wq.x), s.y, o.y);
            o.z = fmaf(sigmoidf_(g[h].z * LOW_TO_FLOAT(uq.y))  * coef * LOW_TO_FLOAT(wq.y),  s.z, o.z);
            o.w = fmaf(sigmoidf_(g[h].w * HIGH_TO_FLOAT(uq.y)) * coef * HIGH_TO_FLOAT(wq.y), s.w, o.w);
        }
        if (HALF_OUT)
        {
            half2* out2 = (half2*) ((half*) mixed + (size_t) r * D);
            out2[c * 2] = __floats2half2_rn(o.x, o.y);
            out2[c * 2 + 1] = __floats2half2_rn(o.z, o.w);
        }
        else
            ((float4*) ((float*) mixed + (size_t) r * D))[c] = o;
    }
}

/*

R-tiled int8 pair. The untiled kernels above grid on (., R) with r = blockIdx.y, so block
(j, r) and block (j, r') read the same weight bytes: a call at R rows issues R * (fn + upt),
6.6 MB * R at the Flash-Next shape, against a 86.4 GB/s device. These tile GR_R_TILE stream
rows INSIDE the block instead, so one weight read serves all of them and a call issues
ceil(R / GR_R_TILE) * 6.6 MB. Streams and dots stay per-row but are two orders of magnitude
smaller and live in L2.

The accumulation and reduction orders are untouched -- same lane-strided rank walk, same
fixed shfl_xor butterfly, no atomics -- so output is bitwise identical to the untiled path at
every R (tests/test_hc_mix_q8_rtile.py pins that, including row invariance vs R = 1).

A partial tail tile skips its surplus rows in gr_dots, where each one would cost a whole
streams slice re-read per fn row, and clamps them onto the last valid row in gr_finalize,
where the surplus is LDS reads and FMAs against a weight byte already in register.

GR_R_TILE costs R_TILE * H float4 of accumulator per lane (16 VGPRs per row) and R_TILE * LR
floats of LDS, and the trade turns over before the traffic does. At the deployed R = 7, 8
reads the weight ONCE against 4's twice and is still 29% SLOWER end to end (HC.mix 43.2-43.5
vs 33.4-33.4 ms, perf/exl3/verifysplit.py, n = 2 processes each): gr_finalize_q8_rt at
R_TILE = 8 hits the 256-VGPR ceiling and spills -- vgpr_count 256, vgpr_spill_count 76,
private_segment_fixed_size 308 -- where R_TILE = 4 sits at 185 with zero spill. Read those
off the code object (llvm-readelf --notes) before raising this on a byte count.

Below GR_RTILE_MIN_R there is no redundancy worth removing and blockIdx.y is parallelism the
small blocks need, so R = 1 decode keeps the untiled geometry.

*/

#define GR_R_TILE 4
#define GR_RTILE_MIN_R 2

// EXL3_HC_R_TILE: pick between the compiled R_TILE widths per call, so both arms interleave
// in one process instead of costing a rebuild and a cross-process comparison. Only 2 and 4
// are instantiated: 8 was measured (+29-30%, vgpr 256 / spill 76 / scratch 308 B) and is not
// compiled, so nothing can select it by accident.
static inline int gr_r_tile_env()
{
    if (const char* e = getenv("EXL3_HC_R_TILE"))
    {
        int v = atoi(e);
        if (v == 2 || v == 4) return v;
    }
    return GR_R_TILE;
}

// EXL3_HC_J_TILE: fn rows per gr_dots block. 1 = the one-row-per-block kernel. Read per call.
#define GR_J_TILE 1

static inline int gr_j_tile_env()
{
    if (const char* e = getenv("EXL3_HC_J_TILE"))
    {
        int v = atoi(e);
        if (v == 1 || v == 2 || v == 4 || v == 8) return v;
    }
    return GR_J_TILE;
}

template <int H, int R_TILE>
__global__ __launch_bounds__(GR_THREADS_A)
void gr_dots_q8_rt_kernel
(
    const float* __restrict__ streams,   // (R, H, D)
    const int8_t* __restrict__ fn,       // (M, H * D) int8
    const half* __restrict__ fn_scale,   // (M, H)
    float* __restrict__ dots,            // (R, M + 1, H)
    const int M,
    const int D,
    const int R
)
{
    const int r0 = blockIdx.y * R_TILE;
    const int nr = min(R_TILE, R - r0);
    const int j = blockIdx.x;
    const int D4 = D / 4;

    const float4* s4[R_TILE];
    #pragma unroll
    for (int t = 0; t < R_TILE; ++t)
        s4[t] = (const float4*) (streams + (size_t) min(r0 + t, R - 1) * H * D);

    __shared__ float red[R_TILE][H][GR_THREADS_A / 32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;

    #pragma unroll
    for (int h = 0; h < H; ++h)
    {
        float a[R_TILE];
        #pragma unroll
        for (int t = 0; t < R_TILE; ++t) a[t] = 0.0f;
        if (j < M)
        {
            const int2* f8 = (const int2*) (fn + ((size_t) j * H + h) * D);
            for (int c = threadIdx.x; c < D4 / 2; c += GR_THREADS_A)
            {
                int2 pk = f8[c];
                char4 q0 = *(char4*) &pk.x;
                char4 q1 = *(char4*) &pk.y;
                #pragma unroll
                for (int t = 0; t < R_TILE; ++t)
                {
                    if (t >= nr) break;
                    float4 s0 = s4[t][(size_t) h * D4 + 2 * c];
                    float4 s1 = s4[t][(size_t) h * D4 + 2 * c + 1];
                    a[t] = fmaf(s0.x, (float) q0.x, a[t]);
                    a[t] = fmaf(s0.y, (float) q0.y, a[t]);
                    a[t] = fmaf(s0.z, (float) q0.z, a[t]);
                    a[t] = fmaf(s0.w, (float) q0.w, a[t]);
                    a[t] = fmaf(s1.x, (float) q1.x, a[t]);
                    a[t] = fmaf(s1.y, (float) q1.y, a[t]);
                    a[t] = fmaf(s1.z, (float) q1.z, a[t]);
                    a[t] = fmaf(s1.w, (float) q1.w, a[t]);
                }
            }
        }
        else
        {
            for (int c = threadIdx.x; c < D4; c += GR_THREADS_A)
            {
                #pragma unroll
                for (int t = 0; t < R_TILE; ++t)
                {
                    if (t >= nr) break;
                    float4 s = s4[t][(size_t) h * D4 + c];
                    a[t] = fmaf(s.x, s.x, fmaf(s.y, s.y, fmaf(s.z, s.z, fmaf(s.w, s.w, a[t]))));
                }
            }
        }
        #pragma unroll
        for (int t = 0; t < R_TILE; ++t)
        {
            if (t >= nr) break;
            for (int offset = 16; offset > 0; offset >>= 1)
                a[t] += __shfl_down_sync(0xffffffffu, a[t], offset);
            if (lane == 0) red[t][h][warp] = a[t];
        }
    }
    __syncthreads();
    if (threadIdx.x < nr * H)
    {
        const int t = threadIdx.x / H;
        const int hh = threadIdx.x % H;
        float v = 0.0f;
        #pragma unroll
        for (int w = 0; w < GR_THREADS_A / 32; ++w)
            v += red[t][hh][w];
        // Row M is the sum of squares of the raw streams, not a weight dot: unscaled
        if (j < M) v *= __half2float(fn_scale[(size_t) j * H + hh]);
        dots[((size_t) (r0 + t) * (M + 1) + j) * H + hh] = v;
    }
}

/*

J-tiled gr_dots. The R-tiled kernel above grids on (M + 1, n_rtiles) and gives each block ONE
fn row, so every block re-reads the SAME R_TILE stream rows: per r-tile it reads 3.32 MB of fn
and 325 x R_TILE x H x D x 4 B = 53.2 MB of streams at the Flash-Next shape. The streams are
164 KB and L2-resident, so this is not DRAM traffic -- but it is 16x the weight traffic through
L2, and it is what the measured cost model says dominates the call:

    us(R, tiles) ~= 51 fixed + 98 per r-tile + 18.8 PER ROW   (perf/exl3/ledger-hcbound.csv)

98 us for the 6.59 MB a tile reads is 67 GB/s against the 86.4 ceiling -- the weight side is
already near the wall. The 18.8 us/row is 13.3 MB of stream re-reads at ~700 GB/s, i.e. L2
rate, and at the deployed R = 7 it is 132 of ~379 us, 35% of the call.

J_TILE fn rows per block divides that term by J_TILE: the streams for a c-slice are loaded once
and applied to J_TILE weight rows. Nothing else changes -- for a fixed (t, u) the accumulation
walks c in exactly the original order with exactly the original 8 fmas, so `a` is bit-identical
to the untiled kernel's. The row-M norm branch (sum of squares of the raw streams) keeps its own
c loop at its own stride precisely so its per-lane partition is unchanged; folding it into the
J loop would re-associate it and break bit-exactness against the fp16 path.

Registers are the risk, not the arithmetic: `float a[R_TILE][J_TILE]` plus the hoisted
`float4 s0/s1[R_TILE]`. Read vgpr_count / vgpr_spill_count off the code object before believing
any timing -- three levers on this part have died at a wider tile (perf/exl3/kstats.sh).

*/

template <int H, int R_TILE, int J_TILE>
__global__ __launch_bounds__(GR_THREADS_A)
void gr_dots_q8_rtj_kernel
(
    const float* __restrict__ streams,   // (R, H, D)
    const int8_t* __restrict__ fn,       // (M, H * D) int8
    const half* __restrict__ fn_scale,   // (M, H)
    float* __restrict__ dots,            // (R, M + 1, H)
    const int M,
    const int D,
    const int R
)
{
    const int r0 = blockIdx.y * R_TILE;
    const int nr = min(R_TILE, R - r0);
    const int j0 = blockIdx.x * J_TILE;
    const int nj = min(J_TILE, (M + 1) - j0);
    if (nj <= 0) return;
    const int nj_fn = min(nj, M - j0);   // rows in this block that are real fn rows
    const int D4 = D / 4;

    const float4* s4[R_TILE];
    #pragma unroll
    for (int t = 0; t < R_TILE; ++t)
        s4[t] = (const float4*) (streams + (size_t) min(r0 + t, R - 1) * H * D);

    __shared__ float red[R_TILE][J_TILE][H][GR_THREADS_A / 32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;

    #pragma unroll
    for (int h = 0; h < H; ++h)
    {
        float a[R_TILE][J_TILE];
        #pragma unroll
        for (int t = 0; t < R_TILE; ++t)
            #pragma unroll
            for (int u = 0; u < J_TILE; ++u) a[t][u] = 0.0f;

        if (nj_fn > 0)
        {
            for (int c = threadIdx.x; c < D4 / 2; c += GR_THREADS_A)
            {
                float4 s0[R_TILE], s1[R_TILE];
                #pragma unroll
                for (int t = 0; t < R_TILE; ++t)
                {
                    if (t >= nr) break;
                    s0[t] = s4[t][(size_t) h * D4 + 2 * c];
                    s1[t] = s4[t][(size_t) h * D4 + 2 * c + 1];
                }
                #pragma unroll
                for (int u = 0; u < J_TILE; ++u)
                {
                    if (u >= nj_fn) break;
                    const int2* f8 = (const int2*) (fn + ((size_t) (j0 + u) * H + h) * D);
                    int2 pk = f8[c];
                    char4 q0 = *(char4*) &pk.x;
                    char4 q1 = *(char4*) &pk.y;
                    #pragma unroll
                    for (int t = 0; t < R_TILE; ++t)
                    {
                        if (t >= nr) break;
                        a[t][u] = fmaf(s0[t].x, (float) q0.x, a[t][u]);
                        a[t][u] = fmaf(s0[t].y, (float) q0.y, a[t][u]);
                        a[t][u] = fmaf(s0[t].z, (float) q0.z, a[t][u]);
                        a[t][u] = fmaf(s0[t].w, (float) q0.w, a[t][u]);
                        a[t][u] = fmaf(s1[t].x, (float) q1.x, a[t][u]);
                        a[t][u] = fmaf(s1[t].y, (float) q1.y, a[t][u]);
                        a[t][u] = fmaf(s1[t].z, (float) q1.z, a[t][u]);
                        a[t][u] = fmaf(s1[t].w, (float) q1.w, a[t][u]);
                    }
                }
            }
        }
        if (nj_fn < nj)
        {
            // Row M: sum of squares of the raw streams. Own loop, own stride, unchanged.
            const int u = nj_fn;
            for (int c = threadIdx.x; c < D4; c += GR_THREADS_A)
            {
                #pragma unroll
                for (int t = 0; t < R_TILE; ++t)
                {
                    if (t >= nr) break;
                    float4 s = s4[t][(size_t) h * D4 + c];
                    a[t][u] = fmaf(s.x, s.x, fmaf(s.y, s.y, fmaf(s.z, s.z,
                                   fmaf(s.w, s.w, a[t][u]))));
                }
            }
        }
        #pragma unroll
        for (int t = 0; t < R_TILE; ++t)
        {
            if (t >= nr) break;
            #pragma unroll
            for (int u = 0; u < J_TILE; ++u)
            {
                if (u >= nj) break;
                for (int offset = 16; offset > 0; offset >>= 1)
                    a[t][u] += __shfl_down_sync(0xffffffffu, a[t][u], offset);
                if (lane == 0) red[t][u][h][warp] = a[t][u];
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < nr * nj * H)
    {
        const int t = threadIdx.x / (nj * H);
        const int rem = threadIdx.x % (nj * H);
        const int u = rem / H;
        const int hh = rem % H;
        const int j = j0 + u;
        float v = 0.0f;
        #pragma unroll
        for (int w = 0; w < GR_THREADS_A / 32; ++w)
            v += red[t][u][hh][w];
        if (j < M) v *= __half2float(fn_scale[(size_t) j * H + hh]);
        dots[((size_t) (r0 + t) * (M + 1) + j) * H + hh] = v;
    }
}

template <int H, int R_TILE, bool HALF_OUT>
__global__ __launch_bounds__(NUM_THREADS)
void gr_finalize_q8_rt_kernel
(
    const float* __restrict__ streams,   // (R, H, D)
    const float* __restrict__ dots,      // (R, M + 1, H) from gr_dots_q8_rt
    const int8_t* __restrict__ upt,      // (H, D / 4, LR, 4) int8
    const half* __restrict__ upt_scale,  // (H * D) per output channel, same layout as w
    const half* __restrict__ w,          // (H * D) half norm weight (incl +1)
    float* __restrict__ post,            // (R, H) or nullptr (final-mixer form)
    void* __restrict__ mixed,            // (R, D) half or float
    const int D,
    const int LR,
    const int chunk_cols,
    const float rms_eps,
    const int R
)
{
    const int r0 = blockIdx.y * R_TILE;
    const int nr = min(R_TILE, R - r0);
    const int M = LR + (post ? H : 0);

    __shared__ float rmr_s[R_TILE][H];
    extern __shared__ float t_s[];       // (R_TILE, LR)

    const float* dr[R_TILE];
    #pragma unroll
    for (int t = 0; t < R_TILE; ++t)
        dr[t] = dots + (size_t) (r0 + min(t, nr - 1)) * (M + 1) * H;

    if (threadIdx.x < R_TILE * H)
    {
        const int t = threadIdx.x / H;
        const int hh = threadIdx.x % H;
        rmr_s[t][hh] = rsqrtf(dr[t][(size_t) M * H + hh] / (float) D + rms_eps);
    }
    __syncthreads();
    const float inv_h = 1.0f / (float) H;
    #pragma unroll
    for (int t = 0; t < R_TILE; ++t)
        for (int i = threadIdx.x; i < LR; i += NUM_THREADS)
        {
            float v = 0.0f;
            #pragma unroll
            for (int h = 0; h < H; ++h)
                v = fmaf(rmr_s[t][h], dr[t][(size_t) i * H + h], v);
            v *= inv_h;
            t_s[t * LR + i] = v * sigmoidf_(v);
        }
    if (post && blockIdx.x == 0 && threadIdx.x < nr * H)
    {
        const int t = threadIdx.x / H;
        const int hh = threadIdx.x % H;
        float v = 0.0f;
        #pragma unroll
        for (int h = 0; h < H; ++h)
            v = fmaf(rmr_s[t][h], dr[t][(size_t) (LR + hh) * H + h], v);
        post[(size_t) (r0 + t) * H + hh] = 2.0f * sigmoidf_(v * inv_h);
    }
    __syncthreads();

    const int c0 = blockIdx.x * chunk_cols;
    const int c1 = min(c0 + chunk_cols, D);
    const int D4 = D / 4;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    for (int c = c0 / 4 + warp; c < c1 / 4; c += NUM_THREADS / 32)
    {
        float4 g[R_TILE][H];
        #pragma unroll
        for (int t = 0; t < R_TILE; ++t)
            #pragma unroll
            for (int h = 0; h < H; ++h) g[t][h] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int i = lane; i < LR; i += 32)
        {
            float ti[R_TILE];
            #pragma unroll
            for (int t = 0; t < R_TILE; ++t) ti[t] = t_s[t * LR + i];
            #pragma unroll
            for (int h = 0; h < H; ++h)
            {
                char4 u = *(const char4*) (upt + ((((size_t) h * (D / 4) + c) * LR + i) * 4));
                #pragma unroll
                for (int t = 0; t < R_TILE; ++t)
                {
                    g[t][h].x = fmaf(ti[t], (float) u.x, g[t][h].x);
                    g[t][h].y = fmaf(ti[t], (float) u.y, g[t][h].y);
                    g[t][h].z = fmaf(ti[t], (float) u.z, g[t][h].z);
                    g[t][h].w = fmaf(ti[t], (float) u.w, g[t][h].w);
                }
            }
        }
        #pragma unroll
        for (int t = 0; t < R_TILE; ++t)
            #pragma unroll
            for (int h = 0; h < H; ++h)
                for (int offset = 16; offset > 0; offset >>= 1)
                {
                    g[t][h].x += __shfl_xor_sync(0xffffffffu, g[t][h].x, offset);
                    g[t][h].y += __shfl_xor_sync(0xffffffffu, g[t][h].y, offset);
                    g[t][h].z += __shfl_xor_sync(0xffffffffu, g[t][h].z, offset);
                    g[t][h].w += __shfl_xor_sync(0xffffffffu, g[t][h].w, offset);
                }
        if (lane != 0) continue;
        #pragma unroll
        for (int t = 0; t < R_TILE; ++t)
        {
            if (t >= nr) break;
            const int r = r0 + t;
            const float4* s4 = (const float4*) (streams + (size_t) r * H * D);
            float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            #pragma unroll
            for (int h = 0; h < H; ++h)
            {
                float4 s = s4[(size_t) h * D4 + c];
                half4 wq = *(const half4*) (w + (size_t) h * D + 4 * c);
                half4 uq = *(const half4*) (upt_scale + (size_t) h * D + 4 * c);
                float coef = rmr_s[t][h] * inv_h;
                o.x = fmaf(sigmoidf_(g[t][h].x * LOW_TO_FLOAT(uq.x))  * coef * LOW_TO_FLOAT(wq.x),  s.x, o.x);
                o.y = fmaf(sigmoidf_(g[t][h].y * HIGH_TO_FLOAT(uq.x)) * coef * HIGH_TO_FLOAT(wq.x), s.y, o.y);
                o.z = fmaf(sigmoidf_(g[t][h].z * LOW_TO_FLOAT(uq.y))  * coef * LOW_TO_FLOAT(wq.y),  s.z, o.z);
                o.w = fmaf(sigmoidf_(g[t][h].w * HIGH_TO_FLOAT(uq.y)) * coef * HIGH_TO_FLOAT(wq.y), s.w, o.w);
            }
            if (HALF_OUT)
            {
                half2* out2 = (half2*) ((half*) mixed + (size_t) r * D);
                out2[c * 2] = __floats2half2_rn(o.x, o.y);
                out2[c * 2 + 1] = __floats2half2_rn(o.z, o.w);
            }
            else
                ((float4*) ((float*) mixed + (size_t) r * D))[c] = o;
        }
    }
}



/*

int4 variants of the gr_mix pair (EXL3_HC_QUANT=int4).

Symmetric 4-bit codes in [-7, 7], stored biased by +8 in a nibble, with fp16 GROUP scales
along the REDUCTION axis of each matmul. int8 could carry a single scale per output slice
because 8 bits spans the dynamic range of a whole 2560- or 320-long row; 4 bits cannot, so
the scale axis moves and the dequant becomes one multiply per group instead of one per dot.

  fn  (M, H, D) -> (M, H * D / 2) uint8, byte k = code 2k (low nibble) | code 2k + 1 (high),
      scales (M, H, D / Q4_FN_GROUP): the reduction runs over D within one stream slice.
  upx (H, D/4, LR, 4) -> (H, D/4, LR/2, 4) uint8, byte (.., p, k) = rank 2p (low) | 2p + 1
      (high) of output channel 4c + k; scales (LR / Q4_UP_GROUP, H * D), group-major so the
      four channels of a quad still arrive in one half4 read, exactly like w.

Q4_UP_GROUP = 64 is not a free parameter: gr_finalize walks the rank with a stride of one
warp over PAIRS, so 32 pairs = 64 ranks makes the loop index and the group index the same
number and the group scale needs no division. LR must be a multiple of 64.
Q4_FN_GROUP is a power of two >= 16 (one thread step is 16 codes) and is passed as a shift.

*/

#define Q4_UP_GROUP 64

// Nibble n of the word is code n, so the codes come out in order for both packings
__device__ __forceinline__ void q4_unpack8(uint32_t b, float4& lo, float4& hi)
{
    lo.x = (float) ((int) ((b >>  0) & 0xfu) - 8);
    lo.y = (float) ((int) ((b >>  4) & 0xfu) - 8);
    lo.z = (float) ((int) ((b >>  8) & 0xfu) - 8);
    lo.w = (float) ((int) ((b >> 12) & 0xfu) - 8);
    hi.x = (float) ((int) ((b >> 16) & 0xfu) - 8);
    hi.y = (float) ((int) ((b >> 20) & 0xfu) - 8);
    hi.z = (float) ((int) ((b >> 24) & 0xfu) - 8);
    hi.w = (float) ((int) ((b >> 28) & 0xfu) - 8);
}

template <int H>
__global__ __launch_bounds__(GR_THREADS_A)
void gr_dots_q4_kernel
(
    const float* __restrict__ streams,   // (R, H, D)
    const uint8_t* __restrict__ fn,      // (M, H * D / 2) packed int4
    const half* __restrict__ fn_scale,   // (M, H, D / Q4_FN_GROUP)
    float* __restrict__ dots,            // (R, M + 1, H)
    const int M,
    const int D,
    const int n_groups,
    const int g_shift                    // log2(Q4_FN_GROUP / 16)
)
{
    const int r = blockIdx.y;
    const int j = blockIdx.x;
    const int D4 = D / 4;
    const int D16 = D / 16;
    const float4* s4 = (const float4*) (streams + (size_t) r * H * D);

    __shared__ float red[H][GR_THREADS_A / 32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;

    #pragma unroll
    for (int h = 0; h < H; ++h)
    {
        float a = 0.0f;
        if (j < M)
        {
            // 8-byte load = 16 codes against four 16-byte stream quads
            const int2* f8 = (const int2*) (fn + ((size_t) j * H + h) * (D / 2));
            const half* fs = fn_scale + ((size_t) j * H + h) * n_groups;
            for (int c = threadIdx.x; c < D16; c += GR_THREADS_A)
            {
                int2 pk = f8[c];
                float4 q0, q1, q2, q3;
                q4_unpack8((uint32_t) pk.x, q0, q1);
                q4_unpack8((uint32_t) pk.y, q2, q3);
                float4 s0 = s4[(size_t) h * D4 + 4 * c];
                float4 s1 = s4[(size_t) h * D4 + 4 * c + 1];
                float4 s2 = s4[(size_t) h * D4 + 4 * c + 2];
                float4 s3 = s4[(size_t) h * D4 + 4 * c + 3];
                float p = 0.0f;
                p = fmaf(s0.x, q0.x, p); p = fmaf(s0.y, q0.y, p);
                p = fmaf(s0.z, q0.z, p); p = fmaf(s0.w, q0.w, p);
                p = fmaf(s1.x, q1.x, p); p = fmaf(s1.y, q1.y, p);
                p = fmaf(s1.z, q1.z, p); p = fmaf(s1.w, q1.w, p);
                p = fmaf(s2.x, q2.x, p); p = fmaf(s2.y, q2.y, p);
                p = fmaf(s2.z, q2.z, p); p = fmaf(s2.w, q2.w, p);
                p = fmaf(s3.x, q3.x, p); p = fmaf(s3.y, q3.y, p);
                p = fmaf(s3.z, q3.z, p); p = fmaf(s3.w, q3.w, p);
                a = fmaf(p, __half2float(fs[c >> g_shift]), a);
            }
        }
        else
        {
            for (int c = threadIdx.x; c < D4; c += GR_THREADS_A)
            {
                float4 s = s4[(size_t) h * D4 + c];
                a = fmaf(s.x, s.x, fmaf(s.y, s.y, fmaf(s.z, s.z, fmaf(s.w, s.w, a))));
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            a += __shfl_down_sync(0xffffffffu, a, offset);
        if (lane == 0) red[h][warp] = a;
    }
    __syncthreads();
    if (threadIdx.x < H)
    {
        float v = 0.0f;
        #pragma unroll
        for (int w = 0; w < GR_THREADS_A / 32; ++w)
            v += red[threadIdx.x][w];
        dots[((size_t) r * (M + 1) + j) * H + threadIdx.x] = v;
    }
}

template <int H, bool HALF_OUT>
__global__ __launch_bounds__(NUM_THREADS)
void gr_finalize_q4_kernel
(
    const float* __restrict__ streams,   // (R, H, D)
    const float* __restrict__ dots,      // (R, M + 1, H) from gr_dots_q4
    const uint8_t* __restrict__ upt,     // (H, D / 4, LR / 2, 4) packed int4
    const half* __restrict__ upt_scale,  // (LR / Q4_UP_GROUP, H * D)
    const half* __restrict__ w,          // (H * D) half norm weight (incl +1)
    float* __restrict__ post,            // (R, H) or nullptr (final-mixer form)
    void* __restrict__ mixed,            // (R, D) half or float
    const int D,
    const int LR,
    const int chunk_cols,
    const float rms_eps
)
{
    const int r = blockIdx.y;
    const int M = LR + (post ? H : 0);
    const float* dr = dots + (size_t) r * (M + 1) * H;

    __shared__ float rmr_s[H];
    extern __shared__ float t_s[];
    if (threadIdx.x < H)
        rmr_s[threadIdx.x] = rsqrtf(dr[(size_t) M * H + threadIdx.x] / (float) D + rms_eps);
    __syncthreads();
    const float inv_h = 1.0f / (float) H;
    for (int i = threadIdx.x; i < LR; i += NUM_THREADS)
    {
        float v = 0.0f;
        #pragma unroll
        for (int h = 0; h < H; ++h)
            v = fmaf(rmr_s[h], dr[(size_t) i * H + h], v);
        v *= inv_h;
        t_s[i] = v * sigmoidf_(v);
    }
    if (post && blockIdx.x == 0 && threadIdx.x < H)
    {
        float v = 0.0f;
        #pragma unroll
        for (int h = 0; h < H; ++h)
            v = fmaf(rmr_s[h], dr[(size_t) (LR + threadIdx.x) * H + h], v);
        post[(size_t) r * H + threadIdx.x] = 2.0f * sigmoidf_(v * inv_h);
    }
    __syncthreads();

    const int c0 = blockIdx.x * chunk_cols;
    const int c1 = min(c0 + chunk_cols, D);
    const int D4 = D / 4;
    const int LR2 = LR / 2;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const float4* s4 = (const float4*) (streams + (size_t) r * H * D);
    for (int c = c0 / 4 + warp; c < c1 / 4; c += NUM_THREADS / 32)
    {
        float4 g[H];
        #pragma unroll
        for (int h = 0; h < H; ++h) g[h] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        // Q4_UP_GROUP = 2 * 32, so one loop step is exactly one scale group
        for (int p = lane, grp = 0; p < LR2; p += 32, ++grp)
        {
            float t0 = t_s[2 * p];
            float t1 = t_s[2 * p + 1];
            #pragma unroll
            for (int h = 0; h < H; ++h)
            {
                uint32_t u = *(const uint32_t*)
                    (upt + ((((size_t) h * (D / 4) + c) * LR2 + p) * 4));
                half4 us = *(const half4*)
                    (upt_scale + (size_t) grp * H * D + (size_t) h * D + 4 * c);
                float4 lo, hi;
                q4_unpack8(u, lo, hi);
                g[h].x = fmaf(fmaf(t0, lo.x, t1 * lo.y), LOW_TO_FLOAT(us.x),  g[h].x);
                g[h].y = fmaf(fmaf(t0, lo.z, t1 * lo.w), HIGH_TO_FLOAT(us.x), g[h].y);
                g[h].z = fmaf(fmaf(t0, hi.x, t1 * hi.y), LOW_TO_FLOAT(us.y),  g[h].z);
                g[h].w = fmaf(fmaf(t0, hi.z, t1 * hi.w), HIGH_TO_FLOAT(us.y), g[h].w);
            }
        }
        #pragma unroll
        for (int h = 0; h < H; ++h)
            for (int offset = 16; offset > 0; offset >>= 1)
            {
                g[h].x += __shfl_xor_sync(0xffffffffu, g[h].x, offset);
                g[h].y += __shfl_xor_sync(0xffffffffu, g[h].y, offset);
                g[h].z += __shfl_xor_sync(0xffffffffu, g[h].z, offset);
                g[h].w += __shfl_xor_sync(0xffffffffu, g[h].w, offset);
            }
        if (lane != 0) continue;
        float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            float4 s = s4[(size_t) h * D4 + c];
            half4 wq = *(const half4*) (w + (size_t) h * D + 4 * c);
            float coef = rmr_s[h] * inv_h;
            o.x = fmaf(sigmoidf_(g[h].x) * coef * LOW_TO_FLOAT(wq.x),  s.x, o.x);
            o.y = fmaf(sigmoidf_(g[h].y) * coef * HIGH_TO_FLOAT(wq.x), s.y, o.y);
            o.z = fmaf(sigmoidf_(g[h].z) * coef * LOW_TO_FLOAT(wq.y),  s.z, o.z);
            o.w = fmaf(sigmoidf_(g[h].w) * coef * HIGH_TO_FLOAT(wq.y), s.w, o.w);
        }
        if (HALF_OUT)
        {
            half2* out2 = (half2*) ((half*) mixed + (size_t) r * D);
            out2[c * 2] = __floats2half2_rn(o.x, o.y);
            out2[c * 2 + 1] = __floats2half2_rn(o.z, o.w);
        }
        else
            ((float4*) ((float*) mixed + (size_t) r * D))[c] = o;
    }
}


// Residual update for one sublayer site: x[h, d] <- post[h] * y[d] + sum_h' comb[h', h] *
// x[h', d]. Without comb (GatedResidual): x[h, d] <- post[h] * y[d] + x[h, d]. Pure per-column
// mix of the H stream rows, so it runs in place: each thread loads all H values of its columns
// into registers before writing any back.

template <int H, typename Y_T, bool HAS_COMB>
__global__ __launch_bounds__(NUM_THREADS)
void hc_apply_kernel
(
    float* __restrict__ x,               // (R, H, D), updated in place
    const Y_T* __restrict__ y,           // (R, D) float or half
    const float* __restrict__ post,      // (R, H)
    const float* __restrict__ comb,      // (R, H, H), or null
    const int D,
    const int chunk_cols                 // multiple of 4
)
{
    const int r = blockIdx.y;
    const int c0 = blockIdx.x * chunk_cols;
    const int c1 = min(c0 + chunk_cols, D);
    const int D4 = D / 4;

    float post_r[H];
    float comb_r[H][H];
    #pragma unroll
    for (int h = 0; h < H; ++h)
    {
        post_r[h] = __ldg(post + (size_t) r * H + h);
        if (HAS_COMB)
        {
            #pragma unroll
            for (int g = 0; g < H; ++g)
                comb_r[h][g] = __ldg(comb + ((size_t) r * H + h) * H + g);
        }
    }

    float4* x4 = (float4*) (x + (size_t) r * H * D);
    for (int c = c0 / 4 + threadIdx.x; c < c1 / 4; c += NUM_THREADS)
    {
        float4 xv[H];
        #pragma unroll
        for (int h = 0; h < H; ++h)
            xv[h] = x4[(size_t) h * D4 + c];

        float4 yv;
        if constexpr (std::is_same_v<Y_T, half>)
        {
            half2 y01 = ((const half2*) (y + (size_t) r * D))[c * 2];
            half2 y23 = ((const half2*) (y + (size_t) r * D))[c * 2 + 1];
            float2 lo = __half22float2(y01);
            float2 hi = __half22float2(y23);
            yv = make_float4(lo.x, lo.y, hi.x, hi.y);
        }
        else
            yv = ((const float4*) (y + (size_t) r * D))[c];

        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            float4 o;
            o.x = post_r[h] * yv.x;
            o.y = post_r[h] * yv.y;
            o.z = post_r[h] * yv.z;
            o.w = post_r[h] * yv.w;
            if (HAS_COMB)
            {
                #pragma unroll
                for (int g = 0; g < H; ++g)
                {
                    o.x = fmaf(comb_r[g][h], xv[g].x, o.x);
                    o.y = fmaf(comb_r[g][h], xv[g].y, o.y);
                    o.z = fmaf(comb_r[g][h], xv[g].z, o.z);
                    o.w = fmaf(comb_r[g][h], xv[g].w, o.w);
                }
            }
            else
            {
                o.x += xv[h].x;
                o.y += xv[h].y;
                o.z += xv[h].z;
                o.w += xv[h].w;
            }
            x4[(size_t) h * D4 + c] = o;
        }
    }
}

// Shared launch logic. mode: fn rows M = 2H + H^2 (mix) or H (head)

static void hc_mix_launch
(
    const at::Tensor& streams,
    const at::Tensor& fn,
    const at::Tensor& base,
    const at::Tensor& scale,
    float rms_eps,
    float hc_eps,
    int sinkhorn_iters,
    at::Tensor& partials,
    at::Tensor* post,
    at::Tensor* comb,
    at::Tensor& collapsed,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(streams.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(streams, kFloat);
    TORCH_CHECK(streams.is_contiguous() && fn.is_contiguous(), "hc_mix: contiguous inputs required");
    int R = streams.size(0);
    int H = streams.size(1);
    int D = streams.size(2);
    int row_len = H * D;
    int M = fn.size(0);
    bool head = M == H;
    TORCH_CHECK(H == 4, "hc_mix: H = 4 only");
    TORCH_CHECK(head || M == 2 * H + H * H, "hc_mix: fn rows must be H or 2H + H^2");
    TORCH_CHECK(fn.size(1) == row_len && D % 4 == 0, "hc_mix: dims");
    bool fn_half = fn.dtype() == at::kHalf;
    if (!fn_half) TORCH_CHECK_DTYPE(fn, kFloat);
    TORCH_CHECK_DTYPE(base, kFloat);
    TORCH_CHECK_DTYPE(scale, kFloat);

    int n_chunks_a = partials.size(1);
    int chunk_cols = ((row_len / n_chunks_a + 4 * NUM_THREADS_A - 1) / (4 * NUM_THREADS_A)) * (4 * NUM_THREADS_A);
    n_chunks_a = (row_len + chunk_cols - 1) / chunk_cols;
    TORCH_CHECK(n_chunks_a <= partials.size(1), "hc_mix: partials workspace too small");

    int chunks_c = std::min(32, std::max(1, 256 / R));
    int chunk_cols_c = ((D / chunks_c + 4 * NUM_THREADS - 1) / (4 * NUM_THREADS)) * (4 * NUM_THREADS);
    int n_chunks_c = (D + chunk_cols_c - 1) / chunk_cols_c;

    bool half_out = collapsed.dtype() == at::kHalf;

    dim3 grid_a(n_chunks_a, R);
    dim3 grid_c(n_chunks_c, R);
    #define ARGS_A(FN_T) \
        (const float*) streams.data_ptr(), (const FN_T*) fn.data_ptr(), \
        (float*) partials.data_ptr(), row_len, chunk_cols
    #define ARGS_C(POST, COMB) \
        (const float*) streams.data_ptr(), (const float*) partials.data_ptr(), \
        (const float*) base.data_ptr(), (const float*) scale.data_ptr(), \
        POST, COMB, collapsed.data_ptr(), \
        D, n_chunks_a, chunk_cols_c, rms_eps, hc_eps, sinkhorn_iters
    if (!head)
    {
        if (fn_half)
            hc_mix_partials_kernel<4, 24, half><<<grid_a, NUM_THREADS_A, 0, stream>>>(ARGS_A(half));
        else
            hc_mix_partials_kernel<4, 24, float><<<grid_a, NUM_THREADS_A, 0, stream>>>(ARGS_A(float));
        cuda_check(cudaPeekAtLastError());
        float* post_p = (float*) post->data_ptr();
        float* comb_p = (float*) comb->data_ptr();
        if (half_out)
            hc_mix_finalize_kernel<4, 24, false, true><<<grid_c, NUM_THREADS, 0, stream>>>(ARGS_C(post_p, comb_p));
        else
            hc_mix_finalize_kernel<4, 24, false, false><<<grid_c, NUM_THREADS, 0, stream>>>(ARGS_C(post_p, comb_p));
    }
    else
    {
        if (fn_half)
            hc_mix_partials_kernel<4, 4, half><<<grid_a, NUM_THREADS_A, 0, stream>>>(ARGS_A(half));
        else
            hc_mix_partials_kernel<4, 4, float><<<grid_a, NUM_THREADS_A, 0, stream>>>(ARGS_A(float));
        cuda_check(cudaPeekAtLastError());
        if (half_out)
            hc_mix_finalize_kernel<4, 4, true, true><<<grid_c, NUM_THREADS, 0, stream>>>(ARGS_C(nullptr, nullptr));
        else
            hc_mix_finalize_kernel<4, 4, true, false><<<grid_c, NUM_THREADS, 0, stream>>>(ARGS_C(nullptr, nullptr));
    }
    #undef ARGS_A
    #undef ARGS_C
    cuda_check(cudaPeekAtLastError());
}

int hc_mix_num_chunks(int R, int row_len)
{
    int chunks_a = std::min(128, std::max(1, 512 / std::max(R, 1)));
    int chunk_cols = ((row_len / chunks_a + 4 * NUM_THREADS_A - 1) / (4 * NUM_THREADS_A)) * (4 * NUM_THREADS_A);
    return (row_len + chunk_cols - 1) / chunk_cols;
}

void hc_mix
(
    const at::Tensor& streams,           // (R, H, D) float
    const at::Tensor& fn,                // (2H + H^2, H * D) float
    const at::Tensor& base,              // (2H + H^2) float
    const at::Tensor& scale,             // (3) float
    double rms_eps,
    double hc_eps,
    int64_t sinkhorn_iters,
    at::Tensor partials,                 // (R, chunks, M + 1) float workspace
    at::Tensor post,                     // (R, H) float out
    at::Tensor comb,                     // (R, H, H) float out
    at::Tensor collapsed                 // (R, D) float or half out
)
{
    hc_mix_launch
    (
        streams,
        fn,
        base,
        scale,
        (float) rms_eps,
        (float) hc_eps,
        (int) sinkhorn_iters,
        partials,
        &post,
        &comb,
        collapsed,
        nullptr
    );
}

void hc_head
(
    const at::Tensor& streams,           // (R, H, D) float
    const at::Tensor& fn,                // (H, H * D) float
    const at::Tensor& base,              // (H) float
    const at::Tensor& scale,             // (1) float
    double rms_eps,
    double hc_eps,
    at::Tensor partials,                 // (R, chunks, H + 1) float workspace
    at::Tensor collapsed                 // (R, D) float or half out
)
{
    hc_mix_launch
    (
        streams,
        fn,
        base,
        scale,
        (float) rms_eps,
        (float) hc_eps,
        0,
        partials,
        nullptr,
        nullptr,
        collapsed,
        nullptr
    );
}

void hc_apply
(
    at::Tensor x,                        // (R, H, D) float, updated IN PLACE
    const at::Tensor& y,                 // (R, D) float or half
    const at::Tensor& post,              // (R, H) float
    const c10::optional<at::Tensor>& comb   // (R, H, H) float, or none (x[h] += post[h] * y)
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(x, kFloat);
    TORCH_CHECK_DTYPE(post, kFloat);
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous() && post.is_contiguous(), "hc_apply: contiguous inputs required");
    int R = x.size(0);
    int H = x.size(1);
    int D = x.size(2);
    TORCH_CHECK(H == 4, "hc_apply: H = 4 only");
    TORCH_CHECK(D % 4 == 0, "hc_apply: dims");
    TORCH_CHECK(y.size(0) == R && y.size(-1) == D, "hc_apply: y shape");
    TORCH_CHECK(post.size(0) == R, "hc_apply: gate shapes");
    const float* comb_p = nullptr;
    if (comb)
    {
        TORCH_CHECK_DTYPE(comb.value(), kFloat);
        TORCH_CHECK(comb.value().is_contiguous() && comb.value().size(0) == R, "hc_apply: comb shape");
        comb_p = (const float*) comb.value().data_ptr();
    }

    int chunks_c = std::min(32, std::max(1, 256 / R));
    int chunk_cols = ((D / chunks_c + 4 * NUM_THREADS - 1)
                      / (4 * NUM_THREADS)) * (4 * NUM_THREADS);
    int n_chunks = (D + chunk_cols - 1) / chunk_cols;

    dim3 grid(n_chunks, R);
    #define ARGS(Y_T) \
        (float*) x.data_ptr(), (const Y_T*) y.data_ptr(), \
        (const float*) post.data_ptr(), comb_p, D, chunk_cols
    if (y.dtype() == at::kHalf)
    {
        if (comb_p) hc_apply_kernel<4, half, true><<<grid, NUM_THREADS, 0, stream>>>(ARGS(half));
        else        hc_apply_kernel<4, half, false><<<grid, NUM_THREADS, 0, stream>>>(ARGS(half));
    }
    else
    {
        if (comb_p) hc_apply_kernel<4, float, true><<<grid, NUM_THREADS, 0, stream>>>(ARGS(float));
        else        hc_apply_kernel<4, float, false><<<grid, NUM_THREADS, 0, stream>>>(ARGS(float));
    }
    #undef ARGS
    cuda_check(cudaPeekAtLastError());
}

// Phase C is warp-per-column-quad: chunk at warp granularity (4 * NUM_THREADS / 32 columns)
// so small R still fills the device
static void gr_finalize_grid(int D, int R, int& chunk_cols, int& n_chunks)
{
    const int gran = 4 * (NUM_THREADS / 32);
    int chunks_c = std::max(1, std::min((D + gran - 1) / gran, 512 / R));
    chunk_cols = ((D / chunks_c + gran - 1) / gran) * gran;
    n_chunks = (D + chunk_cols - 1) / chunk_cols;
}

void gr_mix
(
    const at::Tensor& streams,           // (R, H, D) float
    const at::Tensor& fn,                // (M, H * D) half: cat(down, inject) * w
    const at::Tensor& upt,               // (H, D / 4, LR, 4) half: up repacked lane-contiguous
    const at::Tensor& w,                 // (H * D) half norm weight (incl +1)
    double rms_eps,
    at::Tensor dots,                     // (R, M + 1, H) float workspace
    c10::optional<at::Tensor> post,      // (R, H) float out, or none (final-mixer form)
    at::Tensor mixed                     // (R, D) half or float out
)
{
    const at::cuda::OptionalCUDAGuard device_guard(streams.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(streams, kFloat);
    TORCH_CHECK_DTYPE(fn, kHalf);
    TORCH_CHECK_DTYPE(upt, kHalf);
    TORCH_CHECK_DTYPE(w, kHalf);
    TORCH_CHECK_DTYPE(dots, kFloat);
    TORCH_CHECK(streams.is_contiguous() && fn.is_contiguous() && upt.is_contiguous() &&
                w.is_contiguous() && dots.is_contiguous(), "gr_mix: contiguous inputs required");
    int R = streams.size(0);
    int H = streams.size(1);
    int D = streams.size(2);
    int M = fn.size(0);
    TORCH_CHECK(upt.dim() == 4 && upt.size(0) == H && upt.size(1) == D / 4 && upt.size(3) == 4,
                "gr_mix: upt must be the (H, D / 4, LR, 4) repacked layout");
    int LR = upt.size(2);
    TORCH_CHECK(H == 4, "gr_mix: H = 4 only");
    TORCH_CHECK(D % 8 == 0, "gr_mix: dims");
    TORCH_CHECK(M == LR + (post ? H : 0), "gr_mix: fn rows must be LR (+ H with post)");
    TORCH_CHECK(fn.size(1) == H * D && w.numel() == H * D, "gr_mix: dims");
    TORCH_CHECK(dots.size(0) == R && dots.size(1) == M + 1 && dots.size(2) == H, "gr_mix: dots shape");

    dim3 grid_a(M + 1, R);
    gr_dots_kernel<4><<<grid_a, GR_THREADS_A, 0, stream>>>
    (
        (const float*) streams.data_ptr(), (const half*) fn.data_ptr(),
        (float*) dots.data_ptr(), M, D
    );
    cuda_check(cudaPeekAtLastError());

    int chunk_cols, n_chunks;
    gr_finalize_grid(D, R, chunk_cols, n_chunks);
    dim3 grid_c(n_chunks, R);
    int smem = LR * sizeof(float);
    float* post_p = post ? (float*) post.value().data_ptr() : nullptr;
    #define ARGS \
        (const float*) streams.data_ptr(), (const float*) dots.data_ptr(), \
        (const half*) upt.data_ptr(), (const half*) w.data_ptr(), \
        post_p, mixed.data_ptr(), D, LR, chunk_cols, (float) rms_eps
    if (mixed.dtype() == at::kHalf)
        gr_finalize_kernel<4, true><<<grid_c, NUM_THREADS, smem, stream>>>(ARGS);
    else
        gr_finalize_kernel<4, false><<<grid_c, NUM_THREADS, smem, stream>>>(ARGS);
    #undef ARGS
    cuda_check(cudaPeekAtLastError());
}

/*
One entry point for every quantized width. The two kernels are independent -- gr_dots reads
fn, gr_finalize reads upt -- so the width of each side is chosen by that tensor's dtype
(kChar = int8 with one scale per output slice, kByte = packed int4 with group scales along
the reduction axis). That gives fn and upt independent widths for free, which is what
EXL3_HC_QUANT=int4fn / int4up are.
*/

void gr_mix_q
(
    const at::Tensor& streams,           // (R, H, D) float
    const at::Tensor& fn,                // (M, H * D) int8 | (M, H * D / 2) uint8
    const at::Tensor& fn_scale,          // (M, H) | (M, H, D / GF) half
    const at::Tensor& upt,               // (H, D/4, LR, 4) int8 | (H, D/4, LR/2, 4) uint8
    const at::Tensor& upt_scale,         // (H * D) | (LR / 64, H * D) half
    const at::Tensor& w,                 // (H * D) half norm weight (incl +1)
    double rms_eps,
    at::Tensor dots,                     // (R, M + 1, H) float workspace
    c10::optional<at::Tensor> post,      // (R, H) float out, or none (final-mixer form)
    at::Tensor mixed                     // (R, D) half or float out
)
{
    const at::cuda::OptionalCUDAGuard device_guard(streams.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(streams, kFloat);
    TORCH_CHECK_DTYPE(fn_scale, kHalf);
    TORCH_CHECK_DTYPE(upt_scale, kHalf);
    TORCH_CHECK_DTYPE(w, kHalf);
    TORCH_CHECK_DTYPE(dots, kFloat);
    TORCH_CHECK(fn.dtype() == at::kChar || fn.dtype() == at::kByte,
                "gr_mix_q: fn must be int8 (kChar) or packed int4 (kByte)");
    TORCH_CHECK(upt.dtype() == at::kChar || upt.dtype() == at::kByte,
                "gr_mix_q: upt must be int8 (kChar) or packed int4 (kByte)");
    TORCH_CHECK(streams.is_contiguous() && fn.is_contiguous() && fn_scale.is_contiguous() &&
                upt.is_contiguous() && upt_scale.is_contiguous() && w.is_contiguous() &&
                dots.is_contiguous(), "gr_mix_q: contiguous inputs required");

    const bool fn4 = fn.dtype() == at::kByte;
    const bool up4 = upt.dtype() == at::kByte;
    int R = streams.size(0);
    int H = streams.size(1);
    int D = streams.size(2);
    int M = fn.size(0);
    TORCH_CHECK(upt.dim() == 4 && upt.size(0) == H && upt.size(1) == D / 4 && upt.size(3) == 4,
                "gr_mix_q: upt must be the (H, D / 4, LR [/ 2], 4) repacked layout");
    int LR = upt.size(2) * (up4 ? 2 : 1);
    TORCH_CHECK(H == 4, "gr_mix_q: H = 4 only");
    TORCH_CHECK(D % 8 == 0 && (!fn4 || D % 16 == 0), "gr_mix_q: dims");
    TORCH_CHECK(!up4 || LR % Q4_UP_GROUP == 0, "gr_mix_q: int4 upt needs LR % 64 == 0");
    TORCH_CHECK(M == LR + (post ? H : 0), "gr_mix_q: fn rows must be LR (+ H with post)");
    TORCH_CHECK(fn.size(1) == (fn4 ? H * D / 2 : H * D) && w.numel() == H * D, "gr_mix_q: dims");
    TORCH_CHECK(dots.size(0) == R && dots.size(1) == M + 1 && dots.size(2) == H,
                "gr_mix_q: dots shape");
    TORCH_CHECK(upt_scale.numel() == (int64_t) (up4 ? LR / Q4_UP_GROUP : 1) * H * D,
                "gr_mix_q: upt_scale shape");

    // The R-tiled int8 pair reads the weight once per GR_R_TILE stream rows; int4 has no
    // tiled form, so the two sides pick independently (fn and upt can be different widths)
    const bool rtile = R >= GR_RTILE_MIN_R;
    const int r_tile = gr_r_tile_env();
    const int j_tile = gr_j_tile_env();
    const int n_rtiles = (R + r_tile - 1) / r_tile;
    const bool rtile_fn = rtile && !fn4;
    const bool rtile_up = rtile && !up4;

    const int j_t = rtile_fn ? j_tile : 1;
    dim3 grid_a((M + 1 + j_t - 1) / j_t, rtile_fn ? n_rtiles : R);
    if (fn4)
    {
        TORCH_CHECK(fn_scale.numel() % ((int64_t) M * H) == 0, "gr_mix_q: fn_scale shape");
        int n_groups = fn_scale.numel() / ((int64_t) M * H);
        TORCH_CHECK(n_groups > 0 && D % (16 * n_groups) == 0, "gr_mix_q: fn group size");
        int g_step = D / (16 * n_groups);
        TORCH_CHECK((g_step & (g_step - 1)) == 0, "gr_mix_q: fn group must be 16 << k");
        int g_shift = 0;
        while ((1 << g_shift) < g_step) ++g_shift;
        gr_dots_q4_kernel<4><<<grid_a, GR_THREADS_A, 0, stream>>>
        (
            (const float*) streams.data_ptr(), (const uint8_t*) fn.data_ptr(),
            (const half*) fn_scale.data_ptr(), (float*) dots.data_ptr(), M, D,
            n_groups, g_shift
        );
    }
    else
    {
        TORCH_CHECK(fn_scale.numel() == (int64_t) M * H, "gr_mix_q: fn_scale shape");
        #define ARGS_A \
            (const float*) streams.data_ptr(), (const int8_t*) fn.data_ptr(), \
            (const half*) fn_scale.data_ptr(), (float*) dots.data_ptr(), M, D
        #define LAUNCH_DOTS_J(RT)                                                             \
            switch (j_t)                                                                      \
            {                                                                                 \
                case 8: gr_dots_q8_rtj_kernel<4, RT, 8>                                       \
                            <<<grid_a, GR_THREADS_A, 0, stream>>>(ARGS_A, R); break;           \
                case 4: gr_dots_q8_rtj_kernel<4, RT, 4>                                       \
                            <<<grid_a, GR_THREADS_A, 0, stream>>>(ARGS_A, R); break;           \
                case 2: gr_dots_q8_rtj_kernel<4, RT, 2>                                       \
                            <<<grid_a, GR_THREADS_A, 0, stream>>>(ARGS_A, R); break;           \
                default: gr_dots_q8_rt_kernel<4, RT>                                          \
                            <<<grid_a, GR_THREADS_A, 0, stream>>>(ARGS_A, R); break;           \
            }
        if (rtile_fn && r_tile == 2)
            LAUNCH_DOTS_J(2)
        else if (rtile_fn)
            LAUNCH_DOTS_J(4)
        else
            gr_dots_q8_kernel<4><<<grid_a, GR_THREADS_A, 0, stream>>>(ARGS_A);
        #undef LAUNCH_DOTS_J
        #undef ARGS_A
    }
    cuda_check(cudaPeekAtLastError());

    const int rows_c = rtile_up ? n_rtiles : R;
    int chunk_cols, n_chunks;
    gr_finalize_grid(D, rows_c, chunk_cols, n_chunks);
    dim3 grid_c(n_chunks, rows_c);
    int smem = (rtile_up ? r_tile : 1) * LR * sizeof(float);
    float* post_p = post ? (float*) post.value().data_ptr() : nullptr;
    #define ARGS(T) \
        (const float*) streams.data_ptr(), (const float*) dots.data_ptr(), \
        (const T*) upt.data_ptr(), (const half*) upt_scale.data_ptr(), \
        (const half*) w.data_ptr(), post_p, mixed.data_ptr(), D, LR, chunk_cols, (float) rms_eps
    const bool half_out = mixed.dtype() == at::kHalf;
    if (up4 && half_out)
        gr_finalize_q4_kernel<4, true><<<grid_c, NUM_THREADS, smem, stream>>>(ARGS(uint8_t));
    else if (up4)
        gr_finalize_q4_kernel<4, false><<<grid_c, NUM_THREADS, smem, stream>>>(ARGS(uint8_t));
    else if (rtile_up && half_out && r_tile == 2)
        gr_finalize_q8_rt_kernel<4, 2, true>
            <<<grid_c, NUM_THREADS, smem, stream>>>(ARGS(int8_t), R);
    else if (rtile_up && half_out)
        gr_finalize_q8_rt_kernel<4, 4, true>
            <<<grid_c, NUM_THREADS, smem, stream>>>(ARGS(int8_t), R);
    else if (rtile_up && r_tile == 2)
        gr_finalize_q8_rt_kernel<4, 2, false>
            <<<grid_c, NUM_THREADS, smem, stream>>>(ARGS(int8_t), R);
    else if (rtile_up)
        gr_finalize_q8_rt_kernel<4, 4, false>
            <<<grid_c, NUM_THREADS, smem, stream>>>(ARGS(int8_t), R);
    else if (half_out)
        gr_finalize_q8_kernel<4, true><<<grid_c, NUM_THREADS, smem, stream>>>(ARGS(int8_t));
    else
        gr_finalize_q8_kernel<4, false><<<grid_c, NUM_THREADS, smem, stream>>>(ARGS(int8_t));
    #undef ARGS
    cuda_check(cudaPeekAtLastError());
}
