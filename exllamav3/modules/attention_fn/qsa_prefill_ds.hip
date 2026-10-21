// Qwen QSA sparse prefill attention, NW waves per (query row, kv head): wave w owns output dims DW w .. DW w + DW - 1, so
// each wave keeps 16 / NW accumulator tiles instead of 16 (qsa_prefill.hip spills ~120-360 VGPRs on gfx1150 and runs at <1 TFLOP/s).
// Scores: each wave runs the WMMA chain over its own DW dims, the NW partial tiles are summed through LDS in a fixed order
// (S = S_w0 + S_w1 + ...), so both waves hold the same scores; softmax is computed redundantly in both. Otherwise the recurrence is
// qsa_prefill.hip's QG=0 KS=0 path (32-key chunks in list order, Triton's exp lowering, l = fma(l, alpha, sum(p))): the only
// numerical difference from it is the split score chain.
// MEMORY SAFETY: same IX_* / clamp macros and *_OK guards as qsa_prefill.hip.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef _Float16 half_t;
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef uint32_t v4u __attribute__((ext_vector_type(4)));

#define HD 256
#define NH 24
#define NKV 2
#define GRP 12
#define PAGE 256
#define KPAD 2080
#define BN 32
#define NCH (KPAD / BN)
#define LDV 264
#define LDQ 264
#ifndef NW
#define NW 2              // waves per (row, kv head); wave w owns HD / NW output dims
#endif
#define DW (HD / NW)      // dims per wave
#define NTW (16 / NW)     // 16-dim output tiles per wave, also V / Q rows staged per wave
#ifndef VB
#define VB NTW            // V rows per wave per staging batch
#endif
#ifndef WPE
#define WPE 1
#endif

#define IX_Q(r_, h_, d_, p_) (((int64_t) (r_)) * (p_) + (h_) * HD + (d_))
#define IX_O(r_, h_, d_, p_) (((int64_t) (r_)) * (p_) + (h_) * HD + (d_))
#define IX_IDX(r_, n_) (((int64_t) (r_)) * KPAD + (n_))
#define IX_BT(r_, pg_, p_) (((int64_t) (r_)) * (p_) + (pg_))
#define PGC(pg_, NP_) min(max((pg_), 0), (NP_) - 1)
#define TOKC(t_, NT_) min(max((t_), 0), (NT_) - 1)
#define IX_KV(t_, g_, d_, p_) (((int64_t) (t_)) * (p_) + (g_) * HD + (d_))
#define Q_OK(h_) ((h_) < GRP)
#define O_OK(h_) ((h_) < GRP)

__device__ __forceinline__ v8f wmma(v16h a, v16h b, v8f c) { return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c); }

__device__ __forceinline__ float tl_exp(float x) {
    const float L2E = 1.44269502162933349609375f;
    float t = x * L2E;
    bool lo = (-126.0f > t);
    float u = __builtin_fmaf(x, L2E, lo ? 64.0f : 0.0f);
    float e = __builtin_amdgcn_exp2f(u);
    return __builtin_amdgcn_ldexpf(e, lo ? -64 : 0);
}

__device__ __forceinline__ v16h ld16h(const half_t* p) {
    v4u a = *(const v4u*) p, b = *(const v4u*) (p + 8);
    union { v4u u[2]; v16h h; } x; x.u[0] = a; x.u[1] = b; return x.h;
}

// this wave's NTW key rows (NTW w ..) of half chunk hf_ of the V cache -> sV rows at pitch LDV (invalid keys zero)
#define STAGE_V(hf_) do { \
        for (int stg_i0 = 0; stg_i0 < NTW; stg_i0 += VB) { \
            v4u stg_vb[VB]; \
            _Pragma("unroll") for (int stg_j = 0; stg_j < VB; ++stg_j) { \
                const int stg_r = NTW * w + stg_i0 + stg_j; \
                const int stg_t = sTok[16 * (hf_) + stg_r]; \
                stg_vb[stg_j] = (v4u) {0, 0, 0, 0}; \
                if ((vmask >> (16 * (hf_) + stg_r)) & 1u) stg_vb[stg_j] = *(const v4u*) (vc + IX_KV(stg_t, g, lane * 8, tok_stride)); \
            } \
            _Pragma("unroll") for (int stg_j = 0; stg_j < VB; ++stg_j) *(v4u*) (sV + (NTW * w + stg_i0 + stg_j) * LDV + lane * 8) = stg_vb[stg_j]; \
        } } while (0)

extern "C" __global__ __launch_bounds__(32 * NW) __attribute__((amdgpu_waves_per_eu(WPE))) void qsa_pf(
    const half_t* __restrict__ q, const half_t* __restrict__ kc, const half_t* __restrict__ vc,
    const int* __restrict__ bt, const int* __restrict__ idx, half_t* __restrict__ o,
    int q_stride, int o_stride, int bt_stride, int tok_stride, int ntok, int npages, float scale)
{
    const int pid = blockIdx.x;
    const int row = pid / NKV, g = pid % NKV;
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, n = lane & 15, hh = lane >> 4;
    __shared__ half_t sQ[16 * LDQ];
    __shared__ half_t sV[16 * LDV];
    __shared__ int sTok[BN];
    __shared__ float sS[NW][16][32];

    for (int i = NTW * w; i < NTW * w + NTW; ++i) {  // q rows of this kv head -> LDS (rows >= GRP zero), NTW per wave
        v4u val = {0, 0, 0, 0};
        if (Q_OK(i)) val = *(const v4u*) (q + IX_Q(row, g * GRP + i, lane * 8, q_stride));
        *(v4u*) (sQ + i * LDQ + lane * 8) = val;
    }
    __syncthreads();

    v8f acc[NTW];
#pragma unroll
    for (int i = 0; i < NTW; ++i) acc[i] = (v8f) {0, 0, 0, 0, 0, 0, 0, 0};
    float m = -__builtin_inff(), l = 0.0f;
    const bool head_ok = n < GRP;

    int id = idx[IX_IDX(row, lane)];
    int ph = bt[IX_BT(row, PGC(max(id, 0) / PAGE, npages), bt_stride)];
    for (int c = 0; c < NCH; ++c) {
        const int id_n = idx[IX_IDX(row, min(c + 1, NCH - 1) * BN + lane)];
        const bool kv_valid = id >= 0;
        const unsigned vmask = (unsigned) __ballot(kv_valid);   // same in both waves (same row)
        const int idc = max(id, 0);
        if (vmask == 0u) {
            id = id_n; ph = bt[IX_BT(row, PGC(max(id_n, 0) / PAGE, npages), bt_stride)];
            continue;
        }
        __syncthreads();                    // previous chunk done with sTok / sV / sS
        const int tok = kv_valid ? TOKC(ph * PAGE + idc % PAGE, ntok) : 0;
        if (w == 0) sTok[lane] = tok;
        __syncthreads();

        // partial scores over this wave's DW dims
        const int tk0 = sTok[n], tk1 = sTok[16 + n];
        const half_t* kp0 = kc + IX_KV(tk0, g, DW * w, tok_stride);
        const half_t* kp1 = kc + IX_KV(tk1, g, DW * w, tok_stride);
        const half_t* qp = sQ + n * LDQ + DW * w;
        v8f S[2] = {(v8f) {0, 0, 0, 0, 0, 0, 0, 0}, (v8f) {0, 0, 0, 0, 0, 0, 0, 0}};
#pragma unroll
        for (int ks = 0; ks < DW / 16; ++ks) {
            const v16h B = ld16h(qp + 16 * ks);
            S[0] = wmma(ld16h(kp0 + 16 * ks), B, S[0]);
            S[1] = wmma(ld16h(kp1 + 16 * ks), B, S[1]);
        }
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int r = 0; r < 8; ++r) sS[w][8 * t + r][lane] = S[t][r];
        STAGE_V(0);
        __syncthreads();                    // partial scores and V half 0 complete
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                float a = sS[0][8 * t + r][lane];
#pragma unroll
                for (int ww = 1; ww < NW; ++ww) a = a + sS[ww][8 * t + r][lane];
                S[t][r] = a;
            }

        float s[2][8], mx = -__builtin_inff();
        bool ok[2][8];
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int j = 16 * t + 2 * r + hh;
                ok[t][r] = head_ok && ((vmask >> j) & 1u);
                s[t][r] = ok[t][r] ? S[t][r] * scale : -__builtin_inff();
                mx = fmaxf(mx, s[t][r]);
            }
        mx = fmaxf(mx, __shfl_xor(mx, 16));
        const float m_new = fmaxf(m, mx);
        const float m_exp = (m_new == -__builtin_inff()) ? 0.0f : m_new;
        float p[2][8];
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const float e = tl_exp(s[t][r] - m_exp);
                p[t][r] = ok[t][r] ? e : 0.0f;
            }
        const float alpha = (m == -__builtin_inff()) ? 0.0f : tl_exp(m - m_exp);
        float sum = p[0][0] + p[0][1];
#pragma unroll
        for (int r = 2; r < 8; ++r) sum = sum + p[0][r];
#pragma unroll
        for (int r = 0; r < 8; ++r) sum = sum + p[1][r];
        sum = sum + __shfl_xor(sum, 16);
        l = __builtin_fmaf(l, alpha, sum);
        m = m_new;

        v16h pf[2];
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const half_t own = (half_t) p[t][r];
                union { half_t h; uint16_t u; } cv; cv.h = own;
                const uint16_t other = (uint16_t) __shfl_xor((int) cv.u, 16);
                union { uint16_t u; half_t h; } oc; oc.u = other;
                pf[t][2 * r] = hh == 0 ? own : oc.h;
                pf[t][2 * r + 1] = hh == 0 ? oc.h : own;
            }

#pragma unroll
        for (int nt = 0; nt < NTW; ++nt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[nt][i] = acc[nt][i] * alpha;
            v16h A;
#pragma unroll
            for (int r = 0; r < 16; ++r) A[r] = sV[r * LDV + 16 * (NTW * w + nt) + n];
            acc[nt] = wmma(A, pf[0], acc[nt]);
        }
        __syncthreads();                    // both waves done with V half 0
        STAGE_V(1);
        __syncthreads();
#pragma unroll
        for (int nt = 0; nt < NTW; ++nt) {
            v16h A;
#pragma unroll
            for (int r = 0; r < 16; ++r) A[r] = sV[r * LDV + 16 * (NTW * w + nt) + n];
            acc[nt] = wmma(A, pf[1], acc[nt]);
        }
        id = id_n; ph = bt[IX_BT(row, PGC(max(id_n, 0) / PAGE, npages), bt_stride)];
    }

    const float m_safe = (m == -__builtin_inff()) ? 0.0f : m;
    const float wt = (m == -__builtin_inff()) ? 0.0f : tl_exp(m - m_safe);
    const float lw = l * wt;
    const float l_sum = __builtin_fmaf(l, wt, 0.0f);
    const float den = (lw != 0.0f) ? l_sum : 1.0f;
    __syncthreads();                        // sV free
    half_t* sO = sV;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const float a = __builtin_fmaf(acc[nt][r], wt, 0.0f);
            sO[n * LDV + 16 * (NTW * w + nt) + 2 * r + hh] = (half_t) (a / den);
        }
    __syncthreads();
    for (int h = (GRP / NW) * w; h < (GRP / NW) * (w + 1); ++h)
        if (O_OK(h)) *(v4u*) (o + IX_O(row, g * GRP + h, lane * 8, o_stride)) = *(const v4u*) (sO + h * LDV + lane * 8);
}
