// Production-style fused SwiGLU for the interleaved [tokens, 2*H] serving layout.
//
// Semantics match the shipped two-kernel path exactly, including its rounding:
//   a = x[..., :H]                       (gate, bf16)
//   s = float(a) / (1 + exp(-float(a)))  (fp32, like ATen silu)
//   s = bf16(s)                          (the intermediate round trip)
//   o = bf16(float(s) * float(u))        (bf16 multiply, as ATen does)
// The bf16 rounding of s is what makes the output bit-identical; dropping it
// matches only ~72% of elements.

#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int VEC = 8;  // 8 bf16 = 16 B per load; H is a multiple of 8 here

__device__ __forceinline__ __nv_bfloat16 silu_rounded(__nv_bfloat16 a) {
    float f = __bfloat162float(a);
    float s = f / (1.0f + expf(-f));
    return __float2bfloat16(s);
}

__global__ void silu_and_mul_kernel(const __nv_bfloat16* __restrict__ x,
                                    __nv_bfloat16* __restrict__ out,
                                    long long n, int H) {
    long long base = (long long)(blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    if (base >= n) return;
    long long row = base / H;          // H is a multiple of VEC, so a vector never straddles rows
    int col = (int)(base - row * H);
    const __nv_bfloat16* g = x + (row * 2LL * H + col);
    const __nv_bfloat16* u = g + H;
    __nv_bfloat16* o = out + (row * H + col);

    uint4 gv = *reinterpret_cast<const uint4*>(g);
    uint4 uv = *reinterpret_cast<const uint4*>(u);
    __nv_bfloat16* gs = reinterpret_cast<__nv_bfloat16*>(&gv);
    __nv_bfloat16* us = reinterpret_cast<__nv_bfloat16*>(&uv);
    uint4 ov;
    __nv_bfloat16* os = reinterpret_cast<__nv_bfloat16*>(&ov);
#pragma unroll
    for (int k = 0; k < VEC; ++k) {
        float s = __bfloat162float(silu_rounded(gs[k]));
        float m = s * __bfloat162float(us[k]);
        os[k] = __float2bfloat16(m);
    }
    *reinterpret_cast<uint4*>(o) = ov;
}

}  // namespace

torch::Tensor silu_and_mul_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "input must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "input must be bf16");
    TORCH_CHECK(x.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(x.size(-1) % 2 == 0, "last dim must be even");
    int H = (int)(x.size(-1) / 2);
    TORCH_CHECK(H % VEC == 0, "half width must be a multiple of 8");

    auto x2 = x.view({-1, 2LL * H});
    long long rows = x2.size(0);
    long long n = rows * H;
    auto out = torch::empty({rows, H}, x.options());
    if (n > 0) {
        int threads = 256;
        long long blocks = (n / VEC + threads - 1) / threads;
        silu_and_mul_kernel<<<(unsigned)blocks, threads>>>(
            reinterpret_cast<const __nv_bfloat16*>(x2.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), n, H);
    }
    std::vector<int64_t> shape(x.sizes().begin(), x.sizes().end());
    shape.back() = H;
    return out.view(shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("silu_and_mul", &silu_and_mul_cuda, "fused SwiGLU over interleaved [.., 2H] (bf16)");
}
