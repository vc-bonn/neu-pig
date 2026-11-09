#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>
#include <torch/extension.h>

namespace {
void check(const torch::Tensor &t, const torch::Tensor &anchor, torch::ScalarType dtype) {
    TORCH_CHECK(t.is_cuda() && t.device() == anchor.device() && t.scalar_type() == dtype &&
                    t.is_contiguous(),
                "Expected contiguous tensors on the same CUDA device");
    TORCH_CHECK(t.numel() <= INT_MAX, "Tensor exceeds kernel index range");
}
__device__ void cross(const float *a, const float *b, float *c) {
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}
__global__ void transform_forward(const float *p, const float *vertices, float *out, int count,
                                  int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    p += i * 7;
    const float *v = vertices + (i % count) * 3;
    float w = p[0] + 1.f, a[3] = {p[1], p[2], p[3]}, b[3], c[3];
    float normalizer = sqrtf(w * w + a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) + 1e-8f;
    w /= normalizer;
    for (int k = 0; k < 3; ++k)
        a[k] /= normalizer;
    cross(a, v, b);
    cross(a, b, c);
    float scale = 2.f / (w * w + a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    for (int k = 0; k < 3; ++k)
        out[i * 3 + k] = v[k] + scale * (w * b[k] + c[k]) + tanhf(p[4 + k] * .1f);
}
__global__ void transform_backward(const float *p, const float *vertices, const float *grad,
                                   float *dp, float *dv, int count, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    p += i * 7;
    const float *v = vertices + (i % count) * 3;
    const float *g = grad + i * 3;
    float w = p[0] + 1.f, a[3] = {p[1], p[2], p[3]}, b[3], c[3], vxg[3];
    float raw[4] = {w, a[0], a[1], a[2]};
    float norm = sqrtf(w * w + a[0] * a[0] + a[1] * a[1] + a[2] * a[2]), normalizer = norm + 1e-8f;
    w /= normalizer;
    for (int k = 0; k < 3; ++k)
        a[k] /= normalizer;
    cross(a, v, b);
    cross(a, b, c);
    cross(v, g, vxg);
    float s = w * w + a[0] * a[0] + a[1] * a[1] + a[2] * a[2], scale = 2.f / s;
    float gc = 0, gb = 0, av = 0, ga = 0, gv = 0;
    for (int k = 0; k < 3; ++k) {
        gc += g[k] * (w * b[k] + c[k]);
        gb += g[k] * b[k];
        av += a[k] * v[k];
        ga += g[k] * a[k];
        gv += g[k] * v[k];
    }
    dp[i * 7] = scale * gb - 2.f * scale * w * gc / s;
    for (int k = 0; k < 3; ++k) {
        dp[i * 7 + 1 + k] = scale * (w * vxg[k] + av * g[k] + ga * v[k] - 2.f * gv * a[k]) -
                            2.f * scale * gc * a[k] / s;
        float t = tanhf(p[4 + k] * .1f);
        dp[i * 7 + 4 + k] = g[k] * .1f * (1.f - t * t);
    }
    float dot = 0;
    for (int k = 0; k < 4; ++k)
        dot += dp[i * 7 + k] * raw[k];
    for (int k = 0; k < 4; ++k)
        dp[i * 7 + k] = dp[i * 7 + k] / normalizer -
                        (norm > 0 ? raw[k] * dot / (norm * normalizer * normalizer) : 0.f);
    if (dv) {
        float axg[3], axaxg[3];
        cross(a, g, axg);
        cross(a, axg, axaxg);
        for (int k = 0; k < 3; ++k)
            atomicAdd(dv + (i % count) * 3 + k, g[k] + scale * (-w * axg[k] + axaxg[k]));
    }
}
__global__ void edge_forward(const float *v, const int64_t *edges, const float *base, float *errors,
                             int vertices, int count, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    int e = i % count, offset = (i / count) * vertices * 3;
    int a = edges[e * 2], b = edges[e * 2 + 1];
    float s = 0;
    for (int k = 0; k < 3; ++k) {
        float d = v[offset + a * 3 + k] - v[offset + b * 3 + k];
        s += d * d;
    }
    errors[i] = fabsf(sqrtf(s) - base[e]);
}
__global__ void edge_backward(const float *v, const int64_t *edges, const float *base,
                              const float *grad, float *out, int vertices, int count, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    int e = i % count, offset = (i / count) * vertices * 3;
    int a = edges[e * 2], b = edges[e * 2 + 1];
    float d[3], s = 0;
    for (int k = 0; k < 3; ++k) {
        d[k] = v[offset + a * 3 + k] - v[offset + b * 3 + k];
        s += d[k] * d[k];
    }
    float length = sqrtf(s), delta = length - base[e];
    float scale = length > 0 ? ((delta > 0) - (delta < 0)) * (*grad) / (total * length) : 0.f;
    for (int k = 0; k < 3; ++k) {
        float g = scale * d[k];
        atomicAdd(out + offset + a * 3 + k, g);
        atomicAdd(out + offset + b * 3 + k, -g);
    }
}
__global__ void sample_forward(const float *grid, const int64_t *nodes, const float *weights,
                               float *out, int channels, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    int point = i / channels, c = i % channels;
    float value = 0;
    for (int k = 0; k < 8; ++k)
        value += weights[point * 8 + k] * grid[nodes[point * 8 + k] * channels + c];
    out[i] = value;
}
__global__ void sample_backward(const float *grad, const int64_t *nodes, const float *weights,
                                float *out, int channels, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    int point = i / channels, c = i % channels;
    for (int k = 0; k < 8; ++k)
        atomicAdd(out + nodes[point * 8 + k] * channels + c, weights[point * 8 + k] * grad[i]);
}
void check_geometry(const torch::Tensor &p, const torch::Tensor &v) {
    check(p, p, torch::kFloat32);
    check(v, p, torch::kFloat32);
    TORCH_CHECK(p.dim() == 3 && p.size(2) == 7 && v.dim() == 2 && v.size(1) == 3 &&
                    p.size(1) == v.size(0) && p.size(0) > 0 && v.size(0) > 0,
                "Expected TxNx7 parameters and Nx3 vertices");
}
void check_edges(const torch::Tensor &v, const torch::Tensor &edges, const torch::Tensor &base) {
    check(v, v, torch::kFloat32);
    check(edges, v, torch::kInt64);
    check(base, v, torch::kFloat32);
    TORCH_CHECK(v.dim() == 3 && v.size(2) == 3 && v.size(0) > 0 && v.size(1) > 0 &&
                    edges.dim() == 2 && edges.size(1) == 2 && edges.size(0) > 0 &&
                    base.numel() == edges.size(0),
                "Invalid edge-loss shapes");
    TORCH_CHECK(v.size(0) * edges.size(0) <= INT_MAX, "Too many edges");
}
void check_sampling(const torch::Tensor &grid, const torch::Tensor &nodes,
                    const torch::Tensor &weights) {
    check(grid, grid, torch::kFloat32);
    check(nodes, grid, torch::kInt64);
    check(weights, grid, torch::kFloat32);
    TORCH_CHECK(grid.dim() == 2 && grid.size(1) > 0 && nodes.dim() == 2 && nodes.size(1) == 8 &&
                    weights.sizes() == nodes.sizes(),
                "Invalid fixed sampling shapes");
    TORCH_CHECK(nodes.size(0) * grid.size(1) <= INT_MAX, "Too many sampled features");
}
} // namespace

torch::Tensor transform(torch::Tensor p, torch::Tensor v) {
    check_geometry(p, v);
    c10::cuda::CUDAGuard guard(p.device());
    auto out = torch::empty({p.size(0), p.size(1), 3}, p.options());
    int n = p.size(0) * p.size(1);
    transform_forward<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        p.data_ptr<float>(), v.data_ptr<float>(), out.data_ptr<float>(), v.size(0), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
std::vector<torch::Tensor> transform_grad(torch::Tensor p, torch::Tensor v, torch::Tensor grad,
                                          bool need_v) {
    check_geometry(p, v);
    check(grad, p, torch::kFloat32);
    TORCH_CHECK(grad.dim() == 3 && grad.size(0) == p.size(0) && grad.size(1) == p.size(1) &&
                    grad.size(2) == 3,
                "Invalid transform gradient");
    c10::cuda::CUDAGuard guard(p.device());
    auto dp = torch::empty_like(p);
    auto dv = need_v ? torch::zeros_like(v) : torch::empty({0}, v.options());
    int n = p.size(0) * p.size(1);
    transform_backward<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        p.data_ptr<float>(), v.data_ptr<float>(), grad.data_ptr<float>(), dp.data_ptr<float>(),
        need_v ? dv.data_ptr<float>() : nullptr, v.size(0), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dp, dv};
}
torch::Tensor edge_loss(torch::Tensor v, torch::Tensor edges, torch::Tensor base) {
    check_edges(v, edges, base);
    c10::cuda::CUDAGuard guard(v.device());
    int n = v.size(0) * edges.size(0);
    auto errors = torch::empty({n}, v.options());
    edge_forward<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        v.data_ptr<float>(), edges.data_ptr<int64_t>(), base.data_ptr<float>(),
        errors.data_ptr<float>(), v.size(1), edges.size(0), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return errors.mean();
}
torch::Tensor edge_grad(torch::Tensor v, torch::Tensor edges, torch::Tensor base,
                        torch::Tensor grad) {
    check_edges(v, edges, base);
    check(grad, v, torch::kFloat32);
    TORCH_CHECK(grad.numel() == 1, "Expected scalar loss gradient");
    c10::cuda::CUDAGuard guard(v.device());
    int n = v.size(0) * edges.size(0);
    auto out = torch::zeros_like(v);
    edge_backward<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        v.data_ptr<float>(), edges.data_ptr<int64_t>(), base.data_ptr<float>(),
        grad.data_ptr<float>(), out.data_ptr<float>(), v.size(1), edges.size(0), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
torch::Tensor sample(torch::Tensor grid, torch::Tensor nodes, torch::Tensor weights) {
    check_sampling(grid, nodes, weights);
    c10::cuda::CUDAGuard guard(grid.device());
    auto out = torch::empty({nodes.size(0), grid.size(1)}, grid.options());
    int n = out.numel();
    if (n)
        sample_forward<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            grid.data_ptr<float>(), nodes.data_ptr<int64_t>(), weights.data_ptr<float>(),
            out.data_ptr<float>(), grid.size(1), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
torch::Tensor sample_grad(torch::Tensor grad, torch::Tensor nodes, torch::Tensor weights,
                          int64_t count) {
    check_sampling(grad, nodes, weights);
    TORCH_CHECK(grad.size(0) == nodes.size(0) && count > 0 && count * grad.size(1) <= INT_MAX,
                "Invalid sampling gradient shape");
    c10::cuda::CUDAGuard guard(grad.device());
    auto out = torch::zeros({count, grad.size(1)}, grad.options());
    int n = grad.numel();
    if (n)
        sample_backward<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            grad.data_ptr<float>(), nodes.data_ptr<int64_t>(), weights.data_ptr<float>(),
            out.data_ptr<float>(), grad.size(1), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("transform", &transform);
    m.def("transform_grad", &transform_grad);
    m.def("edge_loss", &edge_loss);
    m.def("edge_grad", &edge_grad);
    m.def("sample", &sample);
    m.def("sample_grad", &sample_grad);
}
