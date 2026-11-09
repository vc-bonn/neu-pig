#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cfloat>
#include <climits>

namespace {
constexpr int R = 32;
constexpr int CELLS = R * R * R;

// Validate metadata without synchronizing the device (also safe in capture).
void check_tensor(const torch::Tensor& tensor, const torch::Tensor& anchor,
                  torch::ScalarType dtype, const char* name) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == anchor.device()
                && tensor.scalar_type() == dtype && tensor.is_contiguous(),
                name, " must be contiguous, have the required dtype, and share the CUDA device");
}

void check_points(const torch::Tensor& points) {
    check_tensor(points, points, torch::kFloat32, "points");
    TORCH_CHECK(points.dim() == 3 && points.size(0) > 0 && points.size(2) == 3,
                "Expected BxNx3 points with a nonempty batch");
    TORCH_CHECK(points.numel() <= INT_MAX, "Point cloud exceeds kernel index range");
    TORCH_CHECK(points.size(0) <= INT_MAX / CELLS, "Batch exceeds kernel index range");
}

void check_bounds(const torch::Tensor& points, const torch::Tensor& lower,
                  const torch::Tensor& width) {
    check_tensor(lower, points, torch::kFloat32, "lower");
    check_tensor(width, points, torch::kFloat32, "width");
    TORCH_CHECK(lower.dim() == 2 && lower.size(0) == points.size(0) && lower.size(1) == 3,
                "Expected Bx3 lower bounds");
    TORCH_CHECK(width.dim() == 1 && width.size(0) == points.size(0), "Expected B widths");
}

__device__ int coordinate(float p, float lo, float width) {
    return max(0, min(R - 1, int(floorf((p - lo) / width))));
}

__global__ void build(const float* points, const float* lower, const float* width,
                      int* heads, int* next, int count, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int batch = i / count;
    const float* p = points + i * 3;
    const float* lo = lower + batch * 3;
    float w = width[batch];
    int cell = (coordinate(p[0], lo[0], w) * R + coordinate(p[1], lo[1], w)) * R
               + coordinate(p[2], lo[2], w);
    next[i] = atomicExch(heads + batch * CELLS + cell, i % count);
}

__global__ void query(const float* points, const float* reference,
                      const float* lower, const float* widths,
                      const int* heads, const int* next,
                      float* distances, int64_t* indices,
                      int count, int reference_count, int total, const int64_t* order) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int batch = i / count;
    if (order) i = batch * count + order[i];
    const float* p = points + i * 3;
    const float* lo = lower + batch * 3;
    float w = widths[batch];
    int center[3] = {coordinate(p[0], lo[0], w), coordinate(p[1], lo[1], w),
                     coordinate(p[2], lo[2], w)};
    float best = FLT_MAX;
    int best_index = 0;
    for (int radius = 0; radius < R; ++radius) {
        int low[3], high[3];
        for (int axis = 0; axis < 3; ++axis) {
            low[axis] = max(0, center[axis] - radius);
            high[axis] = min(R - 1, center[axis] + radius);
        }
        for (int x = low[0]; x <= high[0]; ++x)
        for (int y = low[1]; y <= high[1]; ++y)
        for (int z = low[2]; z <= high[2]; ++z) {
            if (max(max(abs(x - center[0]), abs(y - center[1])), abs(z - center[2])) != radius)
                continue;
            int cell = (x * R + y) * R + z;
            int j = heads[batch * CELLS + cell];
            while (j != -1) {
                const float* other = reference + (batch * reference_count + j) * 3;
                float dx = p[0] - other[0], dy = p[1] - other[1], dz = p[2] - other[2];
                float d = dx * dx + dy * dy + dz * dz;
                if (d < best || (d == best && j < best_index)) {
                    best = d;
                    best_index = j;
                }
                j = next[batch * reference_count + j];
            }
        }
        // An unvisited cell lies beyond at least one of these six planes.
        // Ignore planes outside the grid. This also handles outside queries,
        // anisotropic clouds, and zero-width clouds without approximation.
        float boundary = FLT_MAX;
        for (int axis = 0; axis < 3; ++axis) {
            if (low[axis] > 0)
                boundary = fminf(boundary, fmaxf(p[axis] - (lo[axis] + low[axis] * w), 0.f));
            if (high[axis] < R - 1)
                boundary = fminf(boundary, fmaxf(lo[axis] + (high[axis] + 1) * w - p[axis], 0.f));
        }
        // Strict inequality preserves the lowest-index rule for exact ties.
        boundary = fmaxf(boundary - (1e-5f * w + 1e-7f), 0.f);
        if (best < boundary * boundary) break;
    }
    distances[i] = best;
    indices[i] = best_index;
}

__global__ void keys(const float* points, const float* lower, const float* widths,
                     int64_t* output, int count, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int batch = i / count;
    const float* p = points + i * 3;
    const float* lo = lower + batch * 3;
    float w = widths[batch];
    output[i] = (coordinate(p[0], lo[0], w) * R + coordinate(p[1], lo[1], w)) * R
                + coordinate(p[2], lo[2], w);
}

__global__ void backward(const float* points, const float* reference,
                          const int64_t* indices, const float* grad,
                          float* grad_points, float* grad_reference,
                          int count, int reference_count, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int j = (i / count) * reference_count + indices[i];
    for (int axis = 0; axis < 3; ++axis) {
        float value = 2.f * grad[i] * (points[i * 3 + axis] - reference[j * 3 + axis]);
        if (grad_points) grad_points[i * 3 + axis] = value;
        if (grad_reference) atomicAdd(grad_reference + j * 3 + axis, -value);
    }
}
}  // namespace

std::vector<torch::Tensor> spatial_build(torch::Tensor points,
                                        torch::Tensor lower, torch::Tensor width) {
    check_points(points);
    check_bounds(points, lower, width);
    c10::cuda::CUDAGuard guard(points.device());
    int batches = points.size(0), count = points.size(1);
    TORCH_CHECK(count > 0 && points.size(2) == 3, "Expected nonempty BxNx3 points");
    auto options = points.options().dtype(torch::kInt32);
    auto heads = torch::full({batches, CELLS}, -1, options);
    auto next = torch::empty({batches, count}, options);
    build<<<(batches * count + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        points.data_ptr<float>(), lower.data_ptr<float>(), width.data_ptr<float>(),
        heads.data_ptr<int>(), next.data_ptr<int>(), count, batches * count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {heads, next};
}

std::vector<torch::Tensor> spatial_query(torch::Tensor points, torch::Tensor reference,
                                        torch::Tensor lower, torch::Tensor width,
                                        torch::Tensor heads, torch::Tensor next, bool sort_queries) {
    check_points(points);
    check_points(reference);
    check_bounds(points, lower, width);
    TORCH_CHECK(points.device() == reference.device() && points.size(0) == reference.size(0)
                && reference.size(1) > 0, "Mismatched or empty reference point clouds");
    check_tensor(heads, points, torch::kInt32, "heads");
    check_tensor(next, points, torch::kInt32, "next");
    TORCH_CHECK(heads.dim() == 2 && heads.size(0) == points.size(0) && heads.size(1) == CELLS,
                "Expected BxCELLS heads");
    TORCH_CHECK(next.dim() == 2 && next.size(0) == reference.size(0)
                && next.size(1) == reference.size(1), "Expected BxN reference links");
    c10::cuda::CUDAGuard guard(points.device());
    int batches = points.size(0), count = points.size(1);
    auto distances = torch::empty({batches, count, 1}, points.options());
    auto indices = torch::empty({batches, count, 1}, points.options().dtype(torch::kInt64));
    torch::Tensor order;
    if (count) {
        if (sort_queries) {
            auto cell_keys = torch::empty({batches, count}, points.options().dtype(torch::kInt64));
            keys<<<(batches * count + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                points.data_ptr<float>(), lower.data_ptr<float>(), width.data_ptr<float>(),
                cell_keys.data_ptr<int64_t>(), count, batches * count);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            order = torch::argsort(cell_keys, false, int64_t(1), false);
        }
        query<<<(batches * count + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            points.data_ptr<float>(), reference.data_ptr<float>(), lower.data_ptr<float>(),
            width.data_ptr<float>(), heads.data_ptr<int>(), next.data_ptr<int>(),
            distances.data_ptr<float>(), indices.data_ptr<int64_t>(), count,
            reference.size(1), batches * count, sort_queries ? order.data_ptr<int64_t>() : nullptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {distances, indices};
}

std::vector<torch::Tensor> spatial_backward(torch::Tensor points, torch::Tensor reference,
                                           torch::Tensor indices, torch::Tensor grad,
                                           bool need_points, bool need_reference) {
    check_points(points);
    check_points(reference);
    TORCH_CHECK(points.device() == reference.device() && points.size(0) == reference.size(0)
                && reference.size(1) > 0, "Mismatched or empty reference point clouds");
    check_tensor(indices, points, torch::kInt64, "indices");
    check_tensor(grad, points, torch::kFloat32, "grad");
    TORCH_CHECK(indices.dim() == 3 && indices.size(0) == points.size(0)
                && indices.size(1) == points.size(1) && indices.size(2) == 1,
                "Expected BxNx1 nearest-neighbor indices");
    TORCH_CHECK(grad.sizes() == indices.sizes(), "Expected BxNx1 distance gradients");
    c10::cuda::CUDAGuard guard(points.device());
    auto grad_points = need_points ? torch::empty_like(points) : torch::empty({0}, points.options());
    auto grad_reference = need_reference ? torch::zeros_like(reference) : torch::empty({0}, points.options());
    int total = points.size(0) * points.size(1);
    if (total) {
        backward<<<(total + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            points.data_ptr<float>(), reference.data_ptr<float>(), indices.data_ptr<int64_t>(),
            grad.data_ptr<float>(), need_points ? grad_points.data_ptr<float>() : nullptr,
            need_reference ? grad_reference.data_ptr<float>() : nullptr,
            points.size(1), reference.size(1), total);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {grad_points, grad_reference};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build", &spatial_build);
    m.def("query", &spatial_query);
    m.def("backward", &spatial_backward);
}
