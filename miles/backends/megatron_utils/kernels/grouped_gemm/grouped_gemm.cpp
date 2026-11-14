// Registers miles_grouped_gemm::grouped_mm, torch._grouped_mm's CUTLASS kernel for bf16 and fp16 on sm90/sm100,
// with no per-group cuBLAS fallback. Shape and stride rules follow PyTorch v2.13.0 GroupedMMUtils.h.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <torch/library.h>

namespace miles_grouped_gemm {

void grouped_mm_bf16(const at::Tensor& mat_a, const at::Tensor& mat_b, const at::Tensor& offs, at::Tensor& out);
void grouped_mm_fp16(const at::Tensor& mat_a, const at::Tensor& mat_b, const at::Tensor& offs, at::Tensor& out);

namespace {

void check_strides(const at::Tensor& mat) {
  auto strides = mat.strides();
  auto sizes = mat.sizes();
  int last = mat.dim() - 1;
  int64_t alignment = 16 / mat.element_size();
  TORCH_CHECK(reinterpret_cast<uintptr_t>(mat.data_ptr()) % 16 == 0, "expected data_ptr to be aligned to 16 bytes");
  if (strides[last - 1] == 1 && strides[last] >= std::max<int64_t>(1, sizes[last - 1])) {
    TORCH_CHECK(strides[last] % alignment == 0, "strides should be multiple of 16 bytes");
  } else if (strides[last] == 1 && strides[last - 1] >= std::max<int64_t>(1, sizes[last])) {
    TORCH_CHECK(strides[last - 1] % alignment == 0, "strides should be multiple of 16 bytes");
  } else {
    TORCH_CHECK(false, "Invalid strides/sizes, got ", strides, " for strides and ", sizes, " for sizes");
  }
}

at::Tensor make_output(const at::Tensor& mat_a, const at::Tensor& mat_b, const at::Tensor& offs) {
  const bool a_is_2d = mat_a.dim() == 2;
  const bool b_is_2d = mat_b.dim() == 2;
  c10::SmallVector<int64_t, 3> size;
  if (a_is_2d && b_is_2d) {
    size = {offs.size(0), mat_a.size(0), mat_b.size(1)};
  } else if (a_is_2d) {
    TORCH_CHECK(offs.size(0) == mat_b.size(0), "matrix batch sizes have to match");
    size = {mat_a.size(0), mat_b.size(-1)};
  } else {
    TORCH_CHECK(offs.size(0) == mat_a.size(0), "matrix batch sizes have to match");
    size = {mat_a.size(1), mat_b.size(1)};
  }
  const int64_t alignment = 16 / mat_a.element_size();
  const int64_t padded = (size.back() + alignment - 1) / alignment * alignment;
  c10::SmallVector<int64_t, 3> stride;
  if (a_is_2d != b_is_2d) {
    stride = {padded, 1};
  } else {
    stride = {size[1] * padded, padded, 1};
  }
  return at::empty_strided(size, stride, mat_a.options());
}

at::Tensor grouped_mm_cuda(const at::Tensor& mat_a, const at::Tensor& mat_b, const at::Tensor& offs) {
  TORCH_CHECK(mat_a.is_cuda() && mat_b.device() == mat_a.device() && offs.device() == mat_a.device(),
              "grouped_mm expects all inputs on one CUDA device");
  TORCH_CHECK(mat_a.scalar_type() == mat_b.scalar_type(), "mat_a and mat_b must have the same dtype");
  TORCH_CHECK(mat_a.scalar_type() == at::kBFloat16 || mat_a.scalar_type() == at::kHalf,
              "grouped_mm supports bfloat16 and float16, got ", mat_a.scalar_type());
  TORCH_CHECK((mat_a.dim() == 2 || mat_a.dim() == 3) && (mat_b.dim() == 2 || mat_b.dim() == 3),
              "mat_a and mat_b have to be 2 or 3d");
  TORCH_CHECK(mat_a.dim() == 2 || mat_b.dim() == 2, "grouped_mm needs a 2d ragged operand");
  if (mat_a.dim() != mat_b.dim()) {
    TORCH_CHECK(mat_a.size(-1) == mat_b.size(-2), "contraction dimension of mat_a and mat_b must match");
  }
  TORCH_CHECK(offs.dim() == 1 && offs.scalar_type() == at::kInt, "offs has to be a 1d int32 tensor");
  check_strides(mat_a);
  check_strides(mat_b);

  c10::cuda::CUDAGuard device_guard(mat_a.device());
  const int major = at::cuda::getCurrentDeviceProperties()->major;
  TORCH_CHECK(major == 9 || major == 10, "grouped_mm needs an sm90 or sm100 GPU, got compute capability major ", major);

  at::Tensor out = make_output(mat_a, mat_b, offs);
  if (mat_a.scalar_type() == at::kBFloat16) {
    grouped_mm_bf16(mat_a, mat_b, offs, out);
  } else {
    grouped_mm_fp16(mat_a, mat_b, offs, out);
  }
  return out;
}

} // namespace
} // namespace miles_grouped_gemm

TORCH_LIBRARY(miles_grouped_gemm, m) {
  m.def("grouped_mm(Tensor mat_a, Tensor mat_b, Tensor offs) -> Tensor");
}

TORCH_LIBRARY_IMPL(miles_grouped_gemm, CUDA, m) {
  m.impl("grouped_mm", &miles_grouped_gemm::grouped_mm_cuda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
