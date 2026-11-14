// Registers miles_grouped_gemm::grouped_mm: torch._grouped_mm's CUTLASS kernel on sm90/sm100 for bf16, fp16 and
// fp32, and a CUTLASS 2.x grouped kernel on sm80, with no per-group cuBLAS fallback. Shape and stride rules follow PyTorch v2.13.0 GroupedMMUtils.h.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <tuple>

namespace miles_grouped_gemm {

void grouped_mm_bf16(const at::Tensor& mat_a, const at::Tensor& mat_b, const at::Tensor& offs, at::Tensor& out);
void grouped_mm_fp16(const at::Tensor& mat_a, const at::Tensor& mat_b, const at::Tensor& offs, at::Tensor& out);
void grouped_mm_fp32_sm9x(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const at::Tensor& offs,
    at::Tensor& out,
    bool allow_tf32);
void grouped_mm_sm80(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const at::Tensor& offs,
    at::Tensor& out,
    bool allow_tf32);

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

// Copy with K contiguous and the leading dimension padded to 16 bytes.
at::Tensor k_major_copy(const at::Tensor& t, bool row_major) {
  const int64_t d = t.dim();
  const int64_t align = 16 / t.element_size();
  const int64_t inner_dim = row_major ? d - 1 : d - 2;
  const int64_t outer_dim = row_major ? d - 2 : d - 1;
  const int64_t ld = (t.size(inner_dim) + align - 1) / align * align;
  std::vector<int64_t> strides(d);
  strides[inner_dim] = 1;
  strides[outer_dim] = ld;
  if (d == 3) {
    strides[0] = ld * t.size(outer_dim);
  }
  at::Tensor out = at::empty_strided(t.sizes(), strides, t.options());
  out.copy_(t);
  return out;
}

// With K contiguous, a K-ragged group must start and end on 16 bytes. Copies each group into its own zero-padded slot
// of a multiple of `align` rows; the padding depends only on the group's own size, so it is the same solo and packed.
// Rows past offs[-1] go beyond every slot.
std::tuple<at::Tensor, at::Tensor, at::Tensor> pad_k_groups(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const at::Tensor& offs,
    int64_t align,
    bool k_major) {
  const int64_t total = mat_a.size(1);
  const int64_t groups = offs.size(0);
  at::Tensor ends = offs.to(at::kLong);
  at::Tensor starts = at::cat({at::zeros({1}, ends.options()), ends.slice(0, 0, groups - 1)});
  at::Tensor padded_ends = ((ends - starts + align - 1).div(align, "floor") * align).cumsum(0);
  at::Tensor padded_starts = at::cat({at::zeros({1}, ends.options()), padded_ends.slice(0, 0, groups - 1)});
  const int64_t width = (total + groups * (align - 1) + align - 1) / align * align;
  // Each padded K row is gathered from its source row; padding rows read row 0 and are zeroed.
  at::Tensor rows = at::arange(width, ends.options());
  at::Tensor group = at::searchsorted(padded_ends, rows, /*out_int32=*/false, /*right=*/true).clamp_max_(groups - 1);
  at::Tensor src = rows - padded_starts.index_select(0, group) + starts.index_select(0, group);
  at::Tensor pad = src.ge(ends.index_select(0, group)).unsqueeze(1);
  src.masked_fill_(pad.squeeze(1), 0);
  // (K, M) and (K, N) with M and N contiguous and their leading dimensions padded to 16 bytes.
  auto gather = [&](const at::Tensor& k_rows) {
    const int64_t cols = k_rows.size(1);
    const int64_t elems = 16 / k_rows.element_size();
    const int64_t ld = (cols + elems - 1) / elems * elems;
    at::Tensor gathered = k_rows.index_select(0, src).masked_fill_(pad, 0);
    if (ld == cols) {
      return gathered;
    }
    return at::zeros({width, ld}, gathered.options()).narrow(1, 0, cols).copy_(gathered);
  };
  at::Tensor a = gather(mat_a.mT());
  at::Tensor b = gather(mat_b);
  if (k_major) {
    return {a.mT().contiguous(), b.mT().contiguous().mT(), padded_ends.to(at::kInt)};
  }
  return {a.mT(), b, padded_ends.to(at::kInt)};
}

at::Tensor grouped_mm_cuda(const at::Tensor& mat_a_in, const at::Tensor& mat_b_in, const at::Tensor& offs_in, bool allow_tf32) {
  at::Tensor offs = offs_in;
  at::Tensor mat_a = mat_a_in;
  at::Tensor mat_b = mat_b_in;
  TORCH_CHECK(mat_a.is_cuda() && mat_b.device() == mat_a.device() && offs.device() == mat_a.device(),
              "grouped_mm expects all inputs on one CUDA device");
  TORCH_CHECK(mat_a.scalar_type() == mat_b.scalar_type(), "mat_a and mat_b must have the same dtype");
  const auto dtype = mat_a.scalar_type();
  TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kHalf || dtype == at::kFloat,
              "grouped_mm supports bfloat16, float16 and float32, got ", dtype);
  TORCH_CHECK((mat_a.dim() == 2 || mat_a.dim() == 3) && (mat_b.dim() == 2 || mat_b.dim() == 3),
              "mat_a and mat_b have to be 2 or 3d");
  TORCH_CHECK(mat_a.dim() == 2 || mat_b.dim() == 2, "grouped_mm needs a 2d ragged operand");
  if (mat_a.dim() != mat_b.dim()) {
    TORCH_CHECK(mat_a.size(-1) == mat_b.size(-2), "contraction dimension of mat_a and mat_b must match");
  }
  TORCH_CHECK(offs.dim() == 1 && offs.scalar_type() == at::kInt, "offs has to be a 1d int32 tensor");

  c10::cuda::CUDAGuard device_guard(mat_a.device());
  const int major = at::cuda::getCurrentDeviceProperties()->major;
  TORCH_CHECK(major == 8 || major == 9 || major == 10 || major == 12,
              "grouped_mm needs an sm80, sm90, sm100 or sm120 GPU, got compute capability major ", major);
  const bool sm80_kernel = major == 8 || major == 12 || (dtype == at::kFloat && !allow_tf32);
  if (mat_a.dim() == 2 && mat_b.dim() == 2 && (sm80_kernel || dtype == at::kFloat)) {
    // The sm80 mainloop runs the partial K tile first, so zero rows appended to a group would shift every tile and
    // change its sums; padding each group to whole K tiles (64 rows covers every config) keeps the order fixed. The
    // TMA tf32 kernels instead need each group to start on 16 bytes along a contiguous K.
    if (sm80_kernel) {
      std::tie(mat_a, mat_b, offs) = pad_k_groups(mat_a, mat_b, offs, 64, /*k_major=*/false);
    } else {
      std::tie(mat_a, mat_b, offs) = pad_k_groups(mat_a, mat_b, offs, 16 / mat_a.element_size(), /*k_major=*/true);
    }
  } else if (sm80_kernel) {
    // Column-major mat_a with column-major mat_b has no sm80 instantiation.
    if (mat_a.stride(-1) != 1 && mat_b.stride(-1) != 1) {
      mat_a = k_major_copy(mat_a, /*row_major=*/true);
    }
  } else if (dtype == at::kFloat) {
    // The TMA tf32 kernels read only K-major operands (row-major A, column-major B).
    if (mat_a.stride(-1) != 1) {
      mat_a = k_major_copy(mat_a, /*row_major=*/true);
    }
    if (mat_b.stride(-2) != 1) {
      mat_b = k_major_copy(mat_b, /*row_major=*/false);
    }
  }
  check_strides(mat_a);
  check_strides(mat_b);

  at::Tensor out = make_output(mat_a, mat_b, offs);
  if (sm80_kernel) {
    grouped_mm_sm80(mat_a, mat_b, offs, out, allow_tf32);
  } else if (dtype == at::kBFloat16) {
    grouped_mm_bf16(mat_a, mat_b, offs, out);
  } else if (dtype == at::kHalf) {
    grouped_mm_fp16(mat_a, mat_b, offs, out);
  } else {
    grouped_mm_fp32_sm9x(mat_a, mat_b, offs, out, allow_tf32);
  }
  return out;
}

} // namespace
} // namespace miles_grouped_gemm

TORCH_LIBRARY(miles_grouped_gemm, m) {
  m.def("grouped_mm(Tensor mat_a, Tensor mat_b, Tensor offs, bool allow_tf32=False) -> Tensor");
}

TORCH_LIBRARY_IMPL(miles_grouped_gemm, CUDA, m) {
  m.impl("grouped_mm", &miles_grouped_gemm::grouped_mm_cuda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
