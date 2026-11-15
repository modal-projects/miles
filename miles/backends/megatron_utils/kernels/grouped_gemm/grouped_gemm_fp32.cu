#include "grouped_gemm.cuh"

namespace miles_grouped_gemm {

namespace {

template <bool a_row_major, bool b_row_major>
void tf32_sm100(const at::Tensor& mat_a, const at::Tensor& mat_b, const at::Tensor& offs, at::Tensor& out) {
  using cutlass::arch::Sm100;
  using cutlass::tfloat32_t;
  // fp32 tiles take twice the shared memory of bf16, so K per tile is halved.
  if (use_small_tiles(mat_a, mat_b, offs)) {
    grouped_gemm_impl<tfloat32_t, float, Sm100, a_row_major, b_row_major, Sm100OneSm, cute::_128, cute::_256, cute::_32>(
        mat_a, mat_b, offs, out);
  } else {
    grouped_gemm_impl<tfloat32_t, float, Sm100, a_row_major, b_row_major, Sm100TwoSm, cute::_256, cute::_256, cute::_32>(
        mat_a, mat_b, offs, out);
  }
}

} // namespace

void grouped_mm_fp32_sm9x(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const at::Tensor& offs,
    at::Tensor& out,
    bool allow_tf32) {
  using cutlass::arch::Sm90;
  using cutlass::tfloat32_t;
  TORCH_CHECK(allow_tf32, "fp32 without TF32 runs on the sm80 3xTF32 kernel");
  const bool a_row_major = mat_a.stride(-1) == 1;
  const bool b_row_major = mat_b.stride(-1) == 1;
  if (at::cuda::getCurrentDeviceProperties()->major == 10) {
    if (a_row_major && b_row_major) {
      tf32_sm100<true, true>(mat_a, mat_b, offs, out);
    } else if (a_row_major) {
      tf32_sm100<true, false>(mat_a, mat_b, offs, out);
    } else if (b_row_major) {
      tf32_sm100<false, true>(mat_a, mat_b, offs, out);
    } else {
      tf32_sm100<false, false>(mat_a, mat_b, offs, out);
    }
    return;
  }
  if (!a_row_major && b_row_major) {
    // MN-major operands (a ragged K). With small tiles the sm80 kernel is faster than transposing B in shared memory.
    if (use_small_tiles(mat_a, mat_b, offs)) {
      grouped_mm_sm80(mat_a, mat_b, offs, out, allow_tf32);
      return;
    }
    // B is transposed in shared memory 128 bytes of K at a time, so K per tile is 32.
    grouped_gemm_impl<tfloat32_t, float, Sm90, false, true, Sm90Cooperative, cute::_128, cute::_256, cute::_32>(
        mat_a, mat_b, offs, out);
    return;
  }
  TORCH_CHECK(a_row_major && !b_row_major, "sm90 tf32 grouped_mm needs K-major or MN-major operands");
  if (use_small_tiles(mat_a, mat_b, offs)) {
    grouped_gemm_impl<tfloat32_t, float, Sm90, true, false, Sm90Pingpong, cute::_64, cute::_128, cute::_64>(
        mat_a, mat_b, offs, out);
  } else {
    grouped_gemm_impl<tfloat32_t, float, Sm90, true, false, Sm90Cooperative, cute::_128, cute::_256, cute::_32>(
        mat_a, mat_b, offs, out);
  }
}

} // namespace miles_grouped_gemm
