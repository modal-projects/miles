#include "grouped_gemm.cuh"

namespace miles_grouped_gemm {

void grouped_mm_fp32_sm9x(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const at::Tensor& offs,
    at::Tensor& out,
    bool allow_tf32) {
  using cutlass::arch::Sm100;
  using cutlass::arch::Sm90;
  using cutlass::tfloat32_t;
  bool small = use_small_tiles(mat_a, mat_b, offs);
  bool sm100 = at::cuda::getCurrentDeviceProperties()->major == 10;
  // fp32 tiles take twice the shared memory of bf16, so K per tile is halved.
  TORCH_CHECK(allow_tf32, "fp32 without TF32 runs on the sm80 3xTF32 kernel");
  if (sm100) {
    if (small) {
      grouped_gemm_impl<tfloat32_t, float, Sm100, true, false, Sm100OneSm, cute::_128, cute::_256, cute::_32>(
          mat_a, mat_b, offs, out);
    } else {
      grouped_gemm_impl<tfloat32_t, float, Sm100, true, false, Sm100TwoSm, cute::_256, cute::_256, cute::_32>(
          mat_a, mat_b, offs, out);
    }
  } else {
    if (small) {
      grouped_gemm_impl<tfloat32_t, float, Sm90, true, false, Sm90Pingpong, cute::_64, cute::_128, cute::_64>(
          mat_a, mat_b, offs, out);
    } else {
      grouped_gemm_impl<tfloat32_t, float, Sm90, true, false, Sm90Cooperative, cute::_128, cute::_256, cute::_32>(
          mat_a, mat_b, offs, out);
    }
  }
}

} // namespace miles_grouped_gemm
