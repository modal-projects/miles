#include "grouped_gemm.cuh"

namespace miles_grouped_gemm {

void grouped_mm_bf16(const at::Tensor& mat_a, const at::Tensor& mat_b, const at::Tensor& offs, at::Tensor& out) {
  dispatch_on_layout<cutlass::bfloat16_t>(mat_a, mat_b, offs, out);
}

} // namespace miles_grouped_gemm
