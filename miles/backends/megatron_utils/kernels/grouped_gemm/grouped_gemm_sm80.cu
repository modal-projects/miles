// CUTLASS 2.x grouped GEMM for sm80+ (mma.sync): bf16, fp16, TF32 and 3xTF32 fp32. One tile config per dtype and
// no split-K, so each output tile is summed in the same order whatever the other groups are.
#include "grouped_gemm.cuh"

#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>

#include "cutlass_ext/residual_last_mma.h"

namespace miles_grouped_gemm {
namespace {

// Strides are passed as (batch, rows, cols) of each operand; 2d operands have batch stride 0. The output is row-major.
template <typename ElementIn, typename ElementOut, bool a_row_major, bool b_row_major>
__global__ void prepare_sm80_data(
    ElementIn* A,
    ElementIn* B,
    ElementOut* output,
    ElementIn** A_ptrs,
    ElementIn** B_ptrs,
    ElementOut** output_ptrs,
    cutlass::gemm::GemmCoord* problem_sizes,
    int64_t* lda,
    int64_t* ldb,
    int64_t* ldd,
    const int32_t* offs,
    int32_t M,
    int32_t N,
    int32_t K,
    Strides strideA,
    Strides strideB,
    Strides strideOutput) {
  constexpr int align_in = 16 / sizeof(ElementIn);
  constexpr int align_out = 16 / sizeof(ElementOut);
  int32_t tid = threadIdx.x;
  int32_t start = tid == 0 ? 0 : offs[tid - 1];
  int32_t delta = offs[tid] - start;
  CUDA_KERNEL_ASSERT(delta >= 0 && "expected gemm dimension to be greater or equal 0\n");
  lda[tid] = a_row_major ? strideA[1] : strideA[2];
  ldb[tid] = b_row_major ? strideB[1] : strideB[2];
  ldd[tid] = strideOutput[1];
  if (M < 0) {
    M = delta;
    A_ptrs[tid] = A + start * strideA[1];
    B_ptrs[tid] = B + tid * strideB[0];
    output_ptrs[tid] = output + start * strideOutput[1];
  } else if (N < 0) {
    CUDA_KERNEL_ASSERT(start % align_out == 0 && "expected output dynamic dimension to be a multiple of 16 bytes\n");
    N = delta;
    A_ptrs[tid] = A + tid * strideA[0];
    B_ptrs[tid] = B + start * strideB[2];
    output_ptrs[tid] = output + start;
  } else {
    // Vector loads along a contiguous K would read past a group that ends mid-vector.
    CUDA_KERNEL_ASSERT(
        ((!a_row_major && b_row_major) || delta % align_in == 0) &&
        "expected input dynamic dimension to be a multiple of 16 bytes\n");
    K = delta;
    A_ptrs[tid] = A + start * strideA[2];
    B_ptrs[tid] = B + start * strideB[1];
    output_ptrs[tid] = output + tid * strideOutput[0];
  }
  problem_sizes[tid] = cutlass::gemm::GemmCoord(M, N, K);
}

template <
    typename ElementIn,
    typename ElementOut,
    bool a_row_major,
    bool b_row_major,
    typename MathOp,
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    int Stages,
    bool ragged_k = false>
void sm80_impl(const at::Tensor& mat_a, const at::Tensor& mat_b, const at::Tensor& offs, at::Tensor& out) {
  constexpr int AlignmentIn = 16 / sizeof(ElementIn);
  constexpr int AlignmentOut = 16 / sizeof(ElementOut);
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOut, AlignmentOut, float, float>;
  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
      ElementIn,
      std::conditional_t<a_row_major, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>,
      cutlass::ComplexTransform::kNone,
      AlignmentIn,
      ElementIn,
      std::conditional_t<b_row_major, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>,
      cutlass::ComplexTransform::kNone,
      AlignmentIn,
      ElementOut,
      cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOp,
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
      Stages,
      cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
      MathOp>::GemmKernel;
  using LastTileKernel = cutlass::gemm::kernel::GemmGrouped<
      typename WithLastTileResidue<typename GemmKernel::Mma>::type,
      typename GemmKernel::Epilogue,
      typename GemmKernel::ThreadblockSwizzle,
      cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
      GemmKernel::kTransposed>;
  using Gemm = cutlass::gemm::device::GemmGrouped<std::conditional_t<ragged_k, LastTileKernel, GemmKernel>>;

  int32_t M = mat_a.size(-2);
  int32_t N = mat_b.size(-1);
  int32_t K = mat_a.size(-1);
  int32_t group_count = offs.size(0);
  if (mat_a.dim() == 2 && mat_b.dim() == 2) {
    K = -1;
  } else if (mat_a.dim() == 2) {
    M = -1;
  } else {
    N = -1;
  }
  TORCH_CHECK(group_count < 1024, "Can't process more than 1024 groups");
  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  const int64_t buf_size =
      group_count * (3 * sizeof(int64_t) + 3 * sizeof(void*) + sizeof(cutlass::gemm::GemmCoord));
  auto buf = allocator.allocate(buf_size);
  int64_t* lda = reinterpret_cast<int64_t*>(buf.get());
  int64_t* ldb = lda + group_count;
  int64_t* ldd = ldb + group_count;
  ElementIn** A_ptrs = reinterpret_cast<ElementIn**>(ldd + group_count);
  ElementIn** B_ptrs = A_ptrs + group_count;
  ElementOut** output_ptrs = reinterpret_cast<ElementOut**>(B_ptrs + group_count);
  auto* problem_sizes = reinterpret_cast<cutlass::gemm::GemmCoord*>(output_ptrs + group_count);

  auto stream = at::cuda::getCurrentCUDAStream();
  auto make_strides = [](at::IntArrayRef values) {
    Strides result{};
    std::copy(values.begin(), values.end(), result.begin() + 3 - values.size());
    return result;
  };
  prepare_sm80_data<ElementIn, ElementOut, a_row_major, b_row_major><<<1, group_count, 0, stream.stream()>>>(
      reinterpret_cast<ElementIn*>(mat_a.data_ptr()),
      reinterpret_cast<ElementIn*>(mat_b.data_ptr()),
      reinterpret_cast<ElementOut*>(out.data_ptr()),
      A_ptrs,
      B_ptrs,
      output_ptrs,
      problem_sizes,
      lda,
      ldb,
      ldd,
      offs.const_data_ptr<int32_t>(),
      M,
      N,
      K,
      make_strides(mat_a.strides()),
      make_strides(mat_b.strides()),
      make_strides(out.strides()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int sm_count = at::cuda::getDeviceProperties(out.device().index())->multiProcessorCount;
  const int blocks_per_sm = Gemm::maximum_active_blocks();
  TORCH_CHECK(blocks_per_sm > 0, "grouped_mm sm80 kernel does not fit on this GPU");
  typename Gemm::Arguments arguments(
      problem_sizes,
      group_count,
      blocks_per_sm * sm_count,
      {1.0f, 0.0f},
      A_ptrs,
      B_ptrs,
      output_ptrs,
      output_ptrs,
      lda,
      ldb,
      ldd,
      ldd);
  auto workspace = allocator.allocate(Gemm::get_workspace_size(arguments));
  Gemm gemm;
  TORCH_CHECK(gemm.initialize(arguments, workspace.get()) == cutlass::Status::kSuccess, "cutlass cannot initialize");
  auto status = gemm.run(stream.stream());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "cutlass cannot run, error ", int(status));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

using cutlass::gemm::GemmShape;

template <typename ElementIn, typename MathOp, typename Tile, typename Warp, typename Instruction, int Stages>
void sm80_dispatch(const at::Tensor& mat_a, const at::Tensor& mat_b, const at::Tensor& offs, at::Tensor& out) {
  const bool a_row_major = mat_a.stride(-1) == 1;
  const bool b_row_major = mat_b.stride(-1) == 1;
  if (mat_a.dim() == 2 && mat_b.dim() == 2) {
    TORCH_CHECK(!a_row_major && b_row_major, "grouped_mm sm80 kernel needs K-strided operands for a ragged K");
    sm80_impl<ElementIn, ElementIn, false, true, MathOp, Tile, Warp, Instruction, Stages, true>(
        mat_a, mat_b, offs, out);
  } else if (a_row_major && !b_row_major) {
    sm80_impl<ElementIn, ElementIn, true, false, MathOp, Tile, Warp, Instruction, Stages>(mat_a, mat_b, offs, out);
  } else if (a_row_major) {
    sm80_impl<ElementIn, ElementIn, true, true, MathOp, Tile, Warp, Instruction, Stages>(mat_a, mat_b, offs, out);
  } else {
    TORCH_CHECK(b_row_major, "grouped_mm sm80 kernel needs a row-major mat_a or a row-major mat_b");
    sm80_impl<ElementIn, ElementIn, false, true, MathOp, Tile, Warp, Instruction, Stages>(mat_a, mat_b, offs, out);
  }
}

template <int M, int N, int K, int WarpM, int WarpN, int Stages_>
struct Config {
  using Tile = GemmShape<M, N, K>;
  using Warp = GemmShape<WarpM, WarpN, K>;
  static constexpr int Stages = Stages_;
};

template <typename ElementIn, typename MathOp, typename Instruction, typename Small, typename Large>
void sm80_select(
    bool small,
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const at::Tensor& offs,
    at::Tensor& out) {
  if (small) {
    sm80_dispatch<ElementIn, MathOp, typename Small::Tile, typename Small::Warp, Instruction, Small::Stages>(
        mat_a, mat_b, offs, out);
  } else {
    sm80_dispatch<ElementIn, MathOp, typename Large::Tile, typename Large::Warp, Instruction, Large::Stages>(
        mat_a, mat_b, offs, out);
  }
}

template <typename ElementIn>
void sm80_select_16bit(
    int major,
    bool small,
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const at::Tensor& offs,
    at::Tensor& out) {
  using Add = cutlass::arch::OpMultiplyAdd;
  using Instr = GemmShape<16, 8, 16>;
  using Small = Config<64, 64, 64, 32, 32, 5>;
  if (major == 12) {
    sm80_select<ElementIn, Add, Instr, Small, Config<128, 128, 32, 64, 64, 3>>(small, mat_a, mat_b, offs, out);
  } else {
    sm80_select<ElementIn, Add, Instr, Small, Config<128, 128, 32, 64, 64, 5>>(small, mat_a, mat_b, offs, out);
  }
}

} // namespace

void grouped_mm_sm80(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const at::Tensor& offs,
    at::Tensor& out,
    bool allow_tf32) {
  using Add = cutlass::arch::OpMultiplyAdd;
  using FastF32 = cutlass::arch::OpMultiplyAddFastF32;
  using F32Instr = GemmShape<16, 8, 8>;
  // Tiles from CUTLASS Python sweeps on A100, RTX PRO 6000, H100 and B200. The choice depends only on the GPU and
  // the output's non-ragged dims, so a group's accumulation order never depends on the other groups.
  const int major = at::cuda::getCurrentDeviceProperties()->major;
  int64_t out_fixed = std::min(mat_a.size(-2), mat_b.size(-1));
  if (mat_b.dim() == 3) {
    out_fixed = mat_b.size(-1);
  } else if (mat_a.dim() == 3) {
    out_fixed = mat_a.size(-2);
  }
  const bool small = out_fixed < 128;
  switch (mat_a.scalar_type()) {
    case at::kBFloat16:
      sm80_select_16bit<cutlass::bfloat16_t>(major, small, mat_a, mat_b, offs, out);
      break;
    case at::kHalf:
      sm80_select_16bit<cutlass::half_t>(major, small, mat_a, mat_b, offs, out);
      break;
    default:
      using Tf32Small = Config<64, 64, 32, 32, 32, 5>;
      using Fp32Small = Config<64, 64, 32, 32, 32, 3>;
      if (allow_tf32 && major == 12) {
        sm80_select<float, Add, F32Instr, Tf32Small, Config<128, 128, 16, 64, 64, 5>>(small, mat_a, mat_b, offs, out);
      } else if (allow_tf32) {
        sm80_select<float, Add, F32Instr, Tf32Small, Config<128, 128, 16, 64, 64, 3>>(small, mat_a, mat_b, offs, out);
      } else if (major == 8) {
        sm80_select<float, FastF32, F32Instr, Fp32Small, Config<64, 128, 16, 32, 64, 4>>(
            small, mat_a, mat_b, offs, out);
      } else if (major == 12) {
        sm80_select<float, FastF32, F32Instr, Fp32Small, Config<64, 64, 16, 32, 32, 3>>(
            small, mat_a, mat_b, offs, out);
      } else {
        sm80_select<float, FastF32, F32Instr, Fp32Small, Config<64, 128, 32, 32, 64, 3>>(
            small, mat_a, mat_b, offs, out);
      }
  }
}

} // namespace miles_grouped_gemm
