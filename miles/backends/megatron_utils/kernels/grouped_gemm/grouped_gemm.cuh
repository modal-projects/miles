// Adapted from PyTorch v2.13.0 aten/src/ATen/native/cuda/{GroupMM.cu,GroupMMCommon.cuh,cutlass_common.cuh}
// (BSD-3-Clause, Copyright (c) Meta Platforms, Inc. and affiliates), templated on the input dtype so
// fp16 gets the same CUTLASS grouped GEMM that torch only builds for bf16. Built against CUTLASS v4.4.2,
// the version PyTorch v2.13.0 pins.
// PyTorch license: LICENSE.pytorch in this directory.
#pragma once

#include <array>

#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>

#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/util/packed_stride.hpp>

namespace miles_grouped_gemm {

using Strides = std::array<int64_t, 3>;

template <typename Kernel>
struct enable_for_sm9x : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000
    Kernel::operator()(std::forward<Args>(args)...);
#endif
  }
};

template <typename Kernel>
struct enable_for_sm10x : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200
    Kernel::operator()(std::forward<Args>(args)...);
#endif
  }
};

// One thread per group turns the device-side offsets into per-group pointers, problem shapes and strides,
// so no offsets ever reach the host.
template <
    typename DtypeA,
    typename DtypeB,
    typename DtypeOutput,
    typename ProblemShape,
    typename StrideA,
    typename StrideB,
    typename StrideOutput>
__global__ void prepare_grouped_gemm_data(
    DtypeA* A,
    DtypeB* B,
    DtypeOutput* output,
    DtypeA** A_ptrs,
    DtypeB** B_ptrs,
    DtypeOutput** output_ptrs,
    ProblemShape* problem_sizes,
    StrideA* stride_A,
    StrideB* stride_B,
    StrideOutput* stride_output,
    const int32_t* offs,
    int32_t M,
    int32_t N,
    int32_t K,
    Strides tensor_StrideA,
    Strides tensor_StrideB,
    Strides tensor_StrideOutput,
    Strides tensor_ShapeA,
    Strides tensor_ShapeB,
    bool a_row_major,
    bool b_row_major) {
  int32_t tid = threadIdx.x;
  int32_t start = tid == 0 ? 0 : offs[tid - 1];
  int32_t offset = offs[tid];
  int32_t delta = offset - start;
  CUDA_KERNEL_ASSERT(delta >= 0 && "expected gemm dimension to be greater or equal 0\n");

  // TMA transfers require global memory tensor addresses to be aligned to 16 bytes.
  if (tid < blockDim.x - 1) {
    if ((K < 0 && a_row_major) || (M < 0 && !a_row_major)) {
      int align = 128 / cutlass::sizeof_bits<DtypeA>::value;
      CUDA_KERNEL_ASSERT(delta % align == 0 && "expected input dynamic dimension to be a multiple of 16 bytes\n");
    }
    if ((K < 0 && !b_row_major) || (N < 0 && b_row_major)) {
      int align = 128 / cutlass::sizeof_bits<DtypeB>::value;
      CUDA_KERNEL_ASSERT(delta % align == 0 && "expected input dynamic dimension to be a multiple of 16 bytes\n");
    }
    if (N < 0) {
      int align = 128 / cutlass::sizeof_bits<DtypeOutput>::value;
      CUDA_KERNEL_ASSERT(delta % align == 0 && "expected output dynamic dimension to be a multiple of 16 bytes\n");
    }
  }
  int64_t lda, ldb, ldoutput;
  if (M < 0) {
    CUDA_KERNEL_ASSERT(offset <= tensor_ShapeA[0] && "expected offset to be less than tensor size\n");
    M = delta;
    lda = a_row_major ? tensor_StrideA[0] : tensor_StrideA[1];
    ldb = b_row_major ? tensor_StrideB[1] : tensor_StrideB[2];
    ldoutput = tensor_StrideOutput[0];
    A_ptrs[tid] = A + start * tensor_StrideA[0];
    output_ptrs[tid] = output + start * ldoutput;
    B_ptrs[tid] = B + tid * tensor_StrideB[0];
  } else if (N < 0) {
    CUDA_KERNEL_ASSERT(offset <= tensor_ShapeB[1] && "expected offset to be less than tensor size\n");
    N = delta;
    lda = a_row_major ? tensor_StrideA[1] : tensor_StrideA[2];
    ldb = b_row_major ? tensor_StrideB[0] : tensor_StrideB[1];
    ldoutput = tensor_StrideOutput[0];
    A_ptrs[tid] = A + tid * tensor_StrideA[0];
    output_ptrs[tid] = output + start;
    B_ptrs[tid] = B + start * tensor_StrideB[1];
  } else {
    CUDA_KERNEL_ASSERT(
        offset <= tensor_ShapeA[1] && offset <= tensor_ShapeB[0] && "expected offset to be less than tensor size\n");
    K = delta;
    lda = a_row_major ? tensor_StrideA[0] : tensor_StrideA[1];
    ldb = b_row_major ? tensor_StrideB[0] : tensor_StrideB[1];
    ldoutput = tensor_StrideOutput[1];
    A_ptrs[tid] = A + start * tensor_StrideA[1];
    B_ptrs[tid] = B + start * tensor_StrideB[0];
    output_ptrs[tid] = output + tid * tensor_StrideOutput[0];
  }
  problem_sizes[tid] = ProblemShape(M, N, K);
  // make_cute_packed_stride replaces only the stride element that matters for the layout, so passing
  // the leading dimension twice avoids branching on row/column major.
  stride_A[tid] = cutlass::make_cute_packed_stride(StrideA{}, {lda, lda, 1});
  stride_B[tid] = cutlass::make_cute_packed_stride(StrideB{}, {ldb, ldb, 1});
  stride_output[tid] = cutlass::make_cute_packed_stride(StrideOutput{}, {M, ldoutput, 1});
}

template <typename ArchTag, bool PONGOr2SM>
struct Schedule {
  using CooperativeSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
  using PongSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
  using CooperativeEpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
  using PongEpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using MMA1SMKernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100;
  using MMA1SMEpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
  using MMA2SMKernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100;
  using MMA2SMEpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;

  static constexpr bool is_sm100 = std::is_same_v<ArchTag, cutlass::arch::Sm100>;
  using KernelSchedule = cute::conditional_t<
      is_sm100,
      cute::conditional_t<PONGOr2SM, MMA2SMKernelSchedule, MMA1SMKernelSchedule>,
      cute::conditional_t<PONGOr2SM, PongSchedule, CooperativeSchedule>>;
  using EpilogueSchedule = cute::conditional_t<
      is_sm100,
      cute::conditional_t<PONGOr2SM, MMA2SMEpilogueSchedule, MMA1SMEpilogueSchedule>,
      cute::conditional_t<PONGOr2SM, PongEpilogueSchedule, CooperativeEpilogueSchedule>>;
};

inline int round_up_to_nearest_multiple(int a, int b) {
  return (a + b - 1) / b * b;
}

template <
    typename Dtype,
    typename ArchTag,
    bool a_row_major,
    bool b_row_major,
    bool PONGOr2SM,
    typename TB_M,
    typename TB_N,
    typename TB_K>
void grouped_gemm_impl(const at::Tensor& mat_a, const at::Tensor& mat_b, const at::Tensor& offs, at::Tensor& out) {
  using DtypeAccum = float;
  using LayoutA = cute::conditional_t<a_row_major, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>;
  using LayoutB = cute::conditional_t<b_row_major, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>;
  using LayoutOutput = cutlass::layout::RowMajor;
  constexpr int Alignment = 16 / sizeof(Dtype);
  using TileShape = cute::Shape<TB_M, TB_N, TB_K>;
  using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;
  using KernelSchedule = typename Schedule<ArchTag, PONGOr2SM>::KernelSchedule;
  using EpilogueSchedule = typename Schedule<ArchTag, PONGOr2SM>::EpilogueSchedule;
  using ProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int32_t, int32_t, int32_t>>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      cutlass::arch::OpClassTensorOp,
      TileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      DtypeAccum,
      DtypeAccum,
      void,
      LayoutOutput*,
      Alignment,
      Dtype,
      LayoutOutput*,
      Alignment,
      EpilogueSchedule,
      cutlass::epilogue::fusion::LinearCombination<Dtype, DtypeAccum>>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      cutlass::arch::OpClassTensorOp,
      Dtype,
      LayoutA*,
      Alignment,
      Dtype,
      LayoutB*,
      Alignment,
      DtypeAccum,
      TileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  using GemmKernelBase = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
  using GemmKernel = std::conditional_t<
      std::is_same_v<ArchTag, cutlass::arch::Sm100>,
      enable_for_sm10x<GemmKernelBase>,
      enable_for_sm9x<GemmKernelBase>>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideOutput = typename Gemm::GemmKernel::InternalStrideD;

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
  const int64_t problem_shape_size = group_count * ((int64_t)sizeof(typename ProblemShape::UnderlyingProblemShape));
  const int64_t stride_size = 3 * group_count * ((int64_t)sizeof(StrideA));
  // Dummy TMAs are built from these pointer arrays; CUDA < 12.4 needs them 128-bit aligned.
  const int aligned_group_count = round_up_to_nearest_multiple(group_count, 16 / sizeof(void*));
  int64_t input_args_size = aligned_group_count * 3 * sizeof(void*) + problem_shape_size + stride_size;

  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  auto input_buf = allocator.allocate(input_args_size);
  void* buf_ptr = input_buf.get();
  Dtype** inputA_ptrs = reinterpret_cast<Dtype**>(buf_ptr);
  Dtype** inputB_ptrs = inputA_ptrs + aligned_group_count;
  Dtype** output_ptrs = inputB_ptrs + aligned_group_count;
  static_assert(sizeof(StrideA) == 8, "expected StrideA to be 8 bytes for alignment");
  StrideA* stride_A = reinterpret_cast<StrideA*>(output_ptrs + aligned_group_count);
  StrideB* stride_B = reinterpret_cast<StrideB*>(stride_A + group_count);
  StrideOutput* stride_output = reinterpret_cast<StrideOutput*>(stride_B + group_count);
  auto* problem_sizes =
      reinterpret_cast<typename ProblemShape::UnderlyingProblemShape*>(stride_output + group_count);

  auto stream = at::cuda::getCurrentCUDAStream();
  auto make_strides = [](at::IntArrayRef values) {
    Strides result{};
    std::copy(values.begin(), values.end(), result.begin());
    return result;
  };
  prepare_grouped_gemm_data<<<1, group_count, 0, stream.stream()>>>(
      reinterpret_cast<Dtype*>(mat_a.data_ptr()),
      reinterpret_cast<Dtype*>(mat_b.data_ptr()),
      reinterpret_cast<Dtype*>(out.data_ptr()),
      inputA_ptrs,
      inputB_ptrs,
      output_ptrs,
      problem_sizes,
      stride_A,
      stride_B,
      stride_output,
      offs.const_data_ptr<int32_t>(),
      M,
      N,
      K,
      make_strides(mat_a.strides()),
      make_strides(mat_b.strides()),
      make_strides(out.strides()),
      make_strides(mat_a.sizes()),
      make_strides(mat_b.sizes()),
      a_row_major,
      b_row_major);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {group_count, problem_sizes, nullptr},
      {(const Dtype**)inputA_ptrs, stride_A, (const Dtype**)inputB_ptrs, stride_B},
      {{}, nullptr, stride_output, output_ptrs, stride_output}};
  arguments.epilogue.thread.alpha = 1.0;
  arguments.epilogue.thread.dAlpha = {cute::_0{}, cute::_0{}, 0};
  arguments.hw_info.sm_count = at::cuda::getDeviceProperties(out.device().index())->multiProcessorCount;

  auto workspace = allocator.allocate(Gemm::get_workspace_size(arguments));
  Gemm gemm;
  TORCH_CHECK(gemm.can_implement(arguments) == cutlass::Status::kSuccess, "cutlass cannot implement");
  TORCH_CHECK(gemm.initialize(arguments, workspace.get()) == cutlass::Status::kSuccess, "cutlass cannot initialize");
  auto status = gemm(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "cutlass cannot run, error ", int(status));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Same tile choice as torch: it depends only on N and the average rows per group, never on one group's size.
template <typename Dtype, bool a_row_major, bool b_row_major>
void dispatch_on_tile_size(const at::Tensor& mat_a, const at::Tensor& mat_b, const at::Tensor& offs, at::Tensor& out) {
  int32_t M = mat_a.size(-2);
  int32_t N = mat_b.size(-1);
  int32_t group_count = offs.size(0);
  if (mat_a.dim() == 2 && mat_b.dim() != 2) {
    M = M / group_count;
  } else if (mat_a.dim() != 2 && mat_b.dim() == 2) {
    N = N / group_count;
  }
  bool small = (M <= 128 || N <= 128);
  if (at::cuda::getCurrentDeviceProperties()->major == 10) {
    if (small) {
      grouped_gemm_impl<Dtype, cutlass::arch::Sm100, a_row_major, b_row_major, false, cute::_128, cute::_256, cute::_64>(
          mat_a, mat_b, offs, out);
    } else {
      grouped_gemm_impl<Dtype, cutlass::arch::Sm100, a_row_major, b_row_major, true, cute::_256, cute::_256, cute::_64>(
          mat_a, mat_b, offs, out);
    }
  } else {
    if (small) {
      grouped_gemm_impl<Dtype, cutlass::arch::Sm90, a_row_major, b_row_major, true, cute::_64, cute::_128, cute::_128>(
          mat_a, mat_b, offs, out);
    } else {
      grouped_gemm_impl<Dtype, cutlass::arch::Sm90, a_row_major, b_row_major, false, cute::_128, cute::_256, cute::_64>(
          mat_a, mat_b, offs, out);
    }
  }
}

template <typename Dtype>
void dispatch_on_layout(const at::Tensor& mat_a, const at::Tensor& mat_b, const at::Tensor& offs, at::Tensor& out) {
  bool a_row_major = mat_a.stride(-1) == 1;
  bool b_row_major = mat_b.stride(-1) == 1;
  if (a_row_major && b_row_major) {
    dispatch_on_tile_size<Dtype, true, true>(mat_a, mat_b, offs, out);
  } else if (a_row_major) {
    dispatch_on_tile_size<Dtype, true, false>(mat_a, mat_b, offs, out);
  } else if (b_row_major) {
    dispatch_on_tile_size<Dtype, false, true>(mat_a, mat_b, offs, out);
  } else {
    dispatch_on_tile_size<Dtype, false, false>(mat_a, mat_b, offs, out);
  }
}

void grouped_mm_bf16(const at::Tensor& mat_a, const at::Tensor& mat_b, const at::Tensor& offs, at::Tensor& out);
void grouped_mm_fp16(const at::Tensor& mat_a, const at::Tensor& mat_b, const at::Tensor& offs, at::Tensor& out);

} // namespace miles_grouped_gemm
