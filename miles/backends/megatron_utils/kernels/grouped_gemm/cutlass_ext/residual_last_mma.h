#pragma once

#include <cutlass/arch/arch.h>
#include <cutlass/gemm/threadblock/mma_multistage.h>
#include <cutlass/transform/threadblock/predicated_tile_access_iterator.h>
#include <cutlass/transform/threadblock/predicated_tile_iterator.h>

#include "make_residual_last.h"

namespace miles_grouped_gemm {

// CUTLASS 2.x iterators load the partial K tile first, so the tile boundaries depend on K mod the tile size. This
// iterator loads whole tiles from the start of K and masks only the last one, so zero rows appended to K just fill
// slots that were masked zeros and leave every sum bitwise unchanged.
template <typename Base, bool kAlongColumn>
class LastTileIterator : public cutlass::transform::threadblock::MakeIteratorResidualLast<Base>::Iterator {
  using Super = typename cutlass::transform::threadblock::MakeIteratorResidualLast<Base>::Iterator;
  int tiles_left_;

 public:
  CUTLASS_HOST_DEVICE
  LastTileIterator(
      typename Super::Params const& params,
      typename Super::Pointer pointer,
      typename Super::TensorCoord extent,
      int thread_id,
      typename Super::TensorCoord const& threadblock_offset)
      : Super(params, pointer, extent, thread_id, threadblock_offset) {
    constexpr int kTile = kAlongColumn ? Super::Shape::kColumn : Super::Shape::kRow;
    const int k = kAlongColumn ? extent.column() : extent.row();
    tiles_left_ = (k + kTile - 1) / kTile - 1;
    if (tiles_left_ == 0) {
      this->set_residual_tile(true);
    }
  }

  CUTLASS_HOST_DEVICE
  void add_tile_offset(typename Super::TensorCoord const& tile_offset) {
    Super::add_tile_offset(tile_offset);
    if (--tiles_left_ == 0) {
      this->set_residual_tile(true);
    }
  }
};

template <typename Mma>
struct WithLastTileResidue;

template <
    typename Shape,
    typename IteratorA,
    typename SmemIteratorA,
    cutlass::arch::CacheOperation::Kind CacheOpA,
    typename IteratorB,
    typename SmemIteratorB,
    cutlass::arch::CacheOperation::Kind CacheOpB,
    typename ElementC,
    typename LayoutC,
    typename Policy,
    int Stages,
    cutlass::gemm::SharedMemoryClearOption SharedMemoryClear,
    typename Enable>
struct WithLastTileResidue<cutlass::gemm::threadblock::MmaMultistage<
    Shape,
    IteratorA,
    SmemIteratorA,
    CacheOpA,
    IteratorB,
    SmemIteratorB,
    CacheOpB,
    ElementC,
    LayoutC,
    Policy,
    Stages,
    SharedMemoryClear,
    Enable>> {
  using type = cutlass::gemm::threadblock::MmaMultistage<
      Shape,
      LastTileIterator<IteratorA, true>,
      SmemIteratorA,
      CacheOpA,
      LastTileIterator<IteratorB, false>,
      SmemIteratorB,
      CacheOpB,
      ElementC,
      LayoutC,
      Policy,
      Stages,
      // The mainloop now loads the masked tile, and only zfill writes zeros for masked accesses there.
      cutlass::gemm::SharedMemoryClearOption::kZfill,
      Enable>;
};

} // namespace miles_grouped_gemm
