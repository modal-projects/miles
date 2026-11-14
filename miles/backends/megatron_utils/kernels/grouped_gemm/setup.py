import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# The kernel uses sm90a/sm100a-only instructions (wgmma, tcgen05), so no other architecture is built.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a;10.0a;10.3a")
cutlass = Path(os.environ["CUTLASS_DIR"])

setup(
    name="miles_grouped_gemm",
    ext_modules=[
        CUDAExtension(
            name="miles_grouped_gemm",
            sources=["grouped_gemm.cpp", "grouped_gemm_bf16.cu", "grouped_gemm_fp16.cu"],
            include_dirs=[str(cutlass / "include"), str(cutlass / "tools" / "util" / "include")],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-DNDEBUG"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
