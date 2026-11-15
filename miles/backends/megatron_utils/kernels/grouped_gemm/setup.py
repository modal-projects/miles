import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# sm90a/sm100a kernels use wgmma/tcgen05; the 8.0 and 12.0 builds run the mma.sync kernel (sm80-sm89, sm120).
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;9.0a;10.0a;10.3a;12.0")
cutlass = Path(os.environ["CUTLASS_DIR"])

setup(
    name="miles_grouped_gemm",
    ext_modules=[
        CUDAExtension(
            name="miles_grouped_gemm",
            sources=[
                "grouped_gemm.cpp",
                "grouped_gemm_bf16.cu",
                "grouped_gemm_fp16.cu",
                "grouped_gemm_fp32.cu",
                "grouped_gemm_sm80.cu",
            ],
            include_dirs=[str(cutlass / "include"), str(cutlass / "tools" / "util" / "include")],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-DNDEBUG"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
