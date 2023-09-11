import os
import torch
from pathlib import Path
from setuptools import setup, find_packages
from distutils.sysconfig import get_python_lib
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, CUDAExtension

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

USE_GEMV = os.environ.get("USE_GEMV", "0") == "1"
print('USE_GEMV', USE_GEMV)
common_setup_kwargs = {
    "version": "0.0.2",
    "name": "autoawq",
    "author": "Casper Hansen",
    "license": "MIT",
    "python_requires": ">=3.8.0",
    "description": "AutoAWQ implements the AWQ algorithm for 4-bit quantization with a 2x speedup during inference.",
    "long_description": (Path(__file__).parent / "README.md").read_text(encoding="UTF-8"),
    "long_description_content_type": "text/markdown",
    "url": "https://github.com/casper-hansen/AutoAWQ",
    "keywords": ["awq", "autoawq", "quantization", "transformers"],
    "platforms": ["linux", "windows"],
    "classifiers": [
        "Environment :: GPU :: NVIDIA CUDA :: 11.8",
        "Environment :: GPU :: NVIDIA CUDA :: 12",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
    ]
}

requirements = [
    "torch>=2.0.0",
    "transformers>=4.32.0",
    "tokenizers>=0.12.1",
    "accelerate",
    "sentencepiece",
    "lm_eval",
    "texttable",
    "toml",
    "attributedict",
    "protobuf",
    "torchvision"
]

include_dirs = []

conda_cuda_include_dir = os.path.join(get_python_lib(), "nvidia/cuda_runtime/include")
if os.path.isdir(conda_cuda_include_dir):
    include_dirs.append(conda_cuda_include_dir)

def check_dependencies():
    if CUDA_HOME is None:
        raise RuntimeError(
            f"Cannot find CUDA_HOME. CUDA must be available to build the package.")

def get_compute_capabilities():
    # Collect the compute capabilities of all available GPUs.
    compute_capabilities = set()
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        compute_capabilities.add(major * 10 + minor)

        if not USE_GEMV:
            """GEMM implementation requires compute capability 8.0 or higher."""
            if major < 8:
                raise RuntimeError("GPUs with compute capability less than 8.0 are not supported with GEMM kernel.")

    capability_flags = []
    for cap in compute_capabilities:
        capability_flags += ["-gencode", f"arch=compute_{cap},code=sm_{cap}"]

    return capability_flags

check_dependencies()
arch_flags = get_compute_capabilities()

if os.name == "nt":
    # Relaxed args on Windows
    extra_compile_args={
        "nvcc": arch_flags
    }
else:

    if USE_GEMV:
        # GEMV compilation flags
        extra_compile_args={
            "cxx": [        
                "-g", 
                "-O3", 
                "-fopenmp", 
                "-lgomp", 
                "-std=c++17",
                "-DENABLE_BF16"
            ],
            "nvcc": [        
                "-O3", 
                "-std=c++17",
                "-DENABLE_BF16",  # TODO
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
                "--threads=8"
            ] + arch_flags
        }
    
    else:
        # GEMM compilation flags
        extra_compile_args={
            "cxx": ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17"],
            "nvcc": ["-O3", "-std=c++17"] + arch_flags
        }


cpp_files = [
    "awq_cuda/layernorm/layernorm.cu",
    "awq_cuda/position_embedding/pos_encoding_kernels.cu",
]

if USE_GEMV:
    cpp_files = [
        "awq_cuda/pybind_gemv.cpp",
        "awq_cuda/quantization/gemv_cuda.cu"
    ] + cpp_files

else:
    cpp_files = [
        "awq_cuda/pybind_gemm.cpp",
        "awq_cuda/quantization/gemm_cuda_gen.cu"
    ] + cpp_files

extensions = [
    CUDAExtension(
        "awq_inference_engine",
        cpp_files, 
        extra_compile_args=extra_compile_args
    )
]

additional_setup_kwargs = {
    "ext_modules": extensions,
    "cmdclass": {'build_ext': BuildExtension}
}

common_setup_kwargs.update(additional_setup_kwargs)

setup(
    packages=find_packages(),
    install_requires=requirements,
    include_dirs=include_dirs,
    **common_setup_kwargs
)