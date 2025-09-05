"""
HoME CUDA扩展编译脚本
"""

import os
import torch
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
from setuptools import setup

# 获取CUDA路径
def get_cuda_paths():
    cuda_home = "/usr/local/cuda-12.8"
    if cuda_home is None:
        raise RuntimeError("CUDA_HOME or CUDA_PATH environment variable not set")
    
    cuda_include = os.path.join(cuda_home, 'include')
    cuda_lib = os.path.join(cuda_home, 'lib64')
    
    if not os.path.exists(cuda_include):
        raise RuntimeError(f"CUDA include directory not found: {cuda_include}")
    
    return cuda_home, cuda_include, cuda_lib

# 获取CUTLASS路径
def get_cutlass_paths():
    cutlass_home = os.environ.get('CUTLASS_HOME')
    if cutlass_home is None:
        # 尝试从当前目录查找
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cutlass_home = os.path.join(current_dir, 'third_party', 'cutlass')
    
    cutlass_include = os.path.join(cutlass_home, 'include') if os.path.exists(cutlass_home) else None
    cutlass_extensions_include = os.path.join(cutlass_home, '..', 'cutlass_extensions', 'include') if os.path.exists(cutlass_home) else None
    
    return cutlass_home, cutlass_include, cutlass_extensions_include

def main():
    # 获取路径
    cuda_home, cuda_include, cuda_lib = get_cuda_paths()
    cutlass_home, cutlass_include, cutlass_extensions_include = get_cutlass_paths()

    torch_lib_path = os.path.dirname(torch.__file__)
    torch_include_path = os.path.join(torch_lib_path, 'include')
    torch_lib_dir = os.path.join(torch_lib_path, 'lib')
    
    # 设置编译选项
    extra_compile_args = {
        'cxx': ['-std=c++17', '-O3', '-fPIC'],
        'nvcc': [
            '-std=c++17',
            '-O3',
            '--use_fast_math',
            '-Xptxas=-O3',
            '--expt-relaxed-constexpr',
            '--expt-extended-lambda',
            '-gencode', 'arch=compute_70,code=sm_70',
            '-gencode', 'arch=compute_75,code=sm_75', 
            '-gencode', 'arch=compute_80,code=sm_80',
            '-gencode', 'arch=compute_86,code=sm_86',
        ]
    }
    
    # 设置包含目录
    include_dirs = [cuda_include, torch_include_path]
    
    # 添加CUTLASS路径（如果存在）
    if cutlass_include:
        include_dirs.append(cutlass_include)
    if cutlass_extensions_include:
        include_dirs.append(cutlass_extensions_include)
    
    # 设置库目录
    library_dirs = [cuda_lib, torch_lib_dir]
    
    # 设置链接库
    libraries = ['cudart', 'cublas', 'curand']
    
    # 创建CUDA扩展
    ext_modules = [
        CUDAExtension(
            name='home_kernels',
            sources=['csrc/home_kernel.cu'],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args=extra_compile_args,
            define_macros=[
                ('TORCH_EXTENSION_NAME', 'home_kernels'),
            ]
        )
    ]
    
    # 设置编译
    setup(
        name='home_kernels',
        ext_modules=ext_modules,
        cmdclass={'build_ext': BuildExtension},
        python_requires='>=3.7',
    )

if __name__ == '__main__':
    main()
