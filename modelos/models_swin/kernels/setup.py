from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name='swin_window_process',
    ext_modules=[
        CUDAExtension('swin_window_process', [
            'swin_window_process.cpp',
            'swin_window_process_kernel.cu',
        ],
        extra_compile_args={
            'cxx': ['-O3', '-Wall'],
            'nvcc': ['-O3', '--expt-relaxed-constexpr']
        },
        extra_link_args=[
            '-L/mnt/beegfs/home/hugo.carreira/miniconda3/envs/fastvit/lib',
            '-lcudart'
        ])
    ],
    cmdclass={'build_ext': BuildExtension}
)
