#!/bin/bash
mamba activate laguerre_learning

# needed for TF protobuf / CUDA (otherwise you'd get errors like ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6:)
# Safe LD_LIBRARY_PATH handling:
conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
# conda env config vars set XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX

echo 'export LAGUERREENV='$CONDA_DEFAULT_ENV >>$HOME/.bashrc
echo 'export PYTHONPATH='$PWD'/../../':'$PYTHONPATH' >>$HOME/.bashrc
# to enable CUDA support -> re-activate env
mamba activate laguerre_learning
