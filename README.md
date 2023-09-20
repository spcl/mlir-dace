[![Tests](https://github.com/spcl/mlir-dace/actions/workflows/lit-test.yml/badge.svg)](https://github.com/spcl/mlir-dace/actions/workflows/lit-test.yml)

# MLIR-DaCe
MLIR-DaCe is a project aiming to bridge the gap between control-centric and data-centric intermediate representations.
By bridging these two groups of IRs, it allows the combination of control-centric and data-centric optimizations in optimization pipelines.
In order to achieve this, MLIR-DaCe provides a data-centric dialect in MLIR to connect the MLIR and DaCe frameworks.

## Building MLIR
If you have already MLIR built you can skip to [Building MLIR-DaCe](#building-mlir-dace). 
Keep in mind that there are no guarantees for commits other than the one in the submodule.

First clone with submodules:
```sh
git clone --recurse-submodules https://github.com/spcl/mlir-dace
cd mlir-dace
```

Then build MLIR with:
```sh
cd llvm-project
mkdir build && cd build
cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS="mlir" -DLLVM_TARGETS_TO_BUILD="host" -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON -DLLVM_INSTALL_UTILS=ON
```

## Building MLIR-DaCe
This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. If you built using the directions in [Building MLIR](#building-mlir), replace both with `llvm-project/build/`.
To build and launch the tests, run
```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build . --target check-sdfg-opt
```
To build the documentation from the TableGen description of the dialect operations, run
```sh
cmake --build . --target mlir-doc
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with CMake in order to install `FileCheck` to the chosen installation prefix.

## Publication
The paper on DCIR can be found [here](https://dl.acm.org/doi/10.1145/3579990.3580018).

If you use MLIR-DaCe, cite us:
```bibtex
@inproceedings{mlir-dace,
author = {Ben-Nun, Tal and Ates, Berke and Calotoiu, Alexandru and Hoefler, Torsten},
title = {Bridging Control-Centric and Data-Centric Optimization},
year = {2023},
booktitle = {Proceedings of the 21st ACM/IEEE International Symposium on Code Generation and Optimization},
series = {CGO 2023}
}
```
