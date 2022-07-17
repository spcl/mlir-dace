[![LIT Test](https://github.com/spcl/mlir-dace/actions/workflows/lit-test.yml/badge.svg)](https://github.com/spcl/mlir-dace/actions/workflows/lit-test.yml)
[![Parse Test](https://github.com/spcl/mlir-dace/actions/workflows/parse-test.yml/badge.svg)](https://github.com/spcl/mlir-dace/actions/workflows/parse-test.yml)
[![Translation Test](https://github.com/spcl/mlir-dace/actions/workflows/translation-test.yml/badge.svg)](https://github.com/spcl/mlir-dace/actions/workflows/translation-test.yml)

# MLIR-DaCe
MLIR-DaCe is a project aiming to provide a data-centric dialect in MLIR in order to combine control-centric and data-centric optimizations using the MLIR and DaCe frameworks.

## Building
This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run
```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build . --target check-sdir-opt
```
To build the documentation from the TableGen description of the dialect operations, run
```sh
cmake --build . --target mlir-doc
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with CMake in order to install `FileCheck` to the chosen installation prefix.
