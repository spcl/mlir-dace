[![LIT Test](https://github.com/spcl/mlir-dace/actions/workflows/lit-test.yml/badge.svg)](https://github.com/spcl/mlir-dace/actions/workflows/lit-test.yml)
[![Parse Test](https://github.com/spcl/mlir-dace/actions/workflows/parse-test.yml/badge.svg)](https://github.com/spcl/mlir-dace/actions/workflows/parse-test.yml)
[![Translation Test](https://github.com/spcl/mlir-dace/actions/workflows/translation-test.yml/badge.svg)](https://github.com/spcl/mlir-dace/actions/workflows/translation-test.yml)
[![Conversion Test](https://github.com/spcl/mlir-dace/actions/workflows/conversion-test.yml/badge.svg)](https://github.com/spcl/mlir-dace/actions/workflows/conversion-test.yml)

# MLIR-DaCe
Development repository for the Data-Centric MLIR dialect.

## Building

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run
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
