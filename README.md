[![Build and Test](https://github.com/spcl/mlir-dace/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/spcl/mlir-dace/actions/workflows/build_and_test.yml)

# MLIR-DaCe
Development repository for the Data-Centric MLIR dialect.

## Building

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run
```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build . --target check-sdir
```
To build the documentation from the TableGen description of the dialect operations, run
```sh
cmake --build . --target mlir-doc
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with CMake in order to install `FileCheck` to the chosen installation prefix.