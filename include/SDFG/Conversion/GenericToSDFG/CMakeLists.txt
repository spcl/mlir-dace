# Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name GenericToSDFG)
add_public_tablegen_target(MLIRGenericToSDFGPassIncGen)

target_sources(SOURCE_FILES_H PRIVATE Passes.h)
