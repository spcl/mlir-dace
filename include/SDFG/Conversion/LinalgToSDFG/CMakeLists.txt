set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name LinalgToSDFG)
add_public_tablegen_target(MLIRLinalgToSDFGPassIncGen)

target_sources(SOURCE_FILES_H PRIVATE PassDetail.h Passes.h)
