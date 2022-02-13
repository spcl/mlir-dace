set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name GenericToSDFG)
add_public_tablegen_target(MLIRGenericToSDFGPassIncGen)
