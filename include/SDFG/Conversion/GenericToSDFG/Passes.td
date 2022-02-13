#ifndef SDFG_Conversion_GenericToSDFG
#define SDFG_Conversion_GenericToSDFG

include "mlir/Pass/PassBase.td"
include "SDIR/Dialect/Dialect.td"

def GenericToSDFGPass : Pass<"convert-to-sdfg", "ModuleOp"> {
  let summary = "Convert SCF, Arith and Memref dialect to SDFG dialect";
  let constructor = "mlir::sdfg::conversion::createGenericToSDFGPass()";

  // TODO: Add dialects
  let dependentDialects = ["mlir::sdfg::SDFGDialect"];
}

#endif // SDFG_Conversion_GenericToSDFG
