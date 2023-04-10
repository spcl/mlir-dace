#ifndef SDFG_Conversion_SDFGToGeneric
#define SDFG_Conversion_SDFGToGeneric

include "mlir/Pass/PassBase.td"
include "SDFG/Dialect/Dialect.td"

def SDFGToGenericPass : Pass<"lower-sdfg", "ModuleOp"> {
  let summary = "Convert SDFG dialect to CF, Arith, Math and Memref dialects";
  let constructor = "mlir::sdfg::conversion::createSDFGToGenericPass()";
  let dependentDialects = ["mlir::sdfg::SDFGDialect"];
}

#endif // SDFG_Conversion_SDFGToGeneric
