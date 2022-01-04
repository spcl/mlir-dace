#ifndef SDIR_Conversion_SAMToSDIR
#define SDIR_Conversion_SAMToSDIR

include "mlir/Pass/PassBase.td"
include "SDIR/Dialect/Dialect.td"

def SAMToSDIRPass : Pass<"convert-to-sdir", "ModuleOp"> {
  let summary = "Convert SCF, Arith, Memref dialect to SDIR dialect";
  let constructor = "mlir::sdir::conversion::createSAMToSDIRPass()";

  // TODO: Add dialects
  //let dependentDialects = [""];
}

#endif // SDIR_Conversion_ToSDIR
