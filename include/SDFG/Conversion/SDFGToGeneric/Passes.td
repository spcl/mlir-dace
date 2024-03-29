// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

/// Table-driven file for SDFG to Generic conversion passes.

#ifndef SDFG_Conversion_SDFGToGeneric
#define SDFG_Conversion_SDFGToGeneric

include "mlir/Pass/PassBase.td"
include "SDFG/Dialect/Dialect.td"

/// Define SDFG to generic pass.
def SDFGToGenericPass : Pass<"lower-sdfg", "ModuleOp"> {
  let summary = "Convert SDFG dialect to Func, CF, Memref and SCF dialects";
  let constructor = "mlir::sdfg::conversion::createSDFGToGenericPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect"
  ];
}

#endif // SDFG_Conversion_SDFGToGeneric
