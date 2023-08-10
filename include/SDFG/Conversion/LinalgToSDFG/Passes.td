// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

/// Table-driven file for Linalg to SDFG conversion passes.

#ifndef SDFG_Conversion_LinalgToSDFG
#define SDFG_Conversion_LinalgToSDFG

include "mlir/Pass/PassBase.td"
include "SDFG/Dialect/Dialect.td"

/// Define Linalg to SDFG Pass.
def LinalgToSDFGPass : Pass<"linalg-to-sdfg", "ModuleOp"> {
  let summary = "Convert Linalg dialect to SDFG dialect";
  let constructor = "mlir::sdfg::conversion::createLinalgToSDFGPass()";
  let dependentDialects = ["mlir::sdfg::SDFGDialect"];
}

#endif // SDFG_Conversion_LinalgToSDFG
