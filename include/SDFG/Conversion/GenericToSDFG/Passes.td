// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

/// Table-driven file for Generic to SDFG conversion passes.

#ifndef SDFG_Conversion_GenericToSDFG
#define SDFG_Conversion_GenericToSDFG

include "mlir/Pass/PassBase.td"
include "SDFG/Dialect/Dialect.td"

/// Define generic to SDFG pass.
def GenericToSDFGPass : Pass<"convert-to-sdfg", "ModuleOp"> {
  let summary = "Convert SCF, Arith, Math and Memref dialect to SDFG dialect";
  let dependentDialects = ["mlir::sdfg::SDFGDialect"];
  let options = [
    Option<"mainFuncName", "main-func-name", "std::string", /*default=*/"",
           "Specify which func should be seen as the main func">
  ];
}

#endif // SDFG_Conversion_GenericToSDFG
