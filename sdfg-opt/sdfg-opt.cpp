// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "SDFG/Conversion/GenericToSDFG/Passes.h"
#include "SDFG/Conversion/LinalgToSDFG/Passes.h"
#include "SDFG/Conversion/SDFGToGeneric/Passes.h"
#include "SDFG/Dialect/Dialect.h"

int main(int argc, char **argv) {
  // Register SDFG passes
  mlir::sdfg::conversion::registerGenericToSDFGPasses();
  mlir::sdfg::conversion::registerLinalgToSDFGPasses();
  mlir::sdfg::conversion::registerSDFGToGenericPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::sdfg::SDFGDialect>();
  //  Add the following to include *all* MLIR Core dialects, or selectively
  //  include what you need like above. You only need to register dialects that
  //  will be *parsed* by the tool, not the one generated
  registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "SDFG optimizer driver\n", registry));
}
