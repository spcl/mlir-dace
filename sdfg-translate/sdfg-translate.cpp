// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

/// This file contains the SDFG translator with the translation passes.

#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "llvm/IR/LLVMContext.h"

#include "SDFG/Translate/Translation.h"

int main(int argc, char **argv) {
  mlir::registerAllTranslations();
  mlir::sdfg::translation::registerToSDFGTranslation();

  return mlir::failed(
      mlir::mlirTranslateMain(argc, argv, "MLIR-DaCe Translation Tool"));
}
