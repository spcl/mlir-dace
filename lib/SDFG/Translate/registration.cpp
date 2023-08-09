// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

#include "SDFG/Translate/Translation.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-translate/Translation.h"

//===----------------------------------------------------------------------===//
// SDFG registration
//===----------------------------------------------------------------------===//

/// Registers SDFG to SDFG IR translation.
void mlir::sdfg::translation::registerToSDFGTranslation() {
  mlir::TranslateFromMLIRRegistration registration(
      "mlir-to-sdfg", "Generates a SDFG JSON",
      [](mlir::ModuleOp module, llvm::raw_ostream &output) {
        mlir::sdfg::emitter::JsonEmitter jemit(output);

        mlir::LogicalResult res =
            mlir::sdfg::translation::translateToSDFG(module, jemit);
        mlir::LogicalResult jRes = jemit.finish();

        if (res.failed()) {
          return mlir::failure();
        } else if (jRes.failed()) {
          emitError(module.getLoc(), "Invalid JSON generated");
          return mlir::failure();
        }

        return mlir::success();
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<mlir::sdfg::SDFGDialect>();
        registry.insert<mlir::func::FuncDialect>();
        registry.insert<mlir::arith::ArithDialect>();
        registry.insert<mlir::math::MathDialect>();
        registry.insert<mlir::LLVM::LLVMDialect>();
      });
}
