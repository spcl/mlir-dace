#include "SDFG/Translate/Translation.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Translation.h"

//===----------------------------------------------------------------------===//
// SDFG registration
//===----------------------------------------------------------------------===//

void mlir::sdfg::translation::registerToSDFGTranslation() {
  mlir::TranslateFromMLIRRegistration registration(
      "mlir-to-sdfg",
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
        registry.insert<mlir::StandardOpsDialect>();
        registry.insert<mlir::arith::ArithmeticDialect>();
      });
}
