#include "SDIR/Translate/Translation.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Translation.h"

//===----------------------------------------------------------------------===//
// SDFG registration
//===----------------------------------------------------------------------===//

void mlir::sdir::translation::registerToSDFGTranslation() {
  mlir::TranslateFromMLIRRegistration registration(
      "mlir-to-sdfg",
      [](mlir::ModuleOp module, llvm::raw_ostream &output) {
        mlir::sdir::emitter::JsonEmitter jemit(output);

        mlir::LogicalResult res =
            mlir::sdir::translation::translateToSDFG(module, jemit);
        mlir::LogicalResult jRes = jemit.finish();
        if (jRes.failed()) {
          emitError(module.getLoc(), "Invalid JSON generated");
          return mlir::failure();
        }

        if (res.failed())
          return mlir::failure();

        return mlir::success();
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<mlir::sdir::SDIRDialect>();
        registry.insert<mlir::StandardOpsDialect>();
        registry.insert<mlir::arith::ArithmeticDialect>();
      });
}
