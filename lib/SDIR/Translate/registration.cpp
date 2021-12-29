#include "SDIR/Translate/Translation.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Translation.h"

//===----------------------------------------------------------------------===//
// SDFG registration
//===----------------------------------------------------------------------===//

void registerToSDFGTranslation() {
  mlir::TranslateFromMLIRRegistration registration(
      "mlir-to-sdfg",
      [](mlir::ModuleOp module, llvm::raw_ostream &output) {
        JsonEmitter jemit(output);

        LogicalResult res = translateModuleToSDFG(module, jemit);
        LogicalResult jRes = jemit.finish();
        if (jRes.failed()) {
          emitError(module.getLoc(), "Invalid JSON generated");
          return failure();
        }

        if (res.failed())
          return failure();

        return success();
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<mlir::sdir::SDIRDialect>();
        registry.insert<mlir::StandardOpsDialect>();
        registry.insert<mlir::arith::ArithmeticDialect>();
      });
}
