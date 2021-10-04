#include "SDIR/Translate/Translation.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Translation.h"

//===----------------------------------------------------------------------===//
// SDFG registration
//===----------------------------------------------------------------------===//

void registerToSDFGTranslation() {
  mlir::TranslateFromMLIRRegistration registration(
      "mlir-to-sdfg",
      [](mlir::ModuleOp module, llvm::raw_ostream &output) {
        JsonEmitter jemit(output);

        if (translateModuleToSDFG(module, jemit).failed())
          return mlir::failure();

        if (jemit.finish())
          return mlir::failure();

        return mlir::success();
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<mlir::sdir::SDIRDialect>();
        registry.insert<mlir::StandardOpsDialect>();
      });
}
