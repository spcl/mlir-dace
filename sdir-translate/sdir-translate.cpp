#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"
#include "llvm/IR/LLVMContext.h"

#include "SDIR/Translate/Translation.h"

int main(int argc, char **argv) {
  mlir::registerAllTranslations();
  registerToSDFGTranslation();

  return mlir::failed(mlir::mlirTranslateMain(argc, argv, "MLIR-DaCe Translation Tool"));
}
