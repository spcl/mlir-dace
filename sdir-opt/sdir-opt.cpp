#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"

#include "SDIR/SDIR_Dialect.h"

using namespace mlir;
using namespace sdir;

int main(int argc, char **argv) {
    registerAllPasses();
    // TODO: Register SDIR passes here.

    DialectRegistry registry;
    registry.insert<SDIRDialect>();
    registry.insert<StandardOpsDialect>();
    // Add the following to include *all* MLIR Core dialects, or selectively
    // include what you need like above. You only need to register dialects that
    // will be *parsed* by the tool, not the one generated
    // registerAllDialects(registry);

    return asMainReturnCode(
                MlirOptMain(argc, argv, "SDIR optimizer driver\n", registry));
}
