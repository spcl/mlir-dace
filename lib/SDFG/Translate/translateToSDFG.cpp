#include "SDFG/Translate/Node.h"
#include "SDFG/Translate/Translation.h"
#include "SDFG/Utils/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "llvm/ADT/DenseMap.h"
#include <regex>
#include <string>

using namespace mlir;
using namespace sdfg;

// Checks should be minimal
// A check might indicate that the IR is unsound

LogicalResult translation::translateToSDFG(ModuleOp &op, JsonEmitter &jemit) {
  utils::resetIDGenerator();

  SDFG tls(op.getLoc());
  // tls.emit(jemit);

  return success();
}
