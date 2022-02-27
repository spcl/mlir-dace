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

  if (++op.getOps<SDFGNode>().begin() != op.getOps<SDFGNode>().end()) {
    emitError(op.getLoc(), "Must have exactly one top-level SDFGNode");
    return failure();
  }

  SDFGNode sdfgNode = *op.getOps<SDFGNode>().begin();
  SDFG sdfg(sdfgNode.getLoc());

  for (StateNode stateNode : sdfgNode.getOps<StateNode>()) {
    State state(stateNode.getLoc());
    sdfg.addNode(state);
  }

  for (EdgeOp edgeOp : sdfgNode.getOps<EdgeOp>()) {
    // InterstateEdge edge();
  }

  sdfg.emit(jemit);
  return success();
}
