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
  if (++op.getOps<SDFGNode>().begin() != op.getOps<SDFGNode>().end()) {
    emitError(op.getLoc(), "Must have exactly one top-level SDFGNode");
    return failure();
  }

  SDFGNode sdfgNode = *op.getOps<SDFGNode>().begin();
  SDFG sdfg(sdfgNode.getLoc());

  for (StateNode stateNode : sdfgNode.getOps<StateNode>()) {
    State state(stateNode.getLoc());
    state->setLabel(stateNode.getName());

    sdfg.addState(stateNode.ID(), state);
  }

  for (EdgeOp edgeOp : sdfgNode.getOps<EdgeOp>()) {
    StateNode srcNode = sdfgNode.getStateBySymRef(edgeOp.src());
    StateNode destNode = sdfgNode.getStateBySymRef(edgeOp.dest());

    State src = sdfg.lookup(srcNode.ID());
    State dest = sdfg.lookup(destNode.ID());
    InterstateEdge iEdge(src, dest);

    //  TODO: Add Conditions/Assignments
    sdfg.addEdge(iEdge);
  }

  sdfg.emit(jemit);
  return success();
}
