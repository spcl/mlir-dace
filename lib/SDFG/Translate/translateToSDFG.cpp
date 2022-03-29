#include "SDFG/Translate/Node.h"
#include "SDFG/Translate/Translation.h"
#include "SDFG/Translate/liftToPython.h"
#include "SDFG/Utils/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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
  sdfg.setName(utils::generateName("sdfg"));

  for (StateNode stateNode : sdfgNode.getOps<StateNode>()) {
    if (collect(stateNode, sdfg).failed())
      return failure();
  }

  if (sdfgNode.entry().hasValue()) {
    StateNode entryState =
        sdfgNode.getStateBySymRef(sdfgNode.entry().getValue());
    sdfg.setStartState(sdfg.lookup(entryState.ID()));
  } else {
    sdfg.setStartState(sdfg.lookup(sdfgNode.getFirstState().ID()));
  }

  for (EdgeOp edgeOp : sdfgNode.getOps<EdgeOp>()) {
    if (collect(edgeOp, sdfg).failed())
      return failure();
  }

  sdfg.emit(jemit);
  return success();
}

//===----------------------------------------------------------------------===//
// State
//===----------------------------------------------------------------------===//

LogicalResult translation::collect(StateNode &op, SDFG &sdfg) {
  State state(op.getLoc());
  sdfg.addState(state, op.ID());

  state.setName(op.getName());

  for (TaskletNode taskletNode : op.getOps<TaskletNode>()) {
    if (collect(taskletNode, state).failed())
      return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// EdgeOp
//===----------------------------------------------------------------------===//

LogicalResult translation::collect(EdgeOp &op, SDFG &sdfg) {
  SDFGNode sdfgNode = utils::getParentSDFG(*op);
  StateNode srcNode = sdfgNode.getStateBySymRef(op.src());
  StateNode destNode = sdfgNode.getStateBySymRef(op.dest());

  State src = sdfg.lookup(srcNode.ID());
  State dest = sdfg.lookup(destNode.ID());

  InterstateEdge edge(src, dest);
  sdfg.addEdge(edge);

  edge.setCondition(op.condition());

  for (mlir::Attribute attr : op.assign()) {
    std::pair<StringRef, StringRef> kv =
        attr.cast<StringAttr>().getValue().split(':');

    edge.addAssignment(Assignment(kv.first.trim(), kv.second.trim()));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TaskletNode
//===----------------------------------------------------------------------===//

LogicalResult translation::collect(TaskletNode &op, State &state) {
  Tasklet tasklet(op.getLoc());
  state.addNode(tasklet, op.ID());
  tasklet.setName(getTaskletName(op));

  for (unsigned i = 0; i < op.getNumOperands(); ++i) {
    Connector connector(tasklet, op.getInputName(i));
    tasklet.addInConnector(connector);

    MultiEdge edge(state.lookup(op.getOperand(i)), connector);
    state.addEdge(edge);
  }

  for (unsigned i = 0; i < op.getNumResults(); ++i) {
    Connector connector(tasklet, op.getOutputName(i));
    tasklet.addOutConnector(connector);

    // TODO: Refactor
    Array array(utils::generateName("tmp"), true, "int32");
    static_cast<SDFG>(state.getParent()).addArray(array);

    Access access(op.getLoc());
    access.setName(array.name);
    state.addNode(access);

    Connector accIn(access, "__in");
    Connector accOut(access, "__out");

    access.addInConnector(accIn);
    access.addOutConnector(accOut);

    MultiEdge edge(connector, accIn);
    state.addEdge(edge);

    state.mapConnector(op.getResult(i), accOut);
  }

  Optional<std::string> code = liftToPython(op);
  if (code.hasValue()) {
    tasklet.setCode(code.getValue());
    tasklet.setLanguage("Python");
  } else {
    // TODO: Write content as code
  }

  return success();
}
