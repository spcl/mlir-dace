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

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

void insertTransientArray(Location location, translation::Connector connector,
                          Value value, translation::State &state) {
  using namespace translation;

  Array array(utils::generateName("tmp"), true, value.getType());

  if (utils::isSizedType(value.getType()))
    array = Array(utils::generateName("tmp"), true,
                  utils::getSizedType(value.getType()));

  state.getSDFG().addArray(array);

  Access access(location);
  access.setName(array.name);
  state.addNode(access);

  Connector accIn(access);
  Connector accOut(access);

  access.addInConnector(accIn);
  access.addOutConnector(accOut);

  MultiEdge edge(connector, accIn);
  state.addEdge(edge);

  state.mapConnector(value, accOut);
}

//===----------------------------------------------------------------------===//
// Module
//===----------------------------------------------------------------------===//

LogicalResult translation::translateToSDFG(ModuleOp &op, JsonEmitter &jemit) {
  if (++op.getOps<SDFGNode>().begin() != op.getOps<SDFGNode>().end()) {
    emitError(op.getLoc(), "Must have exactly one top-level SDFGNode");
    return failure();
  }

  SDFGNode sdfgNode = *op.getOps<SDFGNode>().begin();
  SDFG sdfg(sdfgNode.getLoc());

  if (collect(sdfg, sdfgNode).failed())
    return failure();

  sdfg.emit(jemit);
  return success();
}

//===----------------------------------------------------------------------===//
// SDFG
//===----------------------------------------------------------------------===//

LogicalResult translation::collect(SDFG &sdfg, SDFGNode &sdfgNode) {
  sdfg.setName(utils::generateName("sdfg"));

  for (BlockArgument ba : sdfgNode.body().getArguments()) {
    if (utils::isSizedType(ba.getType())) {
      Array array(utils::valueToString(ba), false,
                  utils::getSizedType(ba.getType()));
      sdfg.addArg(array);
    } else {
      Array array(utils::valueToString(ba), false, ba.getType());
      sdfg.addArg(array);
    }
  }

  for (AllocOp allocOp : sdfgNode.getOps<AllocOp>()) {
    if (collect(allocOp, sdfg).failed())
      return failure();
  }

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

  return success();
}

//===----------------------------------------------------------------------===//
// State
//===----------------------------------------------------------------------===//

LogicalResult translation::collect(StateNode &op, SDFG &sdfg) {
  State state(op.getLoc());
  sdfg.addState(state, op.ID());

  state.setName(op.getName());

  for (Operation &operation : op.getOps()) {
    if (TaskletNode oper = dyn_cast<TaskletNode>(operation))
      if (collect(oper, state).failed())
        return failure();

    if (NestedSDFGNode oper = dyn_cast<NestedSDFGNode>(operation))
      if (collect(oper, state).failed())
        return failure();

    if (MapNode oper = dyn_cast<MapNode>(operation))
      if (collect(oper, state).failed())
        return failure();

    if (CopyOp oper = dyn_cast<CopyOp>(operation))
      if (collect(oper, state).failed())
        return failure();

    if (StoreOp oper = dyn_cast<StoreOp>(operation))
      if (collect(oper, state).failed())
        return failure();

    if (LoadOp oper = dyn_cast<LoadOp>(operation))
      if (collect(oper, state).failed())
        return failure();

    if (AllocOp oper = dyn_cast<AllocOp>(operation))
      if (collect(oper, state).failed())
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
// AllocOp
//===----------------------------------------------------------------------===//

LogicalResult translation::collect(AllocOp &op, SDFG &sdfg) {
  Array array(op.getName(), op.transient(), utils::getSizedType(op.getType()));
  sdfg.addArray(array);

  return success();
}

LogicalResult translation::collect(AllocOp &op, State &state) {
  Array array(op.getName(), op.transient(), utils::getSizedType(op.getType()));
  state.getSDFG().addArray(array);

  return success();
}

//===----------------------------------------------------------------------===//
// TaskletNode
//===----------------------------------------------------------------------===//

LogicalResult translation::collect(TaskletNode &op, State &state) {
  Tasklet tasklet(op.getLoc());
  state.addNode(tasklet);
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

    insertTransientArray(op.getLoc(), connector, op.getResult(i), state);
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

//===----------------------------------------------------------------------===//
// NestedSDFGNode
//===----------------------------------------------------------------------===//

LogicalResult translation::collect(NestedSDFGNode &op, State &state) {
  SDFG sdfg(op.getLoc());
  sdfg.setName(utils::generateName("sdfg"));

  for (BlockArgument ba : op.body().getArguments()) {
    if (utils::isSizedType(ba.getType())) {
      Array array(utils::valueToString(ba), false,
                  utils::getSizedType(ba.getType()));
      sdfg.addArg(array);
    } else {
      Array array(utils::valueToString(ba), false, ba.getType());
      sdfg.addArg(array);
    }
  }

  for (AllocOp allocOp : op.getOps<AllocOp>()) {
    if (collect(allocOp, sdfg).failed())
      return failure();
  }

  for (StateNode stateNode : op.getOps<StateNode>()) {
    if (collect(stateNode, sdfg).failed())
      return failure();
  }

  if (op.entry().hasValue()) {
    StateNode entryState = op.getStateBySymRef(op.entry().getValue());
    sdfg.setStartState(sdfg.lookup(entryState.ID()));
  } else {
    sdfg.setStartState(sdfg.lookup(op.getFirstState().ID()));
  }

  for (EdgeOp edgeOp : op.getOps<EdgeOp>()) {
    if (collect(edgeOp, sdfg).failed())
      return failure();
  }

  NestedSDFG nestedSDFG(op.getLoc(), sdfg);
  nestedSDFG.setName(utils::generateName("nested_sdfg"));
  state.addNode(nestedSDFG);

  for (unsigned i = 0; i < op.num_args(); ++i) {
    Connector connector(
        nestedSDFG,
        utils::valueToString(op.body().getArgument(i), *op.getOperation()));
    nestedSDFG.addInConnector(connector);

    MultiEdge edge(state.lookup(op.getOperand(i)), connector);
    state.addEdge(edge);
  }

  for (unsigned i = op.num_args(); i < op.getNumOperands(); ++i) {
    Connector connector(
        nestedSDFG,
        utils::valueToString(op.body().getArgument(i), *op.getOperation()));
    nestedSDFG.addOutConnector(connector);

    Access access(op.getLoc());
    access.setName(utils::valueToString(op.getOperand(i)));
    state.addNode(access);

    Connector accOut(access);
    access.addOutConnector(accOut);

    MultiEdge edge(connector, accOut);
    state.addEdge(edge);

    state.mapConnector(op.getOperand(i), accOut);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MapNode
//===----------------------------------------------------------------------===//

LogicalResult translation::collect(MapNode &op, State &state) {
  return success();
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

LogicalResult translation::collect(CopyOp &op, State &state) {
  Access access(op.getLoc());
  access.setName(utils::valueToString(op.dest()));
  state.addNode(access);

  Connector accOut(access);
  access.addOutConnector(accOut);

  MultiEdge edge(state.lookup(op.src()), accOut);
  state.addEdge(edge);

  state.mapConnector(op.dest(), accOut);

  return success();
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

LogicalResult translation::collect(StoreOp &op, State &state) {
  Access access(op.getLoc());
  access.setName(utils::valueToString(op.arr()));
  state.addNode(access);

  Connector accOut(access);
  access.addOutConnector(accOut);

  MultiEdge edge(state.lookup(op.val()), accOut);
  state.addEdge(edge);

  state.mapConnector(op.arr(), accOut);

  return success();
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

LogicalResult translation::collect(LoadOp &op, State &state) {
  Connector connector = state.lookup(op.arr());
  state.mapConnector(op.res(), connector);

  return success();
}
