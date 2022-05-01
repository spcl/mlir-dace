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
                          Value value, translation::ScopeNode &scope) {
  using namespace translation;

  Array array(utils::generateName("tmp"), true, value.getType());

  if (utils::isSizedType(value.getType()))
    array = Array(utils::generateName("tmp"), true,
                  utils::getSizedType(value.getType()));

  SDFG sdfg = scope.getSDFG();
  sdfg.addArray(array);

  Access access(location);
  access.setName(array.name);
  scope.addNode(access);

  Connector accIn(access);
  Connector accOut(access);

  access.addInConnector(accIn);
  access.addOutConnector(accOut);

  MultiEdge edge(connector, accIn);
  scope.addEdge(edge);

  scope.mapConnector(value, accOut);
}

LogicalResult collectOperations(Operation &op, translation::ScopeNode &scope) {
  using namespace translation;

  for (Operation &operation : op.getRegion(0).getOps()) {
    if (TaskletNode oper = dyn_cast<TaskletNode>(operation))
      if (collect(oper, scope).failed())
        return failure();

    if (NestedSDFGNode oper = dyn_cast<NestedSDFGNode>(operation))
      if (collect(oper, scope).failed())
        return failure();

    if (MapNode oper = dyn_cast<MapNode>(operation))
      if (collect(oper, scope).failed())
        return failure();

    if (ConsumeNode oper = dyn_cast<ConsumeNode>(operation))
      if (collect(oper, scope).failed())
        return failure();

    if (CopyOp oper = dyn_cast<CopyOp>(operation))
      if (collect(oper, scope).failed())
        return failure();

    if (StoreOp oper = dyn_cast<StoreOp>(operation))
      if (collect(oper, scope).failed())
        return failure();

    if (LoadOp oper = dyn_cast<LoadOp>(operation))
      if (collect(oper, scope).failed())
        return failure();

    if (AllocOp oper = dyn_cast<AllocOp>(operation))
      if (collect(oper, scope).failed())
        return failure();
  }

  return success();
}

LogicalResult collectSDFG(Operation &op, translation::SDFG &sdfg) {
  using namespace translation;

  sdfg.setName(utils::generateName("sdfg"));

  for (BlockArgument ba : op.getRegion(0).getArguments()) {
    if (utils::isSizedType(ba.getType())) {
      Array array(utils::valueToString(ba), false,
                  utils::getSizedType(ba.getType()));
      sdfg.addArg(array);
    } else {
      Array array(utils::valueToString(ba), false, ba.getType());
      sdfg.addArg(array);
    }
  }

  for (AllocOp allocOp : op.getRegion(0).getOps<AllocOp>()) {
    if (collect(allocOp, sdfg).failed())
      return failure();
  }

  for (StateNode stateNode : op.getRegion(0).getOps<StateNode>()) {
    if (collect(stateNode, sdfg).failed())
      return failure();
  }

  for (EdgeOp edgeOp : op.getRegion(0).getOps<EdgeOp>()) {
    if (collect(edgeOp, sdfg).failed())
      return failure();
  }

  if (op.hasAttr("entry")) {
    std::string entryName = utils::attributeToString(op.getAttr("entry"), op);
    entryName.erase(0, 1);
    sdfg.setStartState(sdfg.lookup(entryName));
  } else {
    StateNode stateNode = *op.getRegion(0).getOps<StateNode>().begin();
    StringRef entryName = stateNode.sym_name();
    sdfg.setStartState(sdfg.lookup(entryName));
  }

  return success();
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

  if (collectSDFG(*sdfgNode, sdfg).failed())
    return failure();

  sdfg.emit(jemit);
  return success();
}

//===----------------------------------------------------------------------===//
// State
//===----------------------------------------------------------------------===//

LogicalResult translation::collect(StateNode &op, SDFG &sdfg) {
  State state(op.getLoc());
  state.setName(op.getName());
  sdfg.addState(state);

  if (collectOperations(*op, state).failed())
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// EdgeOp
//===----------------------------------------------------------------------===//

LogicalResult translation::collect(EdgeOp &op, SDFG &sdfg) {
  SDFGNode sdfgNode = utils::getParentSDFG(*op);
  StateNode srcNode = sdfgNode.getStateBySymRef(op.src());
  StateNode destNode = sdfgNode.getStateBySymRef(op.dest());

  State src = sdfg.lookup(srcNode.sym_name());
  State dest = sdfg.lookup(destNode.sym_name());

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

LogicalResult translation::collect(AllocOp &op, ScopeNode &scope) {
  Array array(op.getName(), op.transient(), utils::getSizedType(op.getType()));
  scope.getSDFG().addArray(array);

  return success();
}

//===----------------------------------------------------------------------===//
// TaskletNode
//===----------------------------------------------------------------------===//

LogicalResult translation::collect(TaskletNode &op, ScopeNode &scope) {
  Tasklet tasklet(op.getLoc());
  tasklet.setName(getTaskletName(op));
  scope.addNode(tasklet);

  for (unsigned i = 0; i < op.getNumOperands(); ++i) {
    Connector connector(tasklet, op.getInputName(i));
    tasklet.addInConnector(connector);

    MultiEdge edge(scope.lookup(op.getOperand(i)), connector);
    scope.addEdge(edge);
  }

  for (unsigned i = 0; i < op.getNumResults(); ++i) {
    Connector connector(tasklet, op.getOutputName(i));
    tasklet.addOutConnector(connector);

    insertTransientArray(op.getLoc(), connector, op.getResult(i), scope);
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

LogicalResult translation::collect(NestedSDFGNode &op, ScopeNode &scope) {
  SDFG sdfg(op.getLoc());

  if (collectSDFG(*op, sdfg).failed())
    return failure();

  NestedSDFG nestedSDFG(op.getLoc(), sdfg);
  nestedSDFG.setName(utils::generateName("nested_sdfg"));
  scope.addNode(nestedSDFG);

  for (unsigned i = 0; i < op.num_args(); ++i) {
    Connector connector(
        nestedSDFG,
        utils::valueToString(op.body().getArgument(i), *op.getOperation()));
    nestedSDFG.addInConnector(connector);

    MultiEdge edge(scope.lookup(op.getOperand(i)), connector);
    scope.addEdge(edge);
  }

  for (unsigned i = op.num_args(); i < op.getNumOperands(); ++i) {
    Connector connector(
        nestedSDFG,
        utils::valueToString(op.body().getArgument(i), *op.getOperation()));
    nestedSDFG.addOutConnector(connector);

    Access access(op.getLoc());
    access.setName(utils::valueToString(op.getOperand(i)));
    scope.addNode(access);

    Connector accOut(access);
    access.addOutConnector(accOut);

    MultiEdge edge(connector, accOut);
    scope.addEdge(edge);

    scope.mapConnector(op.getOperand(i), accOut);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MapNode
//===----------------------------------------------------------------------===//

LogicalResult translation::collect(MapNode &op, ScopeNode &scope) {
  MapEntry mapEntry(op.getLoc());
  mapEntry.setName(utils::generateName("mapEntry"));

  MapExit mapExit(op.getLoc());
  mapExit.setName(utils::generateName("mapExit"));

  mapExit.setEntry(mapEntry);
  mapEntry.setExit(mapExit);

  for (BlockArgument bArg : op.body().getArguments()) {
    mapEntry.addParam(utils::valueToString(bArg));
  }

  for (unsigned i = 0; i < op.lowerBounds().size(); ++i) {
    std::string lb = utils::attributeToString(op.lowerBounds()[i], *op);
    std::string ub = utils::attributeToString(op.upperBounds()[i], *op);
    std::string st = utils::attributeToString(op.steps()[i], *op);

    Range r(lb, ub, st, "1");
    mapEntry.addRange(r);
  }

  scope.addNode(mapEntry);
  scope.addNode(mapExit);

  if (collectOperations(*op, mapEntry).failed())
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// ConsumeNode
//===----------------------------------------------------------------------===//

LogicalResult translation::collect(ConsumeNode &op, ScopeNode &scope) {
  ConsumeEntry consumeEntry(op.getLoc());
  consumeEntry.setName(utils::generateName("consumeEntry"));

  ConsumeExit consumeExit(op.getLoc());
  consumeExit.setName(utils::generateName("consumeExit"));

  consumeExit.setEntry(consumeEntry);
  consumeEntry.setExit(consumeExit);

  // Stream and all

  Connector elem(consumeEntry, "OUT_e");
  consumeEntry.addOutConnector(elem);
  consumeEntry.mapConnector(op.body().getArgument(1), elem);

  scope.addNode(consumeEntry);
  scope.addNode(consumeExit);

  if (collectOperations(*op, consumeEntry).failed())
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

LogicalResult translation::collect(CopyOp &op, ScopeNode &scope) {
  Access access(op.getLoc());
  access.setName(utils::valueToString(op.dest()));
  scope.addNode(access);

  Connector accOut(access);
  access.addOutConnector(accOut);

  MultiEdge edge(scope.lookup(op.src()), accOut);
  scope.addEdge(edge);

  scope.mapConnector(op.dest(), accOut);
  return success();
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

LogicalResult translation::collect(StoreOp &op, ScopeNode &scope) {
  Access access(op.getLoc());
  access.setName(utils::valueToString(op.arr()));
  scope.addNode(access);

  Connector accOut(access);
  access.addOutConnector(accOut);

  if (scope.getType() == NType::MapEntry) {
    MapExit mapExit = scope.getMapEntry().getExit();

    Connector in(mapExit,
                 "IN_" + std::to_string(mapExit.getInConnectorCount()));
    mapExit.addInConnector(in);
    MultiEdge edgeTmp(scope.lookup(op.val()), in);
    scope.addEdge(edgeTmp);

    Connector out(mapExit,
                  "OUT_" + std::to_string(mapExit.getOutConnectorCount()));
    mapExit.addOutConnector(out);
    MultiEdge edge(out, accOut);
    scope.addEdge(edge);

    scope.getState().mapConnector(op.arr(), accOut);
    return success();
  }

  if (scope.getType() == NType::ConsumeEntry) {
    ConsumeExit consumeExit = scope.getConsumeEntry().getExit();

    Connector in(consumeExit,
                 "IN_" + std::to_string(consumeExit.getInConnectorCount()));
    consumeExit.addInConnector(in);
    MultiEdge edgeTmp(scope.lookup(op.val()), in);
    scope.addEdge(edgeTmp);

    Connector out(consumeExit,
                  "OUT_" + std::to_string(consumeExit.getOutConnectorCount()));
    consumeExit.addOutConnector(out);
    MultiEdge edge(out, accOut);
    scope.addEdge(edge);

    scope.getState().mapConnector(op.arr(), accOut);
    return success();
  }

  MultiEdge edge(scope.lookup(op.val()), accOut);
  scope.addEdge(edge);

  scope.mapConnector(op.arr(), accOut);
  return success();
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

LogicalResult translation::collect(LoadOp &op, ScopeNode &scope) {
  Connector connector = scope.lookup(op.arr());
  scope.mapConnector(op.res(), connector);

  return success();
}
