#include "SDFG/Translate/Node.h"
#include "SDFG/Translate/Translation.h"
#include "SDFG/Translate/liftToPython.h"
#include "SDFG/Utils/Utils.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "llvm/ADT/DenseMap.h"
#include <regex>
#include <string>

using namespace mlir;
using namespace sdfg;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

void insertTransientArray(Location location, translation::Connector connector,
                          Value value, translation::ScopeNode &scope) {
  using namespace translation;

  Array array(utils::generateName("tmp"), /*transient=*/true, /*stream=*/false,
              value.getType());

  if (utils::isSizedType(value.getType()))
    array = Array(utils::generateName("tmp"), /*transient=*/true,
                  /*stream=*/false, utils::getSizedType(value.getType()));

  SDFG sdfg = scope.getSDFG();
  sdfg.addArray(array);

  Access access(location);
  access.setName(array.name);
  scope.addNode(access);

  Connector accIn(access);
  Connector accOut(access);

  accIn.setData(array.name);
  accOut.setData(array.name);

  access.addInConnector(accIn);
  access.addOutConnector(accOut);

  MultiEdge edge(location, connector, accIn);
  scope.addEdge(edge);

  scope.mapConnector(value, accOut);
}

LogicalResult collectOperations(Operation &op, translation::ScopeNode &scope) {
  using namespace translation;

  for (Operation &operation : op.getRegion(0).getOps()) {
    if (TaskletNode oper = dyn_cast<TaskletNode>(operation)) {
      if (collect(oper, scope).failed())
        return failure();
      continue;
    }

    if (LibCallOp oper = dyn_cast<LibCallOp>(operation)) {
      if (collect(oper, scope).failed())
        return failure();
      continue;
    }

    if (NestedSDFGNode oper = dyn_cast<NestedSDFGNode>(operation)) {
      if (collect(oper, scope).failed())
        return failure();
      continue;
    }

    if (MapNode oper = dyn_cast<MapNode>(operation)) {
      if (collect(oper, scope).failed())
        return failure();
      continue;
    }

    if (ConsumeNode oper = dyn_cast<ConsumeNode>(operation)) {
      if (collect(oper, scope).failed())
        return failure();
      continue;
    }

    if (CopyOp oper = dyn_cast<CopyOp>(operation)) {
      if (collect(oper, scope).failed())
        return failure();
      continue;
    }

    if (StoreOp oper = dyn_cast<StoreOp>(operation)) {
      if (collect(oper, scope).failed())
        return failure();
      continue;
    }

    if (LoadOp oper = dyn_cast<LoadOp>(operation)) {
      if (collect(oper, scope).failed())
        return failure();
      continue;
    }

    if (AllocOp oper = dyn_cast<AllocOp>(operation)) {
      if (collect(oper, scope).failed())
        return failure();
      continue;
    }

    if (AllocSymbolOp oper = dyn_cast<AllocSymbolOp>(operation)) {
      if (collect(oper, scope).failed()) {
        return failure();
      }
      continue;
    }

    if (SymOp oper = dyn_cast<SymOp>(operation)) {
      if (collect(oper, scope).failed()) {
        return failure();
      }
      continue;
    }

    if (StreamPushOp oper = dyn_cast<StreamPushOp>(operation)) {
      if (collect(oper, scope).failed()) {
        return failure();
      }
      continue;
    }

    if (StreamPopOp oper = dyn_cast<StreamPopOp>(operation)) {
      if (collect(oper, scope).failed()) {
        return failure();
      }
      continue;
    }

    // emitError(operation.getLoc(), "Unsupported Operation");
    // return failure();
  }

  return success();
}

LogicalResult collectSDFG(Operation &op, translation::SDFG &sdfg) {
  using namespace translation;

  sdfg.setName(utils::generateName("sdfg"));

  for (BlockArgument ba : op.getRegion(0).getArguments()) {
    if (utils::isSizedType(ba.getType())) {
      Array array(utils::valueToString(ba), /*transient=*/false,
                  /*stream=*/false, utils::getSizedType(ba.getType()));
      sdfg.addArg(array);
    } else {
      Array array(utils::valueToString(ba), /*transient=*/false,
                  /*stream=*/false, ba.getType());
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

  for (AllocSymbolOp allocSymbolOp : op.getRegion(0).getOps<AllocSymbolOp>()) {
    if (collect(allocSymbolOp, sdfg).failed())
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
  Operation *sdfgNode = utils::getParentSDFG(*op);

  StateNode srcNode =
      isa<SDFGNode>(sdfgNode)
          ? cast<SDFGNode>(sdfgNode).getStateBySymRef(op.src())
          : cast<NestedSDFGNode>(sdfgNode).getStateBySymRef(op.src());
  StateNode destNode =
      isa<SDFGNode>(sdfgNode)
          ? cast<SDFGNode>(sdfgNode).getStateBySymRef(op.dest())
          : cast<NestedSDFGNode>(sdfgNode).getStateBySymRef(op.dest());

  State src = sdfg.lookup(srcNode.sym_name());
  State dest = sdfg.lookup(destNode.sym_name());

  InterstateEdge edge(op.getLoc(), src, dest);
  sdfg.addEdge(edge);

  std::string refname = "";

  if (op.ref() != Value()) {
    refname = utils::valueToString(op.ref());

    if (op.ref().getDefiningOp() != nullptr) {
      AllocOp allocOp = cast<AllocOp>(op.ref().getDefiningOp());
      refname = allocOp.getName();
    }
  }

  std::string replacedCondition =
      std::regex_replace(op.condition().str(), std::regex("ref"), refname);
  edge.setCondition(StringRef(replacedCondition));

  for (mlir::Attribute attr : op.assign()) {
    std::pair<StringRef, StringRef> kv =
        attr.cast<StringAttr>().getValue().split(':');

    std::string replacedAssignment =
        std::regex_replace(kv.second.trim().str(), std::regex("ref"), refname);

    edge.addAssignment(
        Assignment(kv.first.trim(), StringRef(replacedAssignment)));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

LogicalResult translation::collect(AllocOp &op, SDFG &sdfg) {
  Array array(op.getName(), op.transient(), op.isStream(),
              utils::getSizedType(op.getType()));
  sdfg.addArray(array);

  return success();
}

LogicalResult translation::collect(AllocOp &op, ScopeNode &scope) {
  Array array(op.getName(), op.transient(), op.isStream(),
              utils::getSizedType(op.getType()));
  scope.getSDFG().addArray(array);

  return success();
}

//===----------------------------------------------------------------------===//
// AllocSymbolOp
//===----------------------------------------------------------------------===//

LogicalResult translation::collect(AllocSymbolOp &op, SDFG &sdfg) {
  // NOTE: Support other types?
  Symbol sym(op.sym(), DType::int64);
  sdfg.addSymbol(sym);
  return success();
}

//===----------------------------------------------------------------------===//
// TaskletNode
//===----------------------------------------------------------------------===//

LogicalResult translation::collect(TaskletNode &op, ScopeNode &scope) {
  Tasklet tasklet(op.getLoc());
  tasklet.setName(getTaskletName(*op));
  scope.addNode(tasklet);

  for (unsigned i = 0; i < op.getNumOperands(); ++i) {
    Connector connector(tasklet, op.getInputName(i));
    tasklet.addInConnector(connector);

    MultiEdge edge(op.getLoc(), scope.lookup(op.getOperand(i)), connector);
    scope.addEdge(edge);
  }

  for (unsigned i = 0; i < op.getNumResults(); ++i) {
    Connector connector(tasklet, op.getOutputName(i));
    tasklet.addOutConnector(connector);

    insertTransientArray(op.getLoc(), connector, op.getResult(i), scope);
  }

  if (op->hasAttr("insert_code")) {
    std::string operation = op->getAttr("insert_code").cast<StringAttr>().str();

    if (operation == "cbrt") {
      std::string nameOut = op.getOutputName(0);
      std::string nameIn = op.getInputName(0);
      Code code(nameOut + " = " + nameIn + " ** (1. / 3.)",
                CodeLanguage::Python);
      tasklet.setCode(code);
    }

    if (operation == "exit") {
      Code code("sys.exit()", CodeLanguage::Python);
      tasklet.setCode(code);
    }

  } else {
    Optional<std::string> code_data = liftToPython(*op);
    if (code_data.hasValue()) {
      Code code(code_data.getValue(), CodeLanguage::Python);
      tasklet.setCode(code);
    } else {
      // TODO: Write content as code
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// LibCallOp
//===----------------------------------------------------------------------===//

LogicalResult translation::collect(LibCallOp &op, ScopeNode &scope) {
  Library lib(op.getLoc());
  lib.setName(utils::generateName(op.callee().str()));
  lib.setClasspath(op.callee());
  scope.addNode(lib);

  for (unsigned i = 0; i < op.getNumOperands(); ++i) {
    Connector connector(lib, op.getInputName(i));
    lib.addInConnector(connector);

    MultiEdge edge(op.getLoc(), scope.lookup(op.getOperand(i)), connector);
    scope.addEdge(edge);
  }

  for (unsigned i = 0; i < op.getNumResults(); ++i) {
    Connector connector(lib, op.getOutputName(i));
    lib.addOutConnector(connector);

    insertTransientArray(op.getLoc(), connector, op.getResult(i), scope);
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

    MultiEdge edge(op.getLoc(), scope.lookup(op.getOperand(i)), connector);
    scope.addEdge(edge);
  }

  for (unsigned i = op.num_args(); i < op.getNumOperands(); ++i) {
    Connector connector(
        nestedSDFG,
        utils::valueToString(op.body().getArgument(i), *op.getOperation()));

    nestedSDFG.addOutConnector(connector);
    nestedSDFG.addInConnector(connector);

    MultiEdge edge_in(op.getLoc(), scope.lookup(op.getOperand(i)), connector);
    scope.addEdge(edge_in);

    std::string arrName = utils::valueToString(op.getOperand(i));
    if (op.getOperand(i).getDefiningOp() != nullptr)
      arrName = cast<AllocOp>(op.getOperand(i).getDefiningOp()).getName();

    Connector accOut = scope.lookup(op.getOperand(i));
    MultiEdge edge_out(op.getLoc(), connector, accOut);
    scope.addEdge(edge_out);

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

  scope.addNode(mapEntry);
  scope.addNode(mapExit);

  for (BlockArgument bArg : op.body().getArguments()) {
    std::string name = utils::valueToString(bArg);
    mapEntry.addParam(name);
  }

  ArrayAttr lbList = op->getAttr("lowerBounds_numList").cast<ArrayAttr>();
  ArrayAttr ubList = op->getAttr("upperBounds_numList").cast<ArrayAttr>();
  ArrayAttr stList = op->getAttr("steps_numList").cast<ArrayAttr>();

  unsigned lbSymNumCounter = 0;
  unsigned ubSymNumCounter = 0;
  unsigned stSymNumCounter = 0;

  for (unsigned i = 0; i < lbList.size(); ++i) {
    std::string lb = "";
    std::string ub = "";
    std::string st = "";

    int64_t lbNum = lbList[i].cast<IntegerAttr>().getInt();
    if (lbNum < 0) {
      lb = utils::attributeToString(op.lowerBounds()[lbSymNumCounter++], *op);
    } else {
      Value val = op.getOperand(lbNum);
      lb = utils::generateName("LB");

      Connector valConn(mapEntry, lb);
      mapEntry.addInConnector(valConn);

      MultiEdge multiedge(op.getLoc(), scope.lookup(val), valConn);
      scope.addEdge(multiedge);
    }

    int64_t ubNum = ubList[i].cast<IntegerAttr>().getInt();
    if (ubNum < 0) {
      ub = utils::attributeToString(op.upperBounds()[ubSymNumCounter++], *op);
    } else {
      Value val = op.getOperand(ubNum);
      ub = utils::generateName("UB");

      Connector valConn(mapEntry, ub);
      mapEntry.addInConnector(valConn);

      MultiEdge multiedge(op.getLoc(), scope.lookup(val), valConn);
      scope.addEdge(multiedge);
    }

    int64_t stNum = stList[i].cast<IntegerAttr>().getInt();
    if (stNum < 0) {
      st = utils::attributeToString(op.steps()[stSymNumCounter++], *op);
    } else {
      Value val = op.getOperand(stNum);
      st = utils::generateName("ST");

      Connector valConn(mapEntry, st);
      mapEntry.addInConnector(valConn);

      MultiEdge multiedge(op.getLoc(), scope.lookup(val), valConn);
      scope.addEdge(multiedge);
    }

    Range r(lb, ub, st, "1");
    mapEntry.addRange(r);
  }

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

  scope.addNode(consumeEntry);
  scope.addNode(consumeExit);

  consumeExit.setEntry(consumeEntry);
  consumeEntry.setExit(consumeExit);

  if (op.num_pes().hasValue()) {
    llvm::SmallString<4U> num_pes;
    op.num_pes().getValue().toStringUnsigned(num_pes);
    consumeEntry.setNumPes(num_pes.str().str());
  } else {
    consumeEntry.setNumPes("1");
  }

  consumeEntry.setPeIndex(utils::valueToString(op.pe()));

  if (op.condition().hasValue()) {
    StateNode parentState = utils::getParentState(*op);
    Operation *condFunc = parentState.lookupSymbol(op.condition().getValue());

    Optional<std::string> code_data = liftToPython(*condFunc);
    if (code_data.hasValue()) {
      Code code(code_data.getValue(), CodeLanguage::Python);
      consumeEntry.setCondition(code);
    } else {
      // TODO: Write content as code
    }
  }

  Connector stream(consumeEntry, "IN_stream");
  consumeEntry.addInConnector(stream);

  MultiEdge edge(op.getLoc(), scope.lookup(op.stream()), stream);
  scope.addEdge(edge);

  Connector elem(consumeEntry, "OUT_stream");

  consumeEntry.addOutConnector(elem);
  consumeEntry.mapConnector(op.elem(), elem);

  if (!op.pe().getUses().empty()) {
    // TODO: Handle uses of PE
    /* std::string name = utils::valueToString(op.pe());
    Tasklet task(op.getLoc());
    task.setName("SYM" + name);

    Connector taskOut(task, "_SYM_OUT");
    task.addOutConnector(taskOut);
    Code code("_SYM_OUT = " + name, CodeLanguage::Python);
    task.setCode(code);

    consumeEntry.addNode(task);
    insertTransientArray(op.getLoc(), taskOut, op.pe(), consumeEntry); */
  }

  if (collectOperations(*op, consumeEntry).failed())
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

LogicalResult translation::collect(CopyOp &op, ScopeNode &scope) {
  Access access(op.getLoc());

  std::string name = utils::valueToString(op.dest());

  if (op.dest().getDefiningOp() != nullptr) {
    AllocOp allocOp = cast<AllocOp>(op.dest().getDefiningOp());
    name = allocOp.getName();
  }

  access.setName(name);
  scope.addNode(access);

  Connector accOut(access);
  access.addOutConnector(accOut);

  MultiEdge edge(op.getLoc(), scope.lookup(op.src()), accOut);
  scope.addEdge(edge);

  scope.mapConnector(op.dest(), accOut);
  return success();
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

LogicalResult translation::collect(StoreOp &op, ScopeNode &scope) {
  std::string name = utils::valueToString(op.arr());

  if (op.arr().getDefiningOp() != nullptr) {
    AllocOp allocOp = cast<AllocOp>(op.arr().getDefiningOp());
    name = allocOp.getName();
  }

  Access access(op.getLoc());
  access.setName(name);
  scope.addNode(access);

  Connector accOut(access);
  accOut.setData(name);
  access.addOutConnector(accOut);

  ArrayAttr numList = op->getAttr("indices_numList").cast<ArrayAttr>();
  ArrayAttr symNumList = op->getAttr("indices").cast<ArrayAttr>();

  if (!op.isIndirect()) {
    for (unsigned i = 0; i < numList.size(); ++i) {
      IntegerAttr num = numList[i].cast<IntegerAttr>();
      std::string idx =
          utils::attributeToString(symNumList[-num.getInt() - 1], *op);
      Range range(idx, idx, "1", "1");
      accOut.addRange(range);
    }

    Connector source = scope.lookup(op.val());
    scope.routeWrite(source, accOut);
    scope.mapConnector(op.arr(), accOut);
    return success();
  }

  // If any of the operands comes from a non-map op
  bool dependsOnNonMap = false;

  for (unsigned i = 0; i < numList.size(); ++i) {
    IntegerAttr num = numList[i].cast<IntegerAttr>();
    if (num.getInt() >= 0) {

      if (BlockArgument bArg =
              op.getOperand(num.getInt()).dyn_cast<BlockArgument>()) {
        if (!isa<MapNode>(bArg.getParentRegion()->getParentOp())) {
          dependsOnNonMap = true;
        } else if (dependsOnNonMap) {
          emitError(op.getLoc(),
                    "Mixing of map-indices and non-map-indices not supported");
          return failure();
        }
      } else if (!dependsOnNonMap && i > 0) {
        emitError(op.getLoc(),
                  "Mixing of map-indices and non-map-indices not supported");
        return failure();
      } else {
        dependsOnNonMap = true;
      }
    }
  }

  if (!dependsOnNonMap) {
    for (unsigned i = 0; i < numList.size(); ++i) {
      IntegerAttr num = numList[i].cast<IntegerAttr>();
      if (num.getInt() < 0) {
        std::string idx =
            utils::attributeToString(symNumList[-num.getInt() - 1], *op);
        Range range(idx, idx, "1", "1");
        accOut.addRange(range);
      } else {
        std::string idx = utils::valueToString(op.getOperand(num.getInt()));
        Range range(idx, idx, "1", "1");
        accOut.addRange(range);
      }
    }

    scope.routeWrite(scope.lookup(op.val()), accOut);
    scope.mapConnector(op.arr(), accOut);
    return success();
  }

  Tasklet task(op.getLoc());
  task.setName("indirect_store" + name);
  scope.addNode(task);

  Connector taskOut(task, "_out");
  task.addOutConnector(taskOut);

  Connector taskArr(task, "_array");
  task.addInConnector(taskArr);
  MultiEdge arrayEdge(op.getLoc(), scope.lookup(op.arr()), taskArr);
  scope.addEdge(arrayEdge);

  Connector taskVal(task, "_value");
  task.addInConnector(taskVal);
  MultiEdge valEdge(op.getLoc(), scope.lookup(op.val()), taskVal);
  scope.addEdge(valEdge);

  std::string accessCode = "_array[";

  for (unsigned i = 0; i < numList.size(); ++i) {
    if (i > 0)
      accessCode.append(", ");

    IntegerAttr num = numList[i].cast<IntegerAttr>();
    if (num.getInt() < 0) {
      accessCode.append(
          utils::attributeToString(symNumList[-num.getInt() - 1], *op));
    } else {
      Value idxOp = op.getOperand(num.getInt());
      Connector input(task, "_i" + std::to_string(num.getInt()));
      task.addInConnector(input);
      MultiEdge inputEdge(op.getLoc(), scope.lookup(idxOp), input);
      scope.addEdge(inputEdge);
      accessCode.append("_i" + std::to_string(num.getInt()));
    }
  }

  accessCode += "] = _value";

  // Removes 1x arrays
  if (ArrayType array = op.arr().getType().dyn_cast<ArrayType>()) {
    SizedType sized = array.getDimensions();

    if (sized.getRank() == 1 && sized.getIntegers().size() == 1 &&
        sized.getIntegers()[0] == 1)
      accessCode = "_array = _value";
  }

  task.setCode(Code(accessCode, CodeLanguage::Python));

  scope.routeWrite(taskOut, accOut);
  scope.mapConnector(op.arr(), accOut);
  return success();
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

LogicalResult translation::collect(LoadOp &op, ScopeNode &scope) {
  std::string name = utils::valueToString(op.arr());

  if (op.arr().getDefiningOp() != nullptr) {
    AllocOp allocOp = cast<AllocOp>(op.arr().getDefiningOp());
    name = allocOp.getName();
  }

  ArrayAttr numList = op->getAttr("indices_numList").cast<ArrayAttr>();
  ArrayAttr symNumList = op->getAttr("indices").cast<ArrayAttr>();

  if (!op.isIndirect()) {
    Connector connector = scope.lookup(op.arr());
    connector.setData(name);

    for (unsigned i = 0; i < numList.size(); ++i) {
      IntegerAttr num = numList[i].cast<IntegerAttr>();
      std::string idx =
          utils::attributeToString(symNumList[-num.getInt() - 1], *op);
      Range range(idx, idx, "1", "1");
      connector.addRange(range);
    }

    scope.mapConnector(op.res(), connector);
    return success();
  }

  // If any of the operands comes from a non-map op
  bool dependsOnNonMap = false;

  for (unsigned i = 0; i < numList.size(); ++i) {
    IntegerAttr num = numList[i].cast<IntegerAttr>();
    if (num.getInt() >= 0) {
      if (BlockArgument bArg =
              op.getOperand(num.getInt()).dyn_cast<BlockArgument>()) {

        if (!isa<MapNode>(bArg.getParentRegion()->getParentOp())) {
          dependsOnNonMap = true;
        } else if (dependsOnNonMap) {
          emitError(op.getLoc(),
                    "Mixing of map-indices and non-map-indices not supported");
          return failure();
        }

      } else if (!dependsOnNonMap && i > 0) {
        emitError(op.getLoc(),
                  "Mixing of map-indices and non-map-indices not supported");
        return failure();
      } else {
        dependsOnNonMap = true;
      }
    }
  }

  if (!dependsOnNonMap) {
    Connector connector = scope.lookup(op.arr());
    connector.setData(name);

    for (unsigned i = 0; i < numList.size(); ++i) {
      IntegerAttr num = numList[i].cast<IntegerAttr>();
      if (num.getInt() < 0) {
        std::string idx =
            utils::attributeToString(symNumList[-num.getInt() - 1], *op);
        Range range(idx, idx, "1", "1");
        connector.addRange(range);
      } else {
        std::string idx = utils::valueToString(op.getOperand(num.getInt()));
        Range range(idx, idx, "1", "1");
        connector.addRange(range);
      }
    }

    scope.mapConnector(op.res(), connector);
    return success();
  }

  Tasklet task(op.getLoc());
  task.setName("indirect_load" + name);
  scope.addNode(task);

  Connector taskOut(task, "_out");
  task.addOutConnector(taskOut);
  insertTransientArray(op.getLoc(), taskOut, op.res(), scope);

  Connector taskArr(task, "_array");
  task.addInConnector(taskArr);
  MultiEdge arrayEdge(op.getLoc(), scope.lookup(op.arr()), taskArr);
  scope.addEdge(arrayEdge);

  std::string accessCode = "_out = _array[";

  for (unsigned i = 0; i < numList.size(); ++i) {
    if (i > 0)
      accessCode.append(", ");

    IntegerAttr num = numList[i].cast<IntegerAttr>();
    if (num.getInt() < 0) {
      accessCode.append(
          utils::attributeToString(symNumList[-num.getInt() - 1], *op));
    } else {
      Value idxOp = op.getOperand(num.getInt());
      Connector input(task, "_i" + std::to_string(num.getInt()));
      task.addInConnector(input);
      MultiEdge inputEdge(op.getLoc(), scope.lookup(idxOp), input);
      scope.addEdge(inputEdge);
      accessCode.append("_i" + std::to_string(num.getInt()));
    }
  }

  accessCode += "]";

  // Removes 1x arrays
  if (ArrayType array = op.arr().getType().dyn_cast<ArrayType>()) {
    SizedType sized = array.getDimensions();

    if (sized.getRank() == 1 && sized.getIntegers().size() == 1 &&
        sized.getIntegers()[0] == 1)
      accessCode = "_out = _array";
  }

  task.setCode(Code(accessCode, CodeLanguage::Python));
  return success();
}

//===----------------------------------------------------------------------===//
// AllocSymbolOp
//===----------------------------------------------------------------------===//

LogicalResult translation::collect(AllocSymbolOp &op, ScopeNode &scope) {
  // NOTE: Support other types?
  Symbol sym(op.sym(), DType::int64);

  scope.getSDFG().addSymbol(sym);
  return success();
}

//===----------------------------------------------------------------------===//
// SymOp
//===----------------------------------------------------------------------===//

LogicalResult translation::collect(SymOp &op, ScopeNode &scope) {
  Tasklet task(op.getLoc());
  task.setName("SYM_" + op.expr().str());

  Connector taskOut(task, "_SYM_OUT");
  task.addOutConnector(taskOut);

  Code code("_SYM_OUT = " + op.expr().str(), CodeLanguage::Python);
  task.setCode(code);

  scope.addNode(task);
  insertTransientArray(op.getLoc(), taskOut, op.res(), scope);

  return success();
}

//===----------------------------------------------------------------------===//
// StreamPushOp
//===----------------------------------------------------------------------===//

LogicalResult translation::collect(StreamPushOp &op, ScopeNode &scope) {
  Access access(op.getLoc());
  std::string name = utils::valueToString(op.str());

  if (op.str().getDefiningOp() != nullptr) {
    AllocOp allocOp = cast<AllocOp>(op.str().getDefiningOp());
    name = allocOp.getName();
  }

  access.setName(name);
  scope.addNode(access);

  Connector accOut(access);
  access.addOutConnector(accOut);

  Connector source = scope.lookup(op.val());
  scope.routeWrite(source, accOut);
  scope.mapConnector(op.str(), accOut);

  return success();
}

//===----------------------------------------------------------------------===//
// StreamPopOp
//===----------------------------------------------------------------------===//

LogicalResult translation::collect(StreamPopOp &op, ScopeNode &scope) {
  Connector connector = scope.lookup(op.str());
  scope.mapConnector(op.res(), connector);

  return success();
}
