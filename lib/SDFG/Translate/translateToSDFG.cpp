// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

/// This file contains function to translate the SDFG dialect to the SDFG IR. It
/// performs the translation in two passes. First it collects all operations and
/// generates an internal IR, which in the second pass is used to generate JSON.

#include "SDFG/Translate/Node.h"
#include "SDFG/Translate/Translation.h"
#include "SDFG/Translate/liftToPython.h"
#include "SDFG/Utils/Utils.h"
#include "llvm/ADT/DenseMap.h"
#include <regex>
#include <string>

using namespace mlir;
using namespace sdfg;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Inserts a transient array connecting to the provided connector and mapping
/// to the provided value.
static void insertTransientArray(Location location,
                                 translation::Connector connector, Value value,
                                 translation::ScopeNode &scope) {
  using namespace translation;

  Array array(sdfg::utils::generateName("tmp"), /*transient=*/true,
              /*stream=*/false, value.getType());

  if (sdfg::utils::isSizedType(value.getType()))
    array = Array(sdfg::utils::generateName("tmp"), /*transient=*/true,
                  /*stream=*/false, sdfg::utils::getSizedType(value.getType()));

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

/// Collects a operation by performing a case distinction on the operation type.
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

/// Collects all operations in a SDFG.
LogicalResult collectSDFG(Operation &op, translation::SDFG &sdfg) {
  using namespace translation;

  sdfg.setName(sdfg::utils::generateName("sdfg"));

  for (BlockArgument ba : op.getRegion(0).getArguments()) {
    if (sdfg::utils::isSizedType(ba.getType())) {
      SizedType sizedType = sdfg::utils::getSizedType(ba.getType());

      for (StringAttr sym : sizedType.getSymbols())
        // IDEA: Support other types?
        sdfg.addSymbol(Symbol(sym.getValue(), DType::int64));

      Array array(sdfg::utils::valueToString(ba), /*transient=*/false,
                  /*stream=*/false, sizedType);
      sdfg.addArg(array);
    } else {
      Array array(sdfg::utils::valueToString(ba), /*transient=*/false,
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
    std::string entryName =
        sdfg::utils::attributeToString(op.getAttr("entry"), op);
    entryName.erase(0, 1);
    sdfg.setStartState(sdfg.lookup(entryName));
  } else {
    StateNode stateNode = *op.getRegion(0).getOps<StateNode>().begin();
    StringRef entryName = stateNode.getSymName();
    sdfg.setStartState(sdfg.lookup(entryName));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Module
//===----------------------------------------------------------------------===//

/// Translates a module containing SDFG dialect to SDFG IR, outputs the result
/// to the provided output stream.
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

/// Collects state node information in a top-level SDFG.
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

/// Collects edge information in a top-level SDFG.
LogicalResult translation::collect(EdgeOp &op, SDFG &sdfg) {
  Operation *sdfgNode = sdfg::utils::getParentSDFG(*op);

  StateNode srcNode =
      isa<SDFGNode>(sdfgNode)
          ? cast<SDFGNode>(sdfgNode).getStateBySymRef(op.getSrc())
          : cast<NestedSDFGNode>(sdfgNode).getStateBySymRef(op.getSrc());
  StateNode destNode =
      isa<SDFGNode>(sdfgNode)
          ? cast<SDFGNode>(sdfgNode).getStateBySymRef(op.getDest())
          : cast<NestedSDFGNode>(sdfgNode).getStateBySymRef(op.getDest());

  State src = sdfg.lookup(srcNode.getSymName());
  State dest = sdfg.lookup(destNode.getSymName());

  InterstateEdge edge(op.getLoc(), src, dest);
  sdfg.addEdge(edge);

  std::string refname = "";

  if (op.getRef() != Value()) {
    refname = sdfg::utils::valueToString(op.getRef());

    if (op.getRef().getDefiningOp() != nullptr) {
      AllocOp allocOp = cast<AllocOp>(op.getRef().getDefiningOp());
      refname = allocOp.getName().value_or(refname);
    }
  }

  std::string replacedCondition =
      std::regex_replace(op.getCondition().str(), std::regex("ref"), refname);
  edge.setCondition(StringRef(replacedCondition));

  for (mlir::Attribute attr : op.getAssign()) {
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

/// Collects array/stream allocation information in a top-level SDFG.
LogicalResult translation::collect(AllocOp &op, SDFG &sdfg) {
  Array array(op.getContainerName(), op.getTransient(), op.isStream(),
              sdfg::utils::getSizedType(op.getType()));
  sdfg.addArray(array);

  return success();
}

/// Collects array/stream allocation information in a scope.
LogicalResult translation::collect(AllocOp &op, ScopeNode &scope) {
  Array array(op.getContainerName(), op.getTransient(), op.isStream(),
              sdfg::utils::getSizedType(op.getType()));
  scope.getSDFG().addArray(array);

  return success();
}

//===----------------------------------------------------------------------===//
// AllocSymbolOp
//===----------------------------------------------------------------------===//

/// Collects symbol allocation information in a top-level SDFG.
LogicalResult translation::collect(AllocSymbolOp &op, SDFG &sdfg) {
  // IDEA: Support other types?
  Symbol sym(op.getSym(), DType::int64);
  sdfg.addSymbol(sym);
  return success();
}

//===----------------------------------------------------------------------===//
// TaskletNode
//===----------------------------------------------------------------------===//

/// Collects tasklet information in a scope.
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

  // HACK: Lifts annotated tasklets to python code (generated by the converter)
  if (op->hasAttr("insert_code")) {
    std::string operation = op->getAttr("insert_code").cast<StringAttr>().str();

    if (operation == "cbrt") {
      std::string nameOut = op.getOutputName(0);
      std::string nameIn = op.getInputName(0);
      Code code(nameOut + " = " + nameIn + " ** (1. / 3.)",
                CodeLanguage::Python);
      tasklet.setCode(code);
    } else if (operation == "exit") {
      Code code("sys.exit()", CodeLanguage::Python);
      tasklet.setCode(code);
    } else {
      // TODO: Support inputs & outputs
      if (tasklet.getOutConnectorCount() > 0) {
        emitError(op.getLoc(), "return types not supported");
        return failure();
      }

      if (tasklet.getInConnectorCount() > 0) {
        emitError(op.getLoc(), "input types not supported");
        return failure();
      }

      std::string declString = "extern \\\"C\\\" void " + operation + "();\\n";
      std::string codeString = operation + "();";

      Code code_global(declString, CodeLanguage::CPP);
      Code code(codeString, CodeLanguage::CPP);

      tasklet.setGlobalCode(code_global);
      tasklet.setCode(code);
      tasklet.setHasSideEffect(true);
    }

  } else {
    Optional<std::string> code_data = liftToPython(*op);
    if (code_data.has_value()) {
      Code code(code_data.value(), CodeLanguage::Python);
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

/// Collects library call information in a scope.
LogicalResult translation::collect(LibCallOp &op, ScopeNode &scope) {
  Library lib(op.getLoc());
  lib.setName(sdfg::utils::generateName(op.getCallee().str()));
  lib.setClasspath(op.getCallee());
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

/// Collects nested SDFG node information in a scope.
LogicalResult translation::collect(NestedSDFGNode &op, ScopeNode &scope) {
  SDFG sdfg(op.getLoc());

  if (collectSDFG(*op, sdfg).failed())
    return failure();

  NestedSDFG nestedSDFG(op.getLoc(), sdfg);
  nestedSDFG.setName(sdfg::utils::generateName("nested_sdfg"));
  scope.addNode(nestedSDFG);

  for (unsigned i = 0; i < op.getNumArgs(); ++i) {
    Connector connector(nestedSDFG,
                        sdfg::utils::valueToString(op.getBody().getArgument(i),
                                                   *op.getOperation()));
    nestedSDFG.addInConnector(connector);

    MultiEdge edge(op.getLoc(), scope.lookup(op.getOperand(i)), connector);
    scope.addEdge(edge);
  }

  for (unsigned i = op.getNumArgs(); i < op.getNumOperands(); ++i) {
    Connector connector(nestedSDFG,
                        sdfg::utils::valueToString(op.getBody().getArgument(i),
                                                   *op.getOperation()));

    nestedSDFG.addOutConnector(connector);
    nestedSDFG.addInConnector(connector);

    MultiEdge edge_in(op.getLoc(), scope.lookup(op.getOperand(i)), connector);
    scope.addEdge(edge_in);

    std::string arrName = sdfg::utils::valueToString(op.getOperand(i));
    if (op.getOperand(i).getDefiningOp() != nullptr)
      arrName = cast<AllocOp>(op.getOperand(i).getDefiningOp())
                    .getName()
                    .value_or(arrName);

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

/// Collects map node information in a scope.
LogicalResult translation::collect(MapNode &op, ScopeNode &scope) {
  MapEntry mapEntry(op.getLoc());
  mapEntry.setName(sdfg::utils::generateName("mapEntry"));

  MapExit mapExit(op.getLoc());
  mapExit.setName(sdfg::utils::generateName("mapExit"));

  mapExit.setEntry(mapEntry);
  mapEntry.setExit(mapExit);

  scope.addNode(mapEntry);
  scope.addNode(mapExit);

  for (BlockArgument bArg : op.getBody().getArguments()) {
    std::string name = sdfg::utils::valueToString(bArg);
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
      lb = sdfg::utils::attributeToString(
          op.getLowerBounds()[lbSymNumCounter++], *op);
    } else {
      Value val = op.getOperand(lbNum);
      lb = sdfg::utils::generateName("LB");

      Connector valConn(mapEntry, lb);
      mapEntry.addInConnector(valConn);

      MultiEdge multiedge(op.getLoc(), scope.lookup(val), valConn);
      scope.addEdge(multiedge);
    }

    int64_t ubNum = ubList[i].cast<IntegerAttr>().getInt();
    if (ubNum < 0) {
      ub = sdfg::utils::attributeToString(
          op.getUpperBounds()[ubSymNumCounter++], *op);
    } else {
      Value val = op.getOperand(ubNum);
      ub = sdfg::utils::generateName("UB");

      Connector valConn(mapEntry, ub);
      mapEntry.addInConnector(valConn);

      MultiEdge multiedge(op.getLoc(), scope.lookup(val), valConn);
      scope.addEdge(multiedge);
    }

    int64_t stNum = stList[i].cast<IntegerAttr>().getInt();
    if (stNum < 0) {
      st =
          sdfg::utils::attributeToString(op.getSteps()[stSymNumCounter++], *op);
    } else {
      Value val = op.getOperand(stNum);
      st = sdfg::utils::generateName("ST");

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

/// Collects consume node information in a scope.
LogicalResult translation::collect(ConsumeNode &op, ScopeNode &scope) {
  ConsumeEntry consumeEntry(op.getLoc());
  consumeEntry.setName(sdfg::utils::generateName("consumeEntry"));

  ConsumeExit consumeExit(op.getLoc());
  consumeExit.setName(sdfg::utils::generateName("consumeExit"));

  scope.addNode(consumeEntry);
  scope.addNode(consumeExit);

  consumeExit.setEntry(consumeEntry);
  consumeEntry.setExit(consumeExit);

  if (op.getNumPes().has_value()) {
    llvm::SmallString<4U> num_pes;
    op.getNumPes().value().toStringUnsigned(num_pes);
    consumeEntry.setNumPes(num_pes.str().str());
  } else {
    consumeEntry.setNumPes("1");
  }

  consumeEntry.setPeIndex(sdfg::utils::valueToString(op.pe()));

  if (op.getCondition().has_value()) {
    StateNode parentState = sdfg::utils::getParentState(*op);
    Operation *condFunc = parentState.lookupSymbol(op.getCondition().value());

    Optional<std::string> code_data = liftToPython(*condFunc);
    if (code_data.has_value()) {
      Code code(code_data.value(), CodeLanguage::Python);
      consumeEntry.setCondition(code);
    } else {
      // TODO: Write content as code
    }
  }

  Connector stream(consumeEntry, "IN_stream");
  consumeEntry.addInConnector(stream);

  MultiEdge edge(op.getLoc(), scope.lookup(op.getStream()), stream);
  scope.addEdge(edge);

  Connector elem(consumeEntry, "OUT_stream");

  consumeEntry.addOutConnector(elem);
  consumeEntry.mapConnector(op.elem(), elem);

  if (!op.pe().getUses().empty()) {
    // TODO: Handle uses of PE
    /* std::string name = sdfg::utils::valueToString(op.pe());
    Tasklet task(op.getLoc());
    task.setName("SYM" + name);

    Connector taskOut(task, "_SYM_OUT");
    task.addOutConnector(taskOut);
    Code code("_SYM_OUT = " + name, CodeLanguage::Python);
    task.setCode(code);

    consumeEntry.addNode(task);
    insertTransientArray(op.getLoc(), taskOut, op.getPe(), consumeEntry); */
  }

  if (collectOperations(*op, consumeEntry).failed())
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

/// Collects copy operation information in a scope.
LogicalResult translation::collect(CopyOp &op, ScopeNode &scope) {
  Access access(op.getLoc());

  std::string name = sdfg::utils::valueToString(op.getDest());

  if (op.getDest().getDefiningOp() != nullptr) {
    AllocOp allocOp = cast<AllocOp>(op.getDest().getDefiningOp());
    name = allocOp.getName().value_or(name);
  }

  access.setName(name);
  scope.addNode(access);

  Connector accOut(access);
  access.addOutConnector(accOut);

  MultiEdge edge(op.getLoc(), scope.lookup(op.getSrc()), accOut);
  scope.addEdge(edge);

  scope.mapConnector(op.getDest(), accOut);
  return success();
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

/// Collects store operation information in a scope.
LogicalResult translation::collect(StoreOp &op, ScopeNode &scope) {
  std::string name = sdfg::utils::valueToString(op.getArr());

  if (op.getArr().getDefiningOp() != nullptr) {
    AllocOp allocOp = cast<AllocOp>(op.getArr().getDefiningOp());
    name = allocOp.getName().value_or(name);
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
          sdfg::utils::attributeToString(symNumList[-num.getInt() - 1], *op);
      Range range(idx, idx, "1", "1");
      accOut.addRange(range);
    }

    Connector source = scope.lookup(op.getVal());
    scope.routeWrite(source, accOut);
    scope.mapConnector(op.getArr(), accOut);
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
            sdfg::utils::attributeToString(symNumList[-num.getInt() - 1], *op);
        Range range(idx, idx, "1", "1");
        accOut.addRange(range);
      } else {
        std::string idx =
            sdfg::utils::valueToString(op.getOperand(num.getInt()));
        Range range(idx, idx, "1", "1");
        accOut.addRange(range);
      }
    }

    scope.routeWrite(scope.lookup(op.getVal()), accOut);
    scope.mapConnector(op.getArr(), accOut);
    return success();
  }

  Tasklet task(op.getLoc());
  task.setName("indirect_store" + name);
  scope.addNode(task);

  Connector taskArr(task, "_array");
  task.addOutConnector(taskArr);

  Connector taskVal(task, "_value");
  task.addInConnector(taskVal);
  MultiEdge valEdge(op.getLoc(), scope.lookup(op.getVal()), taskVal);
  scope.addEdge(valEdge);

  std::string accessCode = "_array[";

  for (unsigned i = 0; i < numList.size(); ++i) {
    if (i > 0)
      accessCode.append(", ");

    IntegerAttr num = numList[i].cast<IntegerAttr>();
    if (num.getInt() < 0) {
      accessCode.append(
          sdfg::utils::attributeToString(symNumList[-num.getInt() - 1], *op));
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
  if (ArrayType array = op.getArr().getType().dyn_cast<ArrayType>()) {
    SizedType sized = array.getDimensions();

    if (sized.getRank() == 1 && sized.getIntegers().size() == 1 &&
        sized.getIntegers()[0] == 1)
      accessCode = "_array = _value";
  }

  task.setCode(Code(accessCode, CodeLanguage::Python));

  scope.routeWrite(taskArr, accOut);
  scope.mapConnector(op.getArr(), accOut);
  return success();
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

/// Collects load operation information in a scope.
LogicalResult translation::collect(LoadOp &op, ScopeNode &scope) {
  // TODO: Implement a dce pass
  if (op.use_empty())
    return success();

  std::string name = sdfg::utils::valueToString(op.getArr());

  if (op.getArr().getDefiningOp() != nullptr) {
    AllocOp allocOp = cast<AllocOp>(op.getArr().getDefiningOp());
    name = allocOp.getName().value_or(name);
  }

  ArrayAttr numList = op->getAttr("indices_numList").cast<ArrayAttr>();
  ArrayAttr symNumList = op->getAttr("indices").cast<ArrayAttr>();

  if (!op.isIndirect()) {
    Connector connector = scope.lookup(op.getArr());
    connector.setData(name);

    for (unsigned i = 0; i < numList.size(); ++i) {
      IntegerAttr num = numList[i].cast<IntegerAttr>();
      std::string idx =
          sdfg::utils::attributeToString(symNumList[-num.getInt() - 1], *op);
      Range range(idx, idx, "1", "1");
      connector.addRange(range);
    }

    scope.mapConnector(op.getRes(), connector);
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
    Connector connector = scope.lookup(op.getArr());
    connector.setData(name);

    for (unsigned i = 0; i < numList.size(); ++i) {
      IntegerAttr num = numList[i].cast<IntegerAttr>();
      if (num.getInt() < 0) {
        std::string idx =
            sdfg::utils::attributeToString(symNumList[-num.getInt() - 1], *op);
        Range range(idx, idx, "1", "1");
        connector.addRange(range);
      } else {
        std::string idx =
            sdfg::utils::valueToString(op.getOperand(num.getInt()));
        Range range(idx, idx, "1", "1");
        connector.addRange(range);
      }
    }

    scope.mapConnector(op.getRes(), connector);
    return success();
  }

  Tasklet task(op.getLoc());
  task.setName("indirect_load" + name);
  scope.addNode(task);

  Connector taskOut(task, "_out");
  task.addOutConnector(taskOut);
  insertTransientArray(op.getLoc(), taskOut, op.getRes(), scope);

  Connector taskArr(task, "_array");
  task.addInConnector(taskArr);
  MultiEdge arrayEdge(op.getLoc(), scope.lookup(op.getArr()), taskArr);
  scope.addEdge(arrayEdge);

  std::string accessCode = "_out = _array[";

  for (unsigned i = 0; i < numList.size(); ++i) {
    if (i > 0)
      accessCode.append(", ");

    IntegerAttr num = numList[i].cast<IntegerAttr>();
    if (num.getInt() < 0) {
      accessCode.append(
          sdfg::utils::attributeToString(symNumList[-num.getInt() - 1], *op));
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
  if (ArrayType array = op.getArr().getType().dyn_cast<ArrayType>()) {
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

/// Collects symbol allocation information in a scope.
LogicalResult translation::collect(AllocSymbolOp &op, ScopeNode &scope) {
  // IDEA: Support other types?
  Symbol sym(op.getSym(), DType::int64);

  scope.getSDFG().addSymbol(sym);
  return success();
}

//===----------------------------------------------------------------------===//
// SymOp
//===----------------------------------------------------------------------===//

/// Collects symbolic expression information in a scope.
LogicalResult translation::collect(SymOp &op, ScopeNode &scope) {
  Tasklet task(op.getLoc());
  task.setName("SYM_" + op.getExpr().str());

  Connector taskOut(task, "_SYM_OUT");
  task.addOutConnector(taskOut);

  Code code("_SYM_OUT = " + op.getExpr().str(), CodeLanguage::Python);
  task.setCode(code);

  scope.addNode(task);
  insertTransientArray(op.getLoc(), taskOut, op.getRes(), scope);

  return success();
}

//===----------------------------------------------------------------------===//
// StreamPushOp
//===----------------------------------------------------------------------===//

/// Collects stream push operation information in a scope.
LogicalResult translation::collect(StreamPushOp &op, ScopeNode &scope) {
  Access access(op.getLoc());
  std::string name = sdfg::utils::valueToString(op.getStr());

  if (op.getStr().getDefiningOp() != nullptr) {
    AllocOp allocOp = cast<AllocOp>(op.getStr().getDefiningOp());
    name = allocOp.getName().value_or(name);
  }

  access.setName(name);
  scope.addNode(access);

  Connector accOut(access);
  access.addOutConnector(accOut);

  Connector source = scope.lookup(op.getVal());
  scope.routeWrite(source, accOut);
  scope.mapConnector(op.getStr(), accOut);

  return success();
}

//===----------------------------------------------------------------------===//
// StreamPopOp
//===----------------------------------------------------------------------===//

/// Collects stream pop operation information in a scope.
LogicalResult translation::collect(StreamPopOp &op, ScopeNode &scope) {
  Connector connector = scope.lookup(op.getStr());
  scope.mapConnector(op.getRes(), connector);

  return success();
}
