#include "SDIR/Translate/Translation.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AsmState.h"

using namespace mlir;
using namespace sdir;

//===----------------------------------------------------------------------===//
// TranslateToSDFG
//===----------------------------------------------------------------------===//

LogicalResult translateToSDFG(Operation &op, JsonEmitter &jemit) {
  if (SDFGNode node = dyn_cast<SDFGNode>(op))
    return translateSDFGToSDFG(node, jemit);

  if (StateNode node = dyn_cast<StateNode>(op))
    return translateStateToSDFG(node, jemit);

  if (TaskletNode node = dyn_cast<TaskletNode>(op))
    return translateTaskletToSDFG(node, jemit);

  if (MapNode node = dyn_cast<MapNode>(op))
    return translateMapToSDFG(node, jemit);

  if (ConsumeNode node = dyn_cast<ConsumeNode>(op))
    return translateConsumeToSDFG(node, jemit);

  if (EdgeOp Op = dyn_cast<EdgeOp>(op))
    return translateEdgeToSDFG(Op, jemit);

  if (AllocOp Op = dyn_cast<AllocOp>(op))
    return translateAllocToSDFG(Op, jemit);

  if (AllocTransientOp Op = dyn_cast<AllocTransientOp>(op))
    return translateAllocTransientToSDFG(Op, jemit);

  if (GetAccessOp Op = dyn_cast<GetAccessOp>(op))
    return translateGetAccessToSDFG(Op, jemit);

  if (LoadOp Op = dyn_cast<LoadOp>(op))
    return translateLoadToSDFG(Op, jemit);

  if (StoreOp Op = dyn_cast<StoreOp>(op))
    return translateStoreToSDFG(Op, jemit);

  if (CopyOp Op = dyn_cast<CopyOp>(op))
    return translateCopyToSDFG(Op, jemit);

  if (MemletCastOp Op = dyn_cast<MemletCastOp>(op))
    return translateMemletCastToSDFG(Op, jemit);

  if (ViewCastOp Op = dyn_cast<ViewCastOp>(op))
    return translateViewCastToSDFG(Op, jemit);

  if (SubviewOp Op = dyn_cast<SubviewOp>(op))
    return translateSubviewToSDFG(Op, jemit);

  if (AllocStreamOp Op = dyn_cast<AllocStreamOp>(op))
    return translateAllocStreamToSDFG(Op, jemit);

  if (AllocTransientStreamOp Op = dyn_cast<AllocTransientStreamOp>(op))
    return translateAllocTransientStreamToSDFG(Op, jemit);

  if (StreamPopOp Op = dyn_cast<StreamPopOp>(op))
    return translateStreamPopToSDFG(Op, jemit);

  if (StreamPushOp Op = dyn_cast<StreamPushOp>(op))
    return translateStreamPushToSDFG(Op, jemit);

  if (StreamLengthOp Op = dyn_cast<StreamLengthOp>(op))
    return translateStreamLengthToSDFG(Op, jemit);

  if (sdir::CallOp Op = dyn_cast<sdir::CallOp>(op))
    return translateCallToSDFG(Op, jemit);

  if (LibCallOp Op = dyn_cast<LibCallOp>(op))
    return translateLibCallToSDFG(Op, jemit);

  if (AllocSymbolOp Op = dyn_cast<AllocSymbolOp>(op))
    return translateAllocSymbolToSDFG(Op, jemit);

  if (SymOp Op = dyn_cast<SymOp>(op))
    return translateSymbolExprToSDFG(Op, jemit);

  // TODO: Implement ConstantOp & FuncOp
  if (ConstantOp Op = dyn_cast<ConstantOp>(op))
    return success();

  if (FuncOp Op = dyn_cast<FuncOp>(op))
    return success();

  emitError(op.getLoc(), "Unsupported Operation");
  return failure();
}

//===----------------------------------------------------------------------===//
// ModuleOp
//===----------------------------------------------------------------------===//

LogicalResult translateModuleToSDFG(ModuleOp &op, JsonEmitter &jemit) {
  SDIRDialect::resetIDGenerator();

  for (Operation &oper : op.body().getOps())
    if (SDFGNode sdfg = dyn_cast<SDFGNode>(oper))
      if (translateSDFGToSDFG(sdfg, jemit).failed())
        return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// SDFGNode
//===----------------------------------------------------------------------===//

LogicalResult printSDFGNode(SDFGNode &op, JsonEmitter &jemit) {
  jemit.printKVPair("type", "SDFG");
  jemit.printKVPair("sdfg_list_id", SDIRDialect::getNextID(),
                    /*stringify=*/false);

  jemit.startNamedObject("attributes");
  jemit.printAttributes(
      op->getAttrs(),
      /*elidedAttrs=*/{"ID", "entry", "sym_name", "type", "arg_names"});

  jemit.startNamedObject("constants_prop");
  // TODO: Fill this out
  jemit.endObject(); // constants_prop

  if (containsAttr(*op, "arg_names")) {
    Attribute arg_names = op->getAttr("arg_names");
    if (ArrayAttr arg_names_arr = arg_names.cast<ArrayAttr>()) {
      jemit.startNamedList("arg_names");

      for (Attribute arg_name : arg_names_arr.getValue()) {
        if (StringAttr arg_name_str = arg_name.cast<StringAttr>()) {
          jemit.startEntry();
          jemit.printString(arg_name_str.getValue());
        } else {
          return failure();
        }
      }

      jemit.endList(); // arg_names*/
    } else {
      return failure();
    }
  }

  jemit.startNamedObject("_arrays");
  for (Operation &oper : op.body().getOps()) {
    if (AllocOp alloc = dyn_cast<AllocOp>(oper))
      if (translateToSDFG(*alloc, jemit).failed())
        return failure();

    if (AllocTransientOp alloc = dyn_cast<AllocTransientOp>(oper))
      if (translateToSDFG(*alloc, jemit).failed())
        return failure();

    if (AllocStreamOp alloc = dyn_cast<AllocStreamOp>(oper))
      if (translateToSDFG(*alloc, jemit).failed())
        return failure();

    if (AllocTransientStreamOp alloc = dyn_cast<AllocTransientStreamOp>(oper))
      if (translateToSDFG(*alloc, jemit).failed())
        return failure();
  }

  for (Operation &oper : op.body().getOps())
    if (StateNode state = dyn_cast<StateNode>(oper)) {
      for (AllocOp allocOper : state.getAllocs())
        if (translateToSDFG(*allocOper, jemit).failed())
          return failure();

      for (AllocTransientOp allocOper : state.getTransientAllocs())
        if (translateToSDFG(*allocOper, jemit).failed())
          return failure();

      for (AllocStreamOp allocOper : state.getStreamAllocs())
        if (translateToSDFG(*allocOper, jemit).failed())
          return failure();

      for (AllocTransientStreamOp allocOper : state.getTransientStreamAllocs())
        if (translateToSDFG(*allocOper, jemit).failed())
          return failure();
    }
  jemit.endObject(); // _arrays

  jemit.startNamedObject("symbols");
  for (Operation &oper : op.body().getOps()) {
    if (AllocSymbolOp alloc = dyn_cast<AllocSymbolOp>(oper))
      if (translateToSDFG(*alloc, jemit).failed())
        return failure();
  }

  jemit.endObject(); // symbols

  if (!containsAttr(*op, "instrument"))
    jemit.printKVPair("instrument", "No_Instrumentation");

  jemit.startNamedObject("init_code");
  jemit.startNamedObject("frame");
  jemit.printKVPair("string_data", "");
  jemit.printKVPair("language", "CPP");
  jemit.endObject(); // frame
  jemit.endObject(); // init_code

  jemit.startNamedObject("exit_code");
  jemit.startNamedObject("frame");
  jemit.printKVPair("string_data", "");
  jemit.printKVPair("language", "CPP");
  jemit.endObject(); // frame
  jemit.endObject(); // exit_code

  jemit.startNamedObject("global_code");
  jemit.startNamedObject("frame");
  jemit.printKVPair("string_data", "");
  jemit.printKVPair("language", "CPP");
  jemit.endObject(); // frame
  jemit.endObject(); // global_code

  jemit.printKVPair("name", op.sym_name());
  jemit.endObject(); // attributes

  jemit.startNamedList("nodes");

  unsigned stateID = 0;

  for (Operation &oper : op.body().getOps())
    if (StateNode state = dyn_cast<StateNode>(oper)) {
      state.setID(stateID);
      if (translateStateToSDFG(state, jemit).failed())
        return failure();
      stateID++;
    }

  jemit.endList(); // nodes

  jemit.startNamedList("edges");

  for (Operation &oper : op.body().getOps())
    if (EdgeOp edge = dyn_cast<EdgeOp>(oper))
      if (translateToSDFG(*edge, jemit).failed())
        return failure();

  jemit.endList(); // edges

  StateNode entryState = op.getStateBySymRef(op.entry());
  jemit.printKVPair("start_state", entryState.ID(), /*stringify=*/false);

  return success();
}

LogicalResult translateSDFGToSDFG(SDFGNode &op, JsonEmitter &jemit) {
  if (!op.isNested()) {
    jemit.startObject();

    if (printSDFGNode(op, jemit).failed())
      return failure();

    jemit.endObject();
    return success();
  }

  jemit.startObject();
  jemit.printKVPair("type", "NestedSDFG");
  jemit.printKVPair("id", op.ID(), /*stringify=*/false);

  jemit.startNamedObject("attributes");

  if (!containsAttr(*op, "instrument"))
    jemit.printKVPair("instrument", "No_Instrumentation");

  if (!containsAttr(*op, "schedule"))
    jemit.printKVPair("schedule", "Default");

  jemit.startNamedObject("sdfg");

  if (printSDFGNode(op, jemit).failed())
    return failure();

  jemit.endObject(); // sdfg
  jemit.endObject(); // attributes

  jemit.endObject();
  return success();
}

//===----------------------------------------------------------------------===//
// StateNode
//===----------------------------------------------------------------------===//

LogicalResult translateStateToSDFG(StateNode &op, JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "SDFGState");
  jemit.printKVPair("label", op.sym_name());
  jemit.printKVPair("id", op.ID(), /*stringify=*/false);

  jemit.startNamedObject("attributes");
  jemit.printAttributes(op->getAttrs(), /*elidedAttrs=*/{"ID", "sym_name"});
  if (!containsAttr(*op, "instrument"))
    jemit.printKVPair("instrument", "No_Instrumentation");

  jemit.endObject(); // attributes

  jemit.startNamedList("nodes");

  unsigned nodeID = 0;
  for (Operation &oper : op.body().getOps()) {
    if (TaskletNode tasklet = dyn_cast<TaskletNode>(oper)) {
      tasklet.setID(nodeID);
      if (translateTaskletToSDFG(tasklet, jemit).failed())
        return failure();
    }

    if (SDFGNode sdfg = dyn_cast<SDFGNode>(oper)) {
      sdfg.setID(nodeID);
      if (translateSDFGToSDFG(sdfg, jemit).failed())
        return failure();
    }

    if (GetAccessOp acc = dyn_cast<GetAccessOp>(oper)) {
      acc.setID(nodeID);
      if (translateGetAccessToSDFG(acc, jemit).failed())
        return failure();
    }

    if (MapNode map = dyn_cast<MapNode>(oper)) {
      map.setEntryID(nodeID);
      map.setExitID(++nodeID);
      if (translateMapToSDFG(map, jemit).failed())
        return failure();
    }

    if (ConsumeNode consume = dyn_cast<ConsumeNode>(oper)) {
      consume.setEntryID(nodeID);
      consume.setExitID(++nodeID);
      if (translateConsumeToSDFG(consume, jemit).failed())
        return failure();
    }

    nodeID++;
  }

  jemit.endList(); // nodes

  jemit.startNamedList("edges");

  for (Operation &oper : op.body().getOps()) {
    if (CopyOp edge = dyn_cast<CopyOp>(oper))
      if (translateToSDFG(*edge, jemit).failed())
        return failure();

    if (StoreOp edge = dyn_cast<StoreOp>(oper))
      if (translateToSDFG(*edge, jemit).failed())
        return failure();

    if (sdir::CallOp edge = dyn_cast<sdir::CallOp>(oper))
      if (translateToSDFG(*edge, jemit).failed())
        return failure();
  }

  jemit.endList(); // edges

  jemit.endObject();
  return success();
}

//===----------------------------------------------------------------------===//
// TaskletNode
//===----------------------------------------------------------------------===//

// Temporary auto-lifting. Will be included into DaCe
LogicalResult liftToPython(TaskletNode &op, JsonEmitter &jemit) {
  int numOps = 0;
  Operation *firstOp = nullptr;

  for (Operation &oper : op.body().getOps()) {
    if (numOps >= 2)
      return failure();
    if (numOps == 0)
      firstOp = &oper;
    ++numOps;
  }

  AsmState state(op);

  if (arith::AddFOp oper = dyn_cast<arith::AddFOp>(firstOp)) {
    std::string nameArg0;
    llvm::raw_string_ostream nameArg0Stream(nameArg0);
    op.getArgument(0).printAsOperand(nameArg0Stream, state);
    nameArg0.erase(0, 1); // Remove %-sign

    std::string nameArg1;
    llvm::raw_string_ostream nameArg1Stream(nameArg1);
    op.getArgument(1).printAsOperand(nameArg1Stream, state);
    nameArg1.erase(0, 1); // Remove %-sign

    jemit.printKVPair("string_data", "__out = " + nameArg0 + " + " + nameArg1);
    jemit.printKVPair("language", "Python");
    return success();
  }

  if (arith::MulFOp oper = dyn_cast<arith::MulFOp>(firstOp)) {
    std::string nameArg0;
    llvm::raw_string_ostream nameArg0Stream(nameArg0);
    op.getArgument(0).printAsOperand(nameArg0Stream, state);
    nameArg0.erase(0, 1); // Remove %-sign

    std::string nameArg1;
    llvm::raw_string_ostream nameArg1Stream(nameArg1);
    op.getArgument(1).printAsOperand(nameArg1Stream, state);
    nameArg1.erase(0, 1); // Remove %-sign

    jemit.printKVPair("string_data", "__out = " + nameArg0 + " * " + nameArg1);
    jemit.printKVPair("language", "Python");
    return success();
  }

  if (arith::ConstantFloatOp oper = dyn_cast<arith::ConstantFloatOp>(firstOp)) {
    std::string val = std::to_string(oper.value().convertToDouble());

    jemit.printKVPair("string_data", "__out = dace.float64(" + val + ")");
    jemit.printKVPair("language", "Python");
    return success();
  }

  return failure();
}

LogicalResult translateTaskletToSDFG(TaskletNode &op, JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "Tasklet");
  jemit.printKVPair("label", op.sym_name());

  jemit.startNamedObject("attributes");
  jemit.printKVPair("label", op.sym_name());

  if (!containsAttr(*op, "instrument"))
    jemit.printKVPair("instrument", "No_Instrumentation");

  jemit.startNamedObject("code");

  AsmState state(op);

  if (liftToPython(op, jemit).failed()) {
    std::string code = "module {\\n func @mlir_entry(";

    for (unsigned i = 0; i < op.getNumArguments(); ++i) {
      BlockArgument bArg = op.getArgument(i);

      std::string name;
      llvm::raw_string_ostream nameStream(name);
      bArg.printAsOperand(nameStream, state);

      std::string type;
      llvm::raw_string_ostream typeStream(type);
      bArg.getType().print(typeStream);

      if (i > 0)
        code.append(", ");
      code.append(name);
      code.append(": ");
      code.append(type);
    }

    code.append(") -> ");

    std::string retType;
    llvm::raw_string_ostream retTypeStream(retType);
    for (Type res : op.getCallableResults())
      res.print(retTypeStream);
    code.append(retType);

    code.append(" {\\n");

    for (Operation &oper : op.body().getOps()) {
      std::string codeLine;
      llvm::raw_string_ostream codeLineStream(codeLine);
      oper.print(codeLineStream);

      if (sdir::ReturnOp ret = dyn_cast<sdir::ReturnOp>(oper))
        codeLine.replace(codeLine.find("sdir.return"), 11, "return");

      codeLine.append("\\n");
      code.append(codeLine);
    }

    code.append("}\\n}");
    jemit.printKVPair("string_data", code);
    jemit.printKVPair("language", "MLIR");
  }

  jemit.endObject(); // code

  jemit.startNamedObject("in_connectors");

  for (BlockArgument bArg : op.getArguments()) {
    std::string name;
    llvm::raw_string_ostream nameStream(name);
    bArg.printAsOperand(nameStream, state);
    name.erase(0, 1); // Remove %-sign
    jemit.printKVPair(name, "null", /*stringify=*/false);
    // bArg.getType().print(jemit.ostream());
  }

  jemit.endObject(); // in_connectors

  jemit.startNamedObject("out_connectors");
  // TODO: print out_connectors
  jemit.printKVPair("__out", "null", /*stringify=*/false);
  jemit.endObject(); // out_connectors

  jemit.endObject(); // attributes

  jemit.printKVPair("id", op.ID(), /*stringify=*/false);
  jemit.endObject();
  return success();
}

//===----------------------------------------------------------------------===//
// MapNode
//===----------------------------------------------------------------------===//

LogicalResult translateMapToSDFG(MapNode &op, JsonEmitter &jemit) {
  // MapEntry
  jemit.startObject();
  jemit.printKVPair("type", "MapEntry");

  jemit.startNamedObject("attributes");
  if (!containsAttr(*op, "instrument"))
    jemit.printKVPair("instrument", "No_Instrumentation");
  if (!containsAttr(*op, "schedule"))
    jemit.printKVPair("schedule", "Default");

  jemit.startNamedList("params");
  AsmState state(op);
  for (BlockArgument arg : op.getBody()->getArguments()) {
    jemit.startEntry();
    std::string name;
    llvm::raw_string_ostream nameStream(name);
    arg.printAsOperand(nameStream, state);
    jemit.printString(name);
  }
  jemit.endList(); // params

  jemit.endObject(); // attributes
  jemit.printKVPair("id", op.entryID(), /*stringify=*/false);
  jemit.printKVPair("scope_exit", op.exitID());

  jemit.endObject();

  // MapExit
  jemit.startObject();
  jemit.printKVPair("type", "MapExit");

  jemit.startNamedObject("attributes");

  jemit.endObject(); // attributes
  jemit.printKVPair("id", op.exitID(), /*stringify=*/false);
  jemit.printKVPair("scope_entry", op.entryID());
  jemit.printKVPair("scope_exit", op.exitID());

  jemit.endObject();
  return success();
}

//===----------------------------------------------------------------------===//
// ConsumeNode
//===----------------------------------------------------------------------===//

LogicalResult translateConsumeToSDFG(ConsumeNode &op, JsonEmitter &jemit) {
  // ConsumeEntry
  jemit.startObject();
  jemit.printKVPair("type", "ConsumeEntry");

  jemit.startNamedObject("attributes");
  if (!containsAttr(*op, "instrument"))
    jemit.printKVPair("instrument", "No_Instrumentation");
  if (!containsAttr(*op, "schedule"))
    jemit.printKVPair("schedule", "Default");

  jemit.endObject(); // attributes
  jemit.printKVPair("id", op.entryID(), /*stringify=*/false);
  jemit.printKVPair("scope_exit", op.exitID());

  jemit.endObject();

  // ConsumeExit
  jemit.startObject();
  jemit.printKVPair("type", "ConsumeExit");

  jemit.startNamedObject("attributes");

  jemit.endObject(); // attributes
  jemit.printKVPair("id", op.exitID(), /*stringify=*/false);
  jemit.printKVPair("scope_entry", op.entryID());
  jemit.printKVPair("scope_exit", op.exitID());

  jemit.endObject();
  return success();
}

//===----------------------------------------------------------------------===//
// EdgeOp
//===----------------------------------------------------------------------===//

LogicalResult translateEdgeToSDFG(EdgeOp &op, JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "Edge");

  jemit.startNamedObject("attributes");
  jemit.startNamedObject("data");
  jemit.printKVPair("type", "InterstateEdge");

  jemit.startNamedObject("attributes");
  jemit.startNamedObject("assignments");

  if (op.assign().hasValue()) {
    ArrayAttr assignments = op.assign().getValue();

    for (Attribute assignment : assignments) {
      if (StringAttr strAttr = assignment.dyn_cast<StringAttr>()) {
        StringRef content = strAttr.getValue();
        std::pair<StringRef, StringRef> kv = content.split(':');
        jemit.printKVPair(kv.first.trim(), kv.second.trim());
      } else {
        return failure();
      }
    }
  }

  jemit.endObject(); // assignments
  jemit.startNamedObject("condition");
  jemit.printKVPair("language", "Python");
  if (op.condition().hasValue()) {
    if (op.condition().getValue().empty()) {
      jemit.printKVPair("string_data", "1");
    } else {
      jemit.printKVPair("string_data", op.condition().getValue());
    }
  } else {
    jemit.printKVPair("string_data", "1");
  }
  jemit.endObject(); // condition
  jemit.endObject(); // attributes
  jemit.endObject(); // data
  jemit.endObject(); // attributes

  SDFGNode sdfg = dyn_cast<SDFGNode>(op->getParentOp());

  StateNode srcState = sdfg.getStateBySymRef(op.src());
  unsigned srcIdx = sdfg.getIndexOfState(srcState);
  jemit.printKVPair("src", srcIdx);

  StateNode destState = sdfg.getStateBySymRef(op.dest());
  unsigned destIdx = sdfg.getIndexOfState(destState);
  jemit.printKVPair("dst", destIdx);

  jemit.endObject();
  return success();
}

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

LogicalResult printScalar(AllocOp &op, JsonEmitter &jemit) {
  jemit.startNamedObject(op.getName());
  jemit.printKVPair("type", "Scalar");

  jemit.startNamedObject("attributes");

  Type element = op.getType().getElementType();
  if (translateTypeToSDFG(element, jemit, "dtype").failed())
    return failure();

  jemit.startNamedList("shape");
  jemit.startEntry();
  jemit.printInt(1);
  jemit.endList(); // shape

  jemit.printKVPair("transient", "false", /*stringify=*/false);
  jemit.printKVPair("storage", "Default");
  jemit.printKVPair("lifetime", "Scope");

  jemit.endObject(); // attributes

  jemit.endObject();
  return success();
}

LogicalResult translateAllocToSDFG(AllocOp &op, JsonEmitter &jemit) {
  if (op.getType().getShape().size() == 0)
    return printScalar(op, jemit);

  jemit.startNamedObject(op.getName());
  jemit.printKVPair("type", "Array");

  jemit.startNamedObject("attributes");
  // TODO: Print the values you can derive
  // jemit.printKVPair("allow_conflicts", "false", /*stringify=*/false);

  jemit.startNamedList("strides");
  ArrayRef<int64_t> shape = op.getType().getIntegers();

  for (int i = shape.size() - 1; i > 0; --i) {
    jemit.startEntry();
    jemit.printInt(shape[i]);
  }

  jemit.startEntry();
  jemit.printInt(1);
  jemit.endList(); // strides

  // jemit.printKVPair("total_size", "4");
  jemit.startNamedList("offset");
  for (unsigned i = 0; i < shape.size(); ++i) {
    jemit.startEntry();
    jemit.printInt(0);
  }
  jemit.endList(); // offset

  // jemit.printKVPair("may_alias", "false", /*stringify*/false);
  // jemit.printKVPair("alignment", 0, /*stringify*/false);
  Type element = op.getType().getElementType();
  if (translateTypeToSDFG(element, jemit, "dtype").failed())
    return failure();

  jemit.startNamedList("shape");
  for (int64_t s : shape) {
    jemit.startEntry();
    jemit.printInt(s);
  }
  jemit.endList(); // shape

  jemit.printKVPair("transient", "false", /*stringify=*/false);
  jemit.printKVPair("storage", "Default");
  jemit.printKVPair("lifetime", "Scope");
  // jemit.startNamedObject("location");
  // jemit.endObject(); // location

  // jemit.printKVPair("debuginfo", "null", /*stringify=*/false);

  jemit.endObject(); // attributes
  jemit.endObject();
  return success();
}

//===----------------------------------------------------------------------===//
// AllocTransientOp
//===----------------------------------------------------------------------===//

LogicalResult translateAllocTransientToSDFG(AllocTransientOp &op,
                                            JsonEmitter &jemit) {
  AsmState state(op.getParentSDFG());
  std::string name;
  llvm::raw_string_ostream nameStream(name);
  op->getResult(0).printAsOperand(nameStream, state);

  jemit.startNamedObject(name);
  jemit.printKVPair("type", "Array");

  jemit.startNamedObject("attributes");
  // TODO: Print the values you can derive
  // jemit.printKVPair("allow_conflicts", "false", /*stringify=*/false);
  // jemit.startNamedList("strides");
  // jemit.endList(); // strides

  // jemit.printKVPair("total_size", "4");
  // jemit.startNamedList("offset");
  // jemit.endList(); // offset

  // jemit.printKVPair("may_alias", "false", /*stringify*/false);
  // jemit.printKVPair("alignment", 0, /*stringify*/false);
  jemit.printKVPair("dtype", "int32");
  // jemit.startNamedList("shape");
  // jemit.endList(); // shape

  jemit.printKVPair("transient", "true", /*stringify=*/false);
  jemit.printKVPair("storage", "Default");
  jemit.printKVPair("lifetime", "Scope");
  // jemit.startNamedObject("location");
  // jemit.endObject(); // location

  // jemit.printKVPair("debuginfo", "null", /*stringify=*/false);

  jemit.endObject(); // attributes
  jemit.endObject();
  return success();
}

//===----------------------------------------------------------------------===//
// GetAccessOp
//===----------------------------------------------------------------------===//

LogicalResult translateGetAccessToSDFG(GetAccessOp &op, JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "AccessNode");
  jemit.printKVPair("label", op.getName());

  jemit.startNamedObject("attributes");
  jemit.printKVPair("access", "ReadWrite");
  jemit.printKVPair("setzero", "false", /*stringify=*/false);
  jemit.printKVPair("data", op.getName());
  jemit.startNamedObject("in_connectors");
  jemit.endObject(); // in_connectors
  jemit.startNamedObject("out_connectors");
  jemit.endObject(); // out_connectors
  jemit.endObject(); // attributes

  jemit.printKVPair("id", op.ID(), /*stringify=*/false);
  jemit.printKVPair("scope_entry", "null", /*stringify=*/false);
  jemit.printKVPair("scope_exit", "null", /*stringify=*/false);
  jemit.endObject();
  return success();
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

LogicalResult translateLoadToSDFG(LoadOp &op, JsonEmitter &jemit) {
  return success();
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

LogicalResult translateStoreToSDFG(StoreOp &op, JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "MultiConnectorEdge");

  jemit.startNamedObject("attributes");
  jemit.startNamedObject("data");
  jemit.printKVPair("type", "Memlet");
  jemit.startNamedObject("attributes");

  jemit.printKVPair("volume", 1);
  jemit.printKVPair("num_accesses", 1);

  if (GetAccessOp aNode = dyn_cast<GetAccessOp>(op.arr().getDefiningOp())) {
    jemit.printKVPair("data", aNode.getName());
  } else {
    return failure();
  }

  jemit.startNamedObject("subset");
  jemit.printKVPair("type", "Range");
  jemit.startNamedList("ranges");

  if (ArrayAttr syms = op->getAttr("indices").cast<ArrayAttr>()) {
    for (Attribute sym : syms.getValue()) {
      if (StringAttr sym_str = sym.cast<StringAttr>()) {
        jemit.startObject();
        jemit.printKVPair("start", sym_str.getValue());
        jemit.printKVPair("end", sym_str.getValue());
        jemit.printKVPair("step", 1);
        jemit.printKVPair("tile", 1);
        jemit.endObject();
      } else {
        return failure();
      }
    }
  } else {
    return failure();
  }

  jemit.endList();   // ranges
  jemit.endObject(); // subset

  jemit.startNamedObject("dst_subset");
  jemit.printKVPair("type", "Range");
  jemit.startNamedList("ranges");

  if (ArrayAttr syms = op->getAttr("indices").cast<ArrayAttr>()) {
    for (Attribute sym : syms.getValue()) {
      if (StringAttr sym_str = sym.cast<StringAttr>()) {
        jemit.startObject();
        jemit.printKVPair("start", sym_str.getValue());
        jemit.printKVPair("end", sym_str.getValue());
        jemit.printKVPair("step", 1);
        jemit.printKVPair("tile", 1);
        jemit.endObject();
      } else {
        return failure();
      }
    }
  } else {
    return failure();
  }

  jemit.endList();   // ranges
  jemit.endObject(); // dst_subset

  jemit.endObject(); // attributes
  jemit.endObject(); // data
  jemit.endObject(); // attributes

  if (sdir::CallOp call = dyn_cast<sdir::CallOp>(op.val().getDefiningOp())) {
    TaskletNode aNode = call.getTasklet();
    jemit.printKVPair("src", aNode.ID());
    jemit.printKVPair("src_connector", "__out");
  } else {
    return failure();
  }

  if (GetAccessOp aNode = dyn_cast<GetAccessOp>(op.arr().getDefiningOp())) {
    jemit.printKVPair("dst", aNode.ID());
    jemit.printKVPair("dst_connector", "null", /*stringify=*/false);
  } else {
    return failure();
  }

  jemit.endObject();
  return success();
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

LogicalResult translateCopyToSDFG(CopyOp &op, JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "MultiConnectorEdge");

  jemit.startNamedObject("attributes");
  jemit.startNamedObject("data");
  jemit.printKVPair("type", "Memlet");
  jemit.startNamedObject("attributes");

  // jemit.printKVPair("volume", "1");
  // jemit.printKVPair("dynamic", "false", /*stringify=*/false);

  /*jemit.startNamedObject("subset");
  jemit.printKVPair("type", "Range");
  jemit.startNamedList("ranges");

  jemit.startObject();
  jemit.printKVPair("start", "0");
  jemit.printKVPair("end", "0");
  jemit.printKVPair("step", "1");
  jemit.printKVPair("tile", "1");
  jemit.endObject();

  jemit.endList(); // ranges
  jemit.endObject(); // subset*/

  // jemit.printKVPair("other_subset", "null", /*stringify=*/false);
  if (GetAccessOp aNode = dyn_cast<GetAccessOp>(op.src().getDefiningOp())) {
    jemit.printKVPair("data", aNode.getName());
  } else {
    return failure();
  }
  // jemit.printKVPair("wcr", "null", /*stringify=*/false);
  // jemit.printKVPair("debuginfo", "null", /*stringify=*/false);
  // jemit.printKVPair("wcr_nonatomic", "false", /*stringify=*/false);
  // jemit.printKVPair("allow_oob", "false", /*stringify=*/false);
  // jemit.printKVPair("num_accesses", "1");

  // jemit.printKVPair("src_subset", "null", /*stringify=*/false);
  /*jemit.startNamedObject("src_subset");
  jemit.printKVPair("type", "Range");
  jemit.startNamedList("ranges");

  jemit.startObject();
  jemit.printKVPair("start", "0");
  jemit.printKVPair("end", "0");
  jemit.printKVPair("step", "1");
  jemit.printKVPair("tile", "1");
  jemit.endObject();

  jemit.endList(); // ranges
  jemit.endObject(); // src_subset*/

  // jemit.printKVPair("dst_subset", "null", /*stringify=*/false);

  jemit.endObject(); // attributes
  jemit.endObject(); // data
  jemit.endObject(); // attributes

  if (GetAccessOp aNode = dyn_cast<GetAccessOp>(op.src().getDefiningOp())) {
    jemit.printKVPair("src", aNode.ID());
    jemit.printKVPair("src_connector", "null", /*stringify=*/false);
  } else {
    return failure();
  }

  if (GetAccessOp aNode = dyn_cast<GetAccessOp>(op.dest().getDefiningOp())) {
    jemit.printKVPair("dst", aNode.ID());
    jemit.printKVPair("dst_connector", "null", /*stringify=*/false);
  } else {
    return failure();
  }

  jemit.endObject();
  return success();
}

//===----------------------------------------------------------------------===//
// MemletCastOp
//===----------------------------------------------------------------------===//

LogicalResult translateMemletCastToSDFG(MemletCastOp &op, JsonEmitter &jemit) {
  return success();
}

//===----------------------------------------------------------------------===//
// ViewCastOp
//===----------------------------------------------------------------------===//

LogicalResult translateViewCastToSDFG(ViewCastOp &op, JsonEmitter &jemit) {
  return success();
}

//===----------------------------------------------------------------------===//
// SubviewOp
//===----------------------------------------------------------------------===//

LogicalResult translateSubviewToSDFG(SubviewOp &op, JsonEmitter &jemit) {
  return success();
}

//===----------------------------------------------------------------------===//
// AllocStreamOp
//===----------------------------------------------------------------------===//

LogicalResult translateAllocStreamToSDFG(AllocStreamOp &op,
                                         JsonEmitter &jemit) {
  AsmState state(op.getParentSDFG());
  std::string name;
  llvm::raw_string_ostream nameStream(name);
  op->getResult(0).printAsOperand(nameStream, state);

  jemit.startNamedObject(name);
  jemit.printKVPair("type", "Stream");

  jemit.startNamedObject("attributes");

  // TODO: Print the values you can derive
  // jemit.startNamedList("offset");
  // jemit.endList(); // offset

  // jemit.printKVPair("buffer_size", "1");
  jemit.printKVPair("dtype", "int32");
  // jemit.startNamedList("shape");
  // jemit.endList(); // shape

  jemit.printKVPair("transient", "false", /*stringify=*/false);
  jemit.printKVPair("storage", "Default");
  jemit.printKVPair("lifetime", "Scope");
  // jemit.startNamedObject("location");
  // jemit.endObject(); // location

  // jemit.printKVPair("debuginfo", "null", /*stringify=*/false);

  jemit.endObject(); // attributes
  jemit.endObject();
  return success();
}

//===----------------------------------------------------------------------===//
// AllocTransientStreamOp
//===----------------------------------------------------------------------===//

LogicalResult translateAllocTransientStreamToSDFG(AllocTransientStreamOp &op,
                                                  JsonEmitter &jemit) {
  AsmState state(op.getParentSDFG());
  std::string name;
  llvm::raw_string_ostream nameStream(name);
  op->getResult(0).printAsOperand(nameStream, state);

  jemit.startNamedObject(name);
  jemit.printKVPair("type", "Stream");

  jemit.startNamedObject("attributes");

  // TODO: Print the values you can derive
  // jemit.startNamedList("offset");
  // jemit.endList(); // offset

  // jemit.printKVPair("buffer_size", "1");
  jemit.printKVPair("dtype", "int32");
  // jemit.startNamedList("shape");
  // jemit.endList(); // shape

  jemit.printKVPair("transient", "true", /*stringify=*/false);
  jemit.printKVPair("storage", "Default");
  jemit.printKVPair("lifetime", "Scope");
  // jemit.startNamedObject("location");
  // jemit.endObject(); // location

  // jemit.printKVPair("debuginfo", "null", /*stringify=*/false);

  jemit.endObject(); // attributes
  jemit.endObject();
  return success();
}

//===----------------------------------------------------------------------===//
// StreamPopOp
//===----------------------------------------------------------------------===//

LogicalResult translateStreamPopToSDFG(StreamPopOp &op, JsonEmitter &jemit) {
  return success();
}

//===----------------------------------------------------------------------===//
// StreamPushOp
//===----------------------------------------------------------------------===//

LogicalResult translateStreamPushToSDFG(StreamPushOp &op, JsonEmitter &jemit) {
  return success();
}

//===----------------------------------------------------------------------===//
// StreamLengthOp
//===----------------------------------------------------------------------===//

LogicalResult translateStreamLengthToSDFG(StreamLengthOp &op,
                                          JsonEmitter &jemit) {
  return success();
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult printArrayTaskletEdge(LoadOp &load, TaskletNode &task, int argIdx,
                                    JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "MultiConnectorEdge");

  jemit.startNamedObject("attributes");
  jemit.startNamedObject("data");
  jemit.printKVPair("type", "Memlet");
  jemit.startNamedObject("attributes");

  jemit.printKVPair("volume", 1);
  jemit.printKVPair("num_accesses", 1);

  if (GetAccessOp aNode = dyn_cast<GetAccessOp>(load.arr().getDefiningOp())) {
    jemit.printKVPair("data", aNode.getName());
  } else {
    return failure();
  }

  jemit.startNamedObject("subset");
  jemit.printKVPair("type", "Range");
  jemit.startNamedList("ranges");

  if (ArrayAttr syms = load->getAttr("indices").cast<ArrayAttr>()) {
    if (syms.getValue().size() == 0) {
      jemit.startObject();
      jemit.printKVPair("start", 0);
      jemit.printKVPair("end", 0);
      jemit.printKVPair("step", 1);
      jemit.printKVPair("tile", 1);
      jemit.endObject();
    }

    for (Attribute sym : syms.getValue()) {
      if (StringAttr sym_str = sym.cast<StringAttr>()) {
        jemit.startObject();
        jemit.printKVPair("start", sym_str.getValue());
        jemit.printKVPair("end", sym_str.getValue());
        jemit.printKVPair("step", 1);
        jemit.printKVPair("tile", 1);
        jemit.endObject();
      } else {
        return failure();
      }
    }
  } else {
    return failure();
  }

  jemit.endList();   // ranges
  jemit.endObject(); // subset

  jemit.startNamedObject("src_subset");
  jemit.printKVPair("type", "Range");
  jemit.startNamedList("ranges");

  if (ArrayAttr syms = load->getAttr("indices").cast<ArrayAttr>()) {
    if (syms.getValue().size() == 0) {
      jemit.startObject();
      jemit.printKVPair("start", 0);
      jemit.printKVPair("end", 0);
      jemit.printKVPair("step", 1);
      jemit.printKVPair("tile", 1);
      jemit.endObject();
    }

    for (Attribute sym : syms.getValue()) {
      if (StringAttr sym_str = sym.cast<StringAttr>()) {
        jemit.startObject();
        jemit.printKVPair("start", sym_str.getValue());
        jemit.printKVPair("end", sym_str.getValue());
        jemit.printKVPair("step", 1);
        jemit.printKVPair("tile", 1);
        jemit.endObject();
      } else {
        return failure();
      }
    }
  } else {
    return failure();
  }

  jemit.endList();   // ranges
  jemit.endObject(); // src_subset

  jemit.endObject(); // attributes
  jemit.endObject(); // data
  jemit.endObject(); // attributes

  if (GetAccessOp aNode = dyn_cast<GetAccessOp>(load.arr().getDefiningOp())) {
    jemit.printKVPair("src", aNode.ID());
    jemit.printKVPair("src_connector", "null", /*stringify=*/false);
  } else {
    return failure();
  }

  jemit.printKVPair("dst", task.ID());

  std::string argname;
  AsmState state(task);
  BlockArgument bArg = task.getArgument(argIdx);
  llvm::raw_string_ostream argnameStream(argname);
  bArg.printAsOperand(argnameStream, state);

  argname.erase(0, 1); // remove %-sign
  jemit.printKVPair("dst_connector", argname);

  jemit.endObject();
  return success();
}

LogicalResult printTaskletTaskletEdge(TaskletNode &taskSrc,
                                      TaskletNode &taskDest, int argIdx,
                                      JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "MultiConnectorEdge");

  jemit.startNamedObject("attributes");
  jemit.startNamedObject("data");
  jemit.printKVPair("type", "Memlet");
  jemit.startNamedObject("attributes");

  jemit.printKVPair("volume", 1);
  jemit.printKVPair("num_accesses", 1);

  /*if (GetAccessOp aNode = dyn_cast<GetAccessOp>(load.arr().getDefiningOp())) {
    jemit.printKVPair("data", aNode.getName());
  } else {
    return failure();
  }*/

  /*jemit.startNamedObject("subset");
  jemit.printKVPair("type", "Range");
  jemit.startNamedList("ranges");

  if(ArrayAttr syms = load->getAttr("indices").cast<ArrayAttr>()){
    for(Attribute sym : syms.getValue()){
      if (StringAttr sym_str = sym.cast<StringAttr>()) {
        jemit.startObject();
        jemit.printKVPair("start", sym_str.getValue());
        jemit.printKVPair("end", sym_str.getValue());
        jemit.printKVPair("step", 1);
        jemit.printKVPair("tile", 1);
        jemit.endObject();
      } else {
        return failure();
      }
    }
  } else {
    return failure();
  }

  jemit.endList(); // ranges
  jemit.endObject(); // subset
  */

  /*jemit.startNamedObject("src_subset");
  jemit.printKVPair("type", "Range");
  jemit.startNamedList("ranges");

  if(ArrayAttr syms = load->getAttr("indices").cast<ArrayAttr>()){
    for(Attribute sym : syms.getValue()){
      if (StringAttr sym_str = sym.cast<StringAttr>()) {
        jemit.startObject();
        jemit.printKVPair("start", sym_str.getValue());
        jemit.printKVPair("end", sym_str.getValue());
        jemit.printKVPair("step", 1);
        jemit.printKVPair("tile", 1);
        jemit.endObject();
      } else {
        return failure();
      }
    }
  } else {
    return failure();
  }

  jemit.endList(); // ranges
  jemit.endObject(); // src_subset
  */

  jemit.endObject(); // attributes
  jemit.endObject(); // data
  jemit.endObject(); // attributes

  jemit.printKVPair("src", taskSrc.ID());
  jemit.printKVPair("src_connector", "__out");

  jemit.printKVPair("dst", taskDest.ID());

  std::string argname;
  AsmState state(taskDest);
  BlockArgument bArg = taskDest.getArgument(argIdx);
  llvm::raw_string_ostream argnameStream(argname);
  bArg.printAsOperand(argnameStream, state);

  argname.erase(0, 1); // remove %-sign
  jemit.printKVPair("dst_connector", argname);

  jemit.endObject();
  return success();
}

LogicalResult translateCallToSDFG(sdir::CallOp &op, JsonEmitter &jemit) {
  TaskletNode task = op.getTasklet();

  for (unsigned i = 0; i < op.getNumOperands(); ++i) {
    Value val = op.getOperand(i);
    if (LoadOp load = dyn_cast<LoadOp>(val.getDefiningOp())) {
      if (printArrayTaskletEdge(load, task, i, jemit).failed())
        return failure();
    } else if (sdir::CallOp call =
                   dyn_cast<sdir::CallOp>(val.getDefiningOp())) {
      TaskletNode taskSrc = call.getTasklet();
      if (printTaskletTaskletEdge(taskSrc, task, i, jemit).failed())
        return failure();
    } else {
      // llvm::errs() << "here\n";
      return failure();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// LibCallOp
//===----------------------------------------------------------------------===//

LogicalResult translateLibCallToSDFG(LibCallOp &op, JsonEmitter &jemit) {
  return success();
}

//===----------------------------------------------------------------------===//
// AllocSymbolOp
//===----------------------------------------------------------------------===//

LogicalResult translateAllocSymbolToSDFG(AllocSymbolOp &op,
                                         JsonEmitter &jemit) {
  jemit.printKVPair(op.sym(), "int64");
  return success();
}

//===----------------------------------------------------------------------===//
// SymbolExprOp
//===----------------------------------------------------------------------===//

LogicalResult translateSymbolExprToSDFG(SymOp &op, JsonEmitter &jemit) {
  return success();
}

//===----------------------------------------------------------------------===//
// Translate type
//===----------------------------------------------------------------------===//

LogicalResult translateTypeToSDFG(Type &t, JsonEmitter &jemit, StringRef key) {
  if (t.isF64()) {
    jemit.printKVPair(key, "float64");
    return success();
  }

  if (t.isInteger(64)) {
    jemit.printKVPair(key, "int64");
    return success();
  }

  if (t.isInteger(32)) {
    jemit.printKVPair(key, "int32");
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// Op contains Attr
//===----------------------------------------------------------------------===//

bool containsAttr(Operation &op, StringRef attrName) {
  // TODO: Replace with hasAttr
  for (NamedAttribute attr : op.getAttrs())
    if (attr.getName() == attrName)
      return true;
  return false;
}
