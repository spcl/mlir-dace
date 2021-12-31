#include "SDIR/Translate/Translation.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AsmState.h"

using namespace mlir;
using namespace sdir;
using namespace emitter;
using namespace translation;

//===----------------------------------------------------------------------===//
// ModuleOp
//===----------------------------------------------------------------------===//

LogicalResult translation::translateToSDFG(ModuleOp &op, JsonEmitter &jemit) {
  SDIRDialect::resetIDGenerator();

  for (Operation &oper : op.body().getOps())
    if (SDFGNode sdfg = dyn_cast<SDFGNode>(oper))
      if (translateToSDFG(sdfg, jemit).failed())
        return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// SDFGNode
//===----------------------------------------------------------------------===//

LogicalResult printConstant(arith::ConstantOp &op, JsonEmitter &jemit) {
  std::string val;
  llvm::raw_string_ostream valStream(val);
  op.getValue().print(valStream);
  val.erase(val.find(' '));

  AsmState state(op);
  std::string res;
  llvm::raw_string_ostream resStream(res);
  op.getResult().printAsOperand(resStream, state);

  jemit.startNamedList(res);

  jemit.startObject();
  jemit.printKVPair("type", "Scalar");

  jemit.startNamedObject("attributes");

  Type type = op.getType();
  Location loc = op.getLoc();
  StringRef dtype = translateTypeToSDFG(type, loc, jemit);

  if (dtype != "")
    jemit.printKVPair("dtype", dtype);
  else
    return failure();

  jemit.startNamedList("shape");
  jemit.printString("1");
  jemit.endList(); // shape
  jemit.printKVPair("transient", "false", /*stringify=*/false);
  jemit.endObject(); // attributes

  jemit.endObject();

  jemit.startEntry();
  jemit.printLiteral(val);
  jemit.endList(); // res

  return success();
}

LogicalResult printSDFGNode(SDFGNode &op, JsonEmitter &jemit) {
  jemit.printKVPair("type", "SDFG");
  jemit.printKVPair("sdfg_list_id", SDIRDialect::getNextID(),
                    /*stringify=*/false);

  jemit.startNamedObject("attributes");
  jemit.printAttributes(
      op->getAttrs(),
      /*elidedAttrs=*/{"ID", "entry", "sym_name", "type", "arg_names"});

  jemit.startNamedObject("constants_prop");
  for (StateNode state : op.body().getOps<StateNode>())
    for (arith::ConstantOp constOp : state.body().getOps<arith::ConstantOp>())
      if (printConstant(constOp, jemit).failed())
        return failure();

  jemit.endObject(); // constants_prop

  if ((*op).hasAttr("arg_names")) {
    Attribute arg_names = op->getAttr("arg_names");
    if (ArrayAttr arg_names_arr = arg_names.cast<ArrayAttr>()) {
      jemit.startNamedList("arg_names");

      for (Attribute arg_name : arg_names_arr.getValue()) {
        if (StringAttr arg_name_str = arg_name.cast<StringAttr>()) {
          jemit.startEntry();
          jemit.printString(arg_name_str.getValue());
        } else {
          mlir::emitError(op.getLoc(),
                          "'arg_names' must consist of StringAttr");
          return failure();
        }
      }

      jemit.endList(); // arg_names
    } else {
      mlir::emitError(op.getLoc(), "'arg_names' must be an ArrayAttr");
      return failure();
    }
  } else {
    jemit.startNamedList("arg_names");

    for (BlockArgument bArg : op.getArguments()) {
      AsmState state(op);
      std::string name;
      llvm::raw_string_ostream nameStream(name);
      bArg.printAsOperand(nameStream, state);
      name.erase(0, 1); // Remove %-sign
      jemit.startEntry();
      jemit.printString(name);
    }

    jemit.endList(); // arg_names
  }

  jemit.startNamedObject("_arrays");

  for (AllocOp alloc : op.body().getOps<AllocOp>())
    if (translateToSDFG(alloc, jemit).failed())
      return failure();

  for (AllocTransientOp alloc : op.body().getOps<AllocTransientOp>())
    if (translateToSDFG(alloc, jemit).failed())
      return failure();

  for (AllocStreamOp alloc : op.body().getOps<AllocStreamOp>())
    if (translateToSDFG(alloc, jemit).failed())
      return failure();

  for (AllocTransientStreamOp alloc :
       op.body().getOps<AllocTransientStreamOp>())
    if (translateToSDFG(alloc, jemit).failed())
      return failure();

  for (StateNode state : op.body().getOps<StateNode>()) {
    for (AllocOp allocOper : state.body().getOps<AllocOp>())
      if (translateToSDFG(allocOper, jemit).failed())
        return failure();

    for (AllocTransientOp allocOper : state.body().getOps<AllocTransientOp>())
      if (translateToSDFG(allocOper, jemit).failed())
        return failure();

    for (AllocStreamOp allocOper : state.body().getOps<AllocStreamOp>())
      if (translateToSDFG(allocOper, jemit).failed())
        return failure();

    for (AllocTransientStreamOp allocOper :
         state.body().getOps<AllocTransientStreamOp>())
      if (translateToSDFG(allocOper, jemit).failed())
        return failure();
  }

  jemit.endObject(); // _arrays

  jemit.startNamedObject("symbols");
  for (Operation &oper : op.body().getOps()) {
    if (AllocSymbolOp alloc = dyn_cast<AllocSymbolOp>(oper))
      if (translateToSDFG(alloc, jemit).failed())
        return failure();
  }

  jemit.endObject(); // symbols

  if (!(*op).hasAttr("instrument"))
    jemit.printKVPair("instrument", "No_Instrumentation");

  jemit.printKVPair("name", op.sym_name());
  jemit.endObject(); // attributes

  jemit.startNamedList("nodes");

  unsigned stateID = 0;

  for (Operation &oper : op.body().getOps())
    if (StateNode state = dyn_cast<StateNode>(oper)) {
      state.setID(stateID);
      if (translateToSDFG(state, jemit).failed())
        return failure();
      stateID++;
    }

  jemit.endList(); // nodes

  jemit.startNamedList("edges");

  for (Operation &oper : op.body().getOps())
    if (EdgeOp edge = dyn_cast<EdgeOp>(oper))
      if (translateToSDFG(edge, jemit).failed())
        return failure();

  jemit.endList(); // edges

  StateNode entryState = op.getStateBySymRef(op.entry());
  jemit.printKVPair("start_state", entryState.ID(), /*stringify=*/false);

  return success();
}

LogicalResult translation::translateToSDFG(SDFGNode &op, JsonEmitter &jemit) {
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

  if (!(*op).hasAttr("instrument"))
    jemit.printKVPair("instrument", "No_Instrumentation");

  if (!(*op).hasAttr("schedule"))
    jemit.printKVPair("schedule", "Default");

  jemit.startNamedObject("in_connectors");
  for (BlockArgument bArg : op.getArguments()) {
    AsmState state(op);
    std::string name;
    llvm::raw_string_ostream nameStream(name);
    bArg.printAsOperand(nameStream, state);
    name.erase(0, 1); // Remove %-sign
    jemit.printKVPair(name, "null", /*stringify=*/false);
  }
  jemit.endObject(); // in_connectors

  jemit.startNamedObject("out_connectors");
  // TODO: Print out_connectors
  jemit.printKVPair("__return", "null", /*stringify=*/false);
  jemit.endObject(); // out_connectors

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

LogicalResult translation::translateToSDFG(StateNode &op, JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "SDFGState");
  jemit.printKVPair("label", op.sym_name());
  jemit.printKVPair("id", op.ID(), /*stringify=*/false);

  jemit.startNamedObject("attributes");
  jemit.printAttributes(op->getAttrs(), /*elidedAttrs=*/{"ID", "sym_name"});
  if (!(*op).hasAttr("instrument"))
    jemit.printKVPair("instrument", "No_Instrumentation");

  jemit.endObject(); // attributes

  jemit.startNamedList("nodes");

  unsigned nodeID = 0;
  for (Operation &oper : op.body().getOps()) {
    if (TaskletNode tasklet = dyn_cast<TaskletNode>(oper)) {
      tasklet.setID(nodeID++);
      if (translateToSDFG(tasklet, jemit).failed())
        return failure();
    }

    if (SDFGNode sdfg = dyn_cast<SDFGNode>(oper)) {
      sdfg.setID(nodeID++);
      if (translateToSDFG(sdfg, jemit).failed())
        return failure();
    }

    if (GetAccessOp acc = dyn_cast<GetAccessOp>(oper)) {
      acc.setID(nodeID++);
      if (translateToSDFG(acc, jemit).failed())
        return failure();
    }

    if (MapNode map = dyn_cast<MapNode>(oper)) {
      map.setEntryID(nodeID++);
      map.setExitID(nodeID++);
      if (translateToSDFG(map, jemit).failed())
        return failure();
    }

    if (ConsumeNode consume = dyn_cast<ConsumeNode>(oper)) {
      consume.setEntryID(nodeID++);
      consume.setExitID(nodeID++);
      if (translateToSDFG(consume, jemit).failed())
        return failure();
    }
  }

  jemit.endList(); // nodes

  jemit.startNamedList("edges");

  for (Operation &oper : op.body().getOps()) {
    if (CopyOp edge = dyn_cast<CopyOp>(oper))
      if (translateToSDFG(edge, jemit).failed())
        return failure();

    if (StoreOp edge = dyn_cast<StoreOp>(oper))
      if (translateToSDFG(edge, jemit).failed())
        return failure();

    if (sdir::CallOp edge = dyn_cast<sdir::CallOp>(oper))
      if (translateToSDFG(edge, jemit).failed())
        return failure();
  }

  jemit.endList(); // edges

  jemit.endObject();
  return success();
}

//===----------------------------------------------------------------------===//
// TaskletNode
//===----------------------------------------------------------------------===//

// TODO(later): Temporary auto-lifting. Will be included into DaCe
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

  if (dyn_cast<arith::AddFOp>(firstOp) || dyn_cast<arith::AddIOp>(firstOp)) {
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

  if (dyn_cast<arith::MulFOp>(firstOp) || dyn_cast<arith::MulIOp>(firstOp)) {
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

  if (arith::ConstantOp oper = dyn_cast<arith::ConstantOp>(firstOp)) {
    Type t = oper.value().getType();
    Location loc = oper.getLoc();
    StringRef type = translateTypeToSDFG(t, loc, jemit);
    std::string val;

    if (arith::ConstantFloatOp flop =
            dyn_cast<arith::ConstantFloatOp>(firstOp)) {
      SmallVector<char> flopVec;
      flop.value().toString(flopVec);
      for (char c : flopVec)
        val += c;
    } else if (arith::ConstantIntOp iop =
                   dyn_cast<arith::ConstantIntOp>(firstOp)) {
      val = std::to_string(iop.value());
    }

    std::string entry = "__out = dace." + type.str() + "(" + val + ")";
    jemit.printKVPair("string_data", entry);
    jemit.printKVPair("language", "Python");
    return success();
  }

  return failure();
}

LogicalResult translation::translateToSDFG(TaskletNode &op,
                                           JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "Tasklet");
  jemit.printKVPair("label", op.sym_name());

  jemit.startNamedObject("attributes");
  jemit.printKVPair("label", op.sym_name());

  if (!(*op).hasAttr("instrument"))
    jemit.printKVPair("instrument", "No_Instrumentation");

  jemit.startNamedObject("code");

  AsmState state(op);

  // Try to lift the body of the tasklet
  // If lifting fails (body is complex) then emit MLIR code directly
  // liftToPython() emits automatically emits the generated python code
  if (liftToPython(op, jemit).failed()) {
    // Convention: MLIR tasklets use the mlir_entry function as the entry point
    std::string code = "module {\\n func @mlir_entry(";

    // Prints all arguments with types
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

    // Emits the body of the tasklet
    for (Operation &oper : op.body().getOps()) {
      std::string codeLine;
      llvm::raw_string_ostream codeLineStream(codeLine);
      oper.print(codeLineStream);

      // SDIR is not a core dialect. Therefore "sdir.return" does not exist
      // Replace it with the standard "return"
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

LogicalResult translation::translateToSDFG(MapNode &op, JsonEmitter &jemit) {
  // MapEntry
  jemit.startObject();
  jemit.printKVPair("type", "MapEntry");

  jemit.startNamedObject("attributes");
  if (!(*op).hasAttr("instrument"))
    jemit.printKVPair("instrument", "No_Instrumentation");
  if (!(*op).hasAttr("schedule"))
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

LogicalResult translation::translateToSDFG(ConsumeNode &op,
                                           JsonEmitter &jemit) {
  // ConsumeEntry
  jemit.startObject();
  jemit.printKVPair("type", "ConsumeEntry");

  jemit.startNamedObject("attributes");
  if (!(*op).hasAttr("instrument"))
    jemit.printKVPair("instrument", "No_Instrumentation");
  if (!(*op).hasAttr("schedule"))
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

LogicalResult translation::translateToSDFG(EdgeOp &op, JsonEmitter &jemit) {
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
        mlir::emitError(
            op.getLoc(),
            "'assign' must be an ArrayAttr consisting of StringAttr");
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
  Location loc = op.getLoc();
  StringRef dtype = translateTypeToSDFG(element, loc, jemit);

  if (dtype != "")
    jemit.printKVPair("dtype", dtype);
  else
    return failure();

  jemit.startNamedList("shape");
  jemit.startEntry();
  jemit.printInt(1);
  jemit.endList(); // shape

  jemit.printKVPair("transient", "false", /*stringify=*/false);
  if (!(*op).hasAttr("storage"))
    jemit.printKVPair("storage", "Default");
  if (!(*op).hasAttr("lifetime"))
    jemit.printKVPair("lifetime", "Scope");
  printDebuginfo(*op, jemit);

  jemit.endObject(); // attributes

  jemit.endObject();
  return success();
}

LogicalResult translation::translateToSDFG(AllocOp &op, JsonEmitter &jemit) {
  if (op.getType().getShape().size() == 0)
    return printScalar(op, jemit);

  jemit.startNamedObject(op.getName());
  jemit.printKVPair("type", "Array");

  jemit.startNamedObject("attributes");
  jemit.startNamedList("strides");
  ArrayRef<int64_t> shape = op.getType().getIntegers();

  for (int i = shape.size() - 1; i > 0; --i) {
    jemit.startEntry();
    jemit.printInt(shape[i]);
  }

  jemit.startEntry();
  jemit.printInt(1);
  jemit.endList(); // strides

  Type element = op.getType().getElementType();
  Location loc = op.getLoc();
  StringRef dtype = translateTypeToSDFG(element, loc, jemit);

  if (dtype != "")
    jemit.printKVPair("dtype", dtype);
  else
    return failure();

  jemit.startNamedList("shape");
  for (int64_t s : shape) {
    jemit.startEntry();
    jemit.printInt(s);
  }
  jemit.endList(); // shape

  jemit.printKVPair("transient", "false", /*stringify=*/false);
  if (!(*op).hasAttr("storage"))
    jemit.printKVPair("storage", "Default");
  if (!(*op).hasAttr("lifetime"))
    jemit.printKVPair("lifetime", "Scope");
  printDebuginfo(*op, jemit);

  jemit.endObject(); // attributes
  jemit.endObject();
  return success();
}

//===----------------------------------------------------------------------===//
// AllocTransientOp
//===----------------------------------------------------------------------===//

LogicalResult translation::translateToSDFG(AllocTransientOp &op,
                                           JsonEmitter &jemit) {
  AsmState state(op.getParentSDFG());
  std::string name;
  llvm::raw_string_ostream nameStream(name);
  op->getResult(0).printAsOperand(nameStream, state);

  jemit.startNamedObject(name);
  jemit.printKVPair("type", "Array");

  jemit.startNamedObject("attributes");

  Type element = op.getType().getElementType();
  Location loc = op.getLoc();
  StringRef dtype = translateTypeToSDFG(element, loc, jemit);

  if (dtype != "")
    jemit.printKVPair("dtype", dtype);
  else
    return failure();

  jemit.printKVPair("transient", "true", /*stringify=*/false);
  if (!(*op).hasAttr("storage"))
    jemit.printKVPair("storage", "Default");
  if (!(*op).hasAttr("lifetime"))
    jemit.printKVPair("lifetime", "Scope");
  printDebuginfo(*op, jemit);

  jemit.endObject(); // attributes
  jemit.endObject();
  return success();
}

//===----------------------------------------------------------------------===//
// GetAccessOp
//===----------------------------------------------------------------------===//

LogicalResult translation::translateToSDFG(GetAccessOp &op,
                                           JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "AccessNode");
  jemit.printKVPair("label", op.getName());

  jemit.startNamedObject("attributes");
  if (!(*op).hasAttr("access"))
    jemit.printKVPair("access", "ReadWrite");
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

LogicalResult translation::translateToSDFG(LoadOp &op, JsonEmitter &jemit) {
  // TODO: Implement translateLoadToSDFG
  return success();
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

LogicalResult translation::translateToSDFG(StoreOp &op, JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "MultiConnectorEdge");

  jemit.startNamedObject("attributes");
  jemit.startNamedObject("data");
  jemit.printKVPair("type", "Memlet");
  jemit.startNamedObject("attributes");

  jemit.startNamedList("strides");
  ArrayRef<int64_t> shape = op.arr().getType().cast<MemletType>().getIntegers();
  for (int i = shape.size() - 1; i > 0; --i) {
    jemit.startEntry();
    jemit.printInt(shape[i]);
  }

  jemit.startEntry();
  jemit.printInt(1);
  jemit.endList(); // strides

  if (GetAccessOp aNode = dyn_cast<GetAccessOp>(op.arr().getDefiningOp())) {
    jemit.printKVPair("data", aNode.getName());
  } else {
    mlir::emitError(op.getLoc(), "Array must be defined by a GetAccessOp");
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
        mlir::emitError(op.getLoc(), "'indices' must consist of StringAttr");
        return failure();
      }
    }
  } else {
    mlir::emitError(op.getLoc(), "'indices' must be an ArrayAttr");
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
        mlir::emitError(op.getLoc(), "'indices' must consist of StringAttr");
        return failure();
      }
    }
  } else {
    mlir::emitError(op.getLoc(), "'indices' must be an ArrayAttr");
    return failure();
  }

  jemit.endList();   // ranges
  jemit.endObject(); // dst_subset

  jemit.endObject(); // attributes
  jemit.endObject(); // data
  jemit.endObject(); // attributes

  // Get the ID of the tasklet if this StoreOp represents
  // a tasklet -> access node edge
  if (sdir::CallOp call = dyn_cast<sdir::CallOp>(op.val().getDefiningOp())) {
    TaskletNode aNode = call.getTasklet();
    jemit.printKVPair("src", aNode.ID());
    jemit.printKVPair("src_connector", "__out");
  } else if (arith::ConstantOp con =
                 dyn_cast<arith::ConstantOp>(op.val().getDefiningOp())) {
    // TODO: Add constants
  } else {
    mlir::emitError(op.getLoc(),
                    "Value must be result of TaskletNode or ConstantOp");
    return failure();
  }

  if (GetAccessOp aNode = dyn_cast<GetAccessOp>(op.arr().getDefiningOp())) {
    jemit.printKVPair("dst", aNode.ID());
    jemit.printKVPair("dst_connector", "null", /*stringify=*/false);
  } else {
    mlir::emitError(op.getLoc(), "Array must be defined by a GetAccessOp");
    return failure();
  }

  jemit.endObject();
  return success();
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

LogicalResult translation::translateToSDFG(CopyOp &op, JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "MultiConnectorEdge");

  jemit.startNamedObject("attributes");
  jemit.startNamedObject("data");
  jemit.printKVPair("type", "Memlet");
  jemit.startNamedObject("attributes");

  if (GetAccessOp aNode = dyn_cast<GetAccessOp>(op.src().getDefiningOp())) {
    jemit.printKVPair("data", aNode.getName());
  } else {
    mlir::emitError(op.getLoc(),
                    "Source array must be defined by a GetAccessOp");
    return failure();
  }

  jemit.endObject(); // attributes
  jemit.endObject(); // data
  jemit.endObject(); // attributes

  if (GetAccessOp aNode = dyn_cast<GetAccessOp>(op.src().getDefiningOp())) {
    jemit.printKVPair("src", aNode.ID());
    jemit.printKVPair("src_connector", "null", /*stringify=*/false);
  } else {
    mlir::emitError(op.getLoc(),
                    "Source array must be defined by a GetAccessOp");
    return failure();
  }

  if (GetAccessOp aNode = dyn_cast<GetAccessOp>(op.dest().getDefiningOp())) {
    jemit.printKVPair("dst", aNode.ID());
    jemit.printKVPair("dst_connector", "null", /*stringify=*/false);
  } else {
    mlir::emitError(op.getLoc(),
                    "Destination array must be defined by a GetAccessOp");
    return failure();
  }

  jemit.endObject();
  return success();
}

//===----------------------------------------------------------------------===//
// MemletCastOp
//===----------------------------------------------------------------------===//

LogicalResult translation::translateToSDFG(MemletCastOp &op,
                                           JsonEmitter &jemit) {
  // TODO: Implement translateMemletCastToSDFG
  return success();
}

//===----------------------------------------------------------------------===//
// ViewCastOp
//===----------------------------------------------------------------------===//

LogicalResult translation::translateToSDFG(ViewCastOp &op, JsonEmitter &jemit) {
  // TODO: Implement translateViewCastToSDFG
  return success();
}

//===----------------------------------------------------------------------===//
// SubviewOp
//===----------------------------------------------------------------------===//

LogicalResult translation::translateToSDFG(SubviewOp &op, JsonEmitter &jemit) {
  // TODO: Implement translateSubviewToSDFG
  return success();
}

//===----------------------------------------------------------------------===//
// AllocStreamOp
//===----------------------------------------------------------------------===//

LogicalResult translation::translateToSDFG(AllocStreamOp &op,
                                           JsonEmitter &jemit) {
  AsmState state(op.getParentSDFG());
  std::string name;
  llvm::raw_string_ostream nameStream(name);
  op->getResult(0).printAsOperand(nameStream, state);

  jemit.startNamedObject(name);
  jemit.printKVPair("type", "Stream");

  jemit.startNamedObject("attributes");

  Type type = op.getType().getElementType();
  Location loc = op.getLoc();
  StringRef dtype = translateTypeToSDFG(type, loc, jemit);

  if (dtype != "")
    jemit.printKVPair("dtype", dtype);
  else
    return failure();

  jemit.printKVPair("transient", "false", /*stringify=*/false);
  if (!(*op).hasAttr("storage"))
    jemit.printKVPair("storage", "Default");
  if (!(*op).hasAttr("lifetime"))
    jemit.printKVPair("lifetime", "Scope");
  printDebuginfo(*op, jemit);

  jemit.endObject(); // attributes
  jemit.endObject();
  return success();
}

//===----------------------------------------------------------------------===//
// AllocTransientStreamOp
//===----------------------------------------------------------------------===//

LogicalResult translation::translateToSDFG(AllocTransientStreamOp &op,
                                           JsonEmitter &jemit) {
  AsmState state(op.getParentSDFG());
  std::string name;
  llvm::raw_string_ostream nameStream(name);
  op->getResult(0).printAsOperand(nameStream, state);

  jemit.startNamedObject(name);
  jemit.printKVPair("type", "Stream");

  jemit.startNamedObject("attributes");

  Type type = op.getType().getElementType();
  Location loc = op.getLoc();
  StringRef dtype = translateTypeToSDFG(type, loc, jemit);

  if (dtype != "")
    jemit.printKVPair("dtype", dtype);
  else
    return failure();

  jemit.printKVPair("transient", "true", /*stringify=*/false);
  if (!(*op).hasAttr("storage"))
    jemit.printKVPair("storage", "Default");
  if (!(*op).hasAttr("lifetime"))
    jemit.printKVPair("lifetime", "Scope");
  printDebuginfo(*op, jemit);

  jemit.endObject(); // attributes
  jemit.endObject();
  return success();
}

//===----------------------------------------------------------------------===//
// StreamPopOp
//===----------------------------------------------------------------------===//

LogicalResult translation::translateToSDFG(StreamPopOp &op,
                                           JsonEmitter &jemit) {
  // TODO: Implement translateStreamPopToSDFG
  return success();
}

//===----------------------------------------------------------------------===//
// StreamPushOp
//===----------------------------------------------------------------------===//

LogicalResult translation::translateToSDFG(StreamPushOp &op,
                                           JsonEmitter &jemit) {
  // TODO: Implement translateStreamPushToSDFG
  return success();
}

//===----------------------------------------------------------------------===//
// StreamLengthOp
//===----------------------------------------------------------------------===//

LogicalResult translation::translateToSDFG(StreamLengthOp &op,
                                           JsonEmitter &jemit) {
  // TODO: Implement translateStreamLengthToSDFG
  return success();
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult printLoadTaskletEdge(LoadOp &load, TaskletNode &task, int argIdx,
                                   JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "MultiConnectorEdge");

  jemit.startNamedObject("attributes");
  jemit.startNamedObject("data");
  jemit.printKVPair("type", "Memlet");
  jemit.startNamedObject("attributes");

  jemit.startNamedList("strides");
  ArrayRef<int64_t> shape =
      load.arr().getType().cast<MemletType>().getIntegers();
  for (int i = shape.size() - 1; i > 0; --i) {
    jemit.startEntry();
    jemit.printInt(shape[i]);
  }

  jemit.startEntry();
  jemit.printInt(1);
  jemit.endList(); // strides

  if (GetAccessOp aNode = dyn_cast<GetAccessOp>(load.arr().getDefiningOp())) {
    jemit.printKVPair("data", aNode.getName());
  } else {
    mlir::emitError(load.getLoc(), "Array must be defined by a GetAccessOp");
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
        mlir::emitError(load.getLoc(), "'indices' must consist of StringAttr");
        return failure();
      }
    }
  } else {
    mlir::emitError(load.getLoc(), "'indices' must be an ArrayAttr");
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
        mlir::emitError(load.getLoc(), "'indices' must consist of StringAttr");
        return failure();
      }
    }
  } else {
    mlir::emitError(load.getLoc(), "'indices' must be an ArrayAttr");
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
    mlir::emitError(load.getLoc(), "Array must be defined by a GetAccessOp");
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

LogicalResult printAccessSDFGEdge(GetAccessOp &access, SDFGNode &sdfg,
                                  int argIdx, JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "MultiConnectorEdge");

  jemit.startNamedObject("attributes");
  jemit.startNamedObject("data");
  jemit.printKVPair("type", "Memlet");
  jemit.startNamedObject("attributes");
  jemit.printKVPair("data", access.getName());
  jemit.endObject(); // attributes
  jemit.endObject(); // data
  jemit.endObject(); // attributes
  jemit.printKVPair("src", access.ID());
  jemit.printKVPair("src_connector", "null", /*stringify=*/false);
  jemit.printKVPair("dst", sdfg.ID());

  std::string argname;
  AsmState state(sdfg);
  BlockArgument bArg = sdfg.getArgument(argIdx);
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

LogicalResult translation::translateToSDFG(sdir::CallOp &op,
                                           JsonEmitter &jemit) {
  if (op.callsTasklet()) {
    TaskletNode task = op.getTasklet();
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      Value val = op.getOperand(i);
      if (LoadOp load = dyn_cast<LoadOp>(val.getDefiningOp())) {
        if (printLoadTaskletEdge(load, task, i, jemit).failed())
          return failure();
      } else if (sdir::CallOp call =
                     dyn_cast<sdir::CallOp>(val.getDefiningOp())) {
        TaskletNode taskSrc = call.getTasklet();
        if (printTaskletTaskletEdge(taskSrc, task, i, jemit).failed())
          return failure();
      } else if (arith::ConstantOp con =
                     dyn_cast<arith::ConstantOp>(val.getDefiningOp())) {
        // TODO: Add constants
      } else {
        mlir::emitError(op.getLoc(), "Operands must be results of GetAccessOp, "
                                     "LoadOp, TaskletNode or ConstantOp");
        return failure();
      }
    }
  } else {
    // calls nested SDFG
    SDFGNode sdfg = op.getSDFG();
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      Value val = op.getOperand(i);
      if (GetAccessOp acc = dyn_cast<GetAccessOp>(val.getDefiningOp())) {
        if (printAccessSDFGEdge(acc, sdfg, i, jemit).failed())
          return failure();
      } else {
        mlir::emitError(op.getLoc(),
                        "Operands must be results of GetAccessOp, LoadOp, "
                        "TaskletNode or ConstantOp");
        return failure();
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// LibCallOp
//===----------------------------------------------------------------------===//

LogicalResult translation::translateToSDFG(LibCallOp &op, JsonEmitter &jemit) {
  // TODO: Implement translateLibCallToSDFG
  return success();
}

//===----------------------------------------------------------------------===//
// AllocSymbolOp
//===----------------------------------------------------------------------===//

LogicalResult translation::translateToSDFG(AllocSymbolOp &op,
                                           JsonEmitter &jemit) {
  jemit.printKVPair(op.sym(), "int64");
  return success();
}

//===----------------------------------------------------------------------===//
// SymbolExprOp
//===----------------------------------------------------------------------===//

LogicalResult translation::translateToSDFG(SymOp &op, JsonEmitter &jemit) {
  // TODO: Implement translateSymbolExprToSDFG
  return success();
}

//===----------------------------------------------------------------------===//
// Translate type
//===----------------------------------------------------------------------===//

StringRef translation::translateTypeToSDFG(Type &t, Location &loc,
                                           JsonEmitter &jemit) {
  if (t.isF64())
    return "float64";

  if (t.isF32())
    return "float32";

  if (t.isInteger(64))
    return "int64";

  if (t.isInteger(32))
    return "int32";

  if (t.isIndex())
    return "int64";

  std::string type;
  llvm::raw_string_ostream typeStream(type);
  t.print(typeStream);
  mlir::emitError(loc, "Unsupported type: " + type);

  return "";
}

//===----------------------------------------------------------------------===//
// Print debuginfo
//===----------------------------------------------------------------------===//

inline void translation::printDebuginfo(Operation &op, JsonEmitter &jemit) {
  std::string loc;
  llvm::raw_string_ostream locStream(loc);
  op.getLoc().print(locStream);
  remove(loc.begin(), loc.end(), '\"');
  jemit.printKVPair("debuginfo", loc);
}
