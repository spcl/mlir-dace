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

  if (sdir::ReturnOp Op = dyn_cast<sdir::ReturnOp>(op))
    return translateReturnToSDFG(Op, jemit);

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
  for (Operation &oper : op.body().getOps())
    if (translateToSDFG(oper, jemit).failed())
      return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// SDFGNode
//===----------------------------------------------------------------------===//

LogicalResult printSDFGNode(SDFGNode &op, JsonEmitter &jemit) {
  jemit.printKVPair("type", "SDFG");

  jemit.startNamedObject("attributes");
  jemit.printAttributes(op->getAttrs(),
                        /*elidedAttrs=*/{"ID", "entry", "sym_name", "type"});

  jemit.startNamedObject("constants_prop");
  // TODO: Fill this out
  jemit.endObject(); // constants_prop

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
  jemit.endObject(); // _arrays

  if (!containsAttr(*op, "instrument"))
    jemit.printKVPair("instrument", "No_Instrumentation");
  jemit.printKVPair("name", op.sym_name());
  jemit.endObject(); // attributes

  jemit.startNamedList("nodes");

  for (Operation &oper : op.body().getOps()) {
    // Skip edges to print them in "edges"
    if (EdgeOp edge = dyn_cast<EdgeOp>(oper))
      continue;

    // Skip allocs because they're already printed
    if (AllocOp alloc = dyn_cast<AllocOp>(oper))
      continue;

    if (AllocTransientOp alloc = dyn_cast<AllocTransientOp>(oper))
      continue;

    if (AllocStreamOp alloc = dyn_cast<AllocStreamOp>(oper))
      continue;

    if (AllocTransientStreamOp alloc = dyn_cast<AllocTransientStreamOp>(oper))
      continue;

    if (translateToSDFG(oper, jemit).failed())
      return failure();
  }

  jemit.endList(); // nodes

  jemit.startNamedList("edges");

  for (Operation &oper : op.body().getOps())
    if (EdgeOp edge = dyn_cast<EdgeOp>(oper))
      if (translateToSDFG(*edge, jemit).failed())
        return failure();

  jemit.endList(); // edges

  jemit.startNamedObject("_arrays");
  jemit.endObject();

  jemit.printKVPair("sdfg_list_id", op.ID(), /*stringify=*/false);

  StateNode entryState = op.getStateBySymRef(op.entry());
  unsigned start_state_idx = op.getIndexOfState(entryState);
  jemit.printKVPair("start_state", start_state_idx, /*stringify=*/false);

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

  SDFGNode sdfg = dyn_cast<SDFGNode>(op->getParentOp());
  // jemit.printKVPair("id", sdfg.getIndexOfState(op), /*stringify=*/false);
  jemit.printKVPair("id", op.ID(), /*stringify=*/false);

  jemit.startNamedObject("attributes");
  jemit.printAttributes(op->getAttrs(), /*elidedAttrs=*/{"ID", "sym_name"});
  if (!containsAttr(*op, "instrument"))
    jemit.printKVPair("instrument", "No_Instrumentation");
  jemit.endObject(); // attributes

  jemit.startNamedList("nodes");

  for (Operation &oper : op.body().getOps()) {
    // Skip allocs
    if (AllocOp alloc = dyn_cast<AllocOp>(oper))
      continue;

    if (AllocTransientOp alloc = dyn_cast<AllocTransientOp>(oper))
      continue;

    if (AllocStreamOp alloc = dyn_cast<AllocStreamOp>(oper))
      continue;

    if (AllocTransientStreamOp alloc = dyn_cast<AllocTransientStreamOp>(oper))
      continue;

    if (translateToSDFG(oper, jemit).failed())
      return failure();
  }

  jemit.endList(); // nodes

  jemit.startNamedList("edges");
  // TODO: Fill edges
  jemit.endList(); // edges

  jemit.endObject();
  return success();
}

//===----------------------------------------------------------------------===//
// TaskletNode
//===----------------------------------------------------------------------===//

LogicalResult translateTaskletToSDFG(TaskletNode &op, JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "Tasklet");
  jemit.printKVPair("label", op.sym_name());

  jemit.startNamedObject("attributes");
  if (!containsAttr(*op, "instrument"))
    jemit.printKVPair("instrument", "No_Instrumentation");

  jemit.startNamedObject("code");
  jemit.printKVPair("string_data", op.body());
  jemit.printKVPair("language", "MLIR");
  jemit.endObject(); // code

  jemit.startNamedObject("in_connectors");

  AsmState state(op);
  for (BlockArgument bArg : op.getArguments()) {
    std::string name;
    llvm::raw_string_ostream nameStream(name);
    bArg.printAsOperand(nameStream, state);
    jemit.printKVPair(name, "null", /*stringify=*/false);
    // bArg.getType().print(jemit.ostream());
  }

  jemit.endObject(); // in_connectors

  jemit.startNamedObject("out_connectors");
  // TODO: print out_connectors
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
  // TODO: Fill in the assignments
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

LogicalResult translateAllocToSDFG(AllocOp &op, JsonEmitter &jemit) {
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

  AsmState state(op.getParentSDFG());
  std::string name;
  llvm::raw_string_ostream nameStream(name);
  op.arr().printAsOperand(nameStream, state);
  jemit.printKVPair("label", name);

  jemit.startNamedObject("attributes");
  jemit.printKVPair("access", "ReadWrite");
  jemit.printKVPair("setzero", "false", /*stringify=*/false);
  jemit.printKVPair("data", name);
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
  return success();
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

LogicalResult translateCopyToSDFG(CopyOp &op, JsonEmitter &jemit) {
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
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult translateReturnToSDFG(sdir::ReturnOp &op, JsonEmitter &jemit) {
  return success();
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult translateCallToSDFG(sdir::CallOp &op, JsonEmitter &jemit) {
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
  return success();
}

//===----------------------------------------------------------------------===//
// SymbolExprOp
//===----------------------------------------------------------------------===//

LogicalResult translateSymbolExprToSDFG(SymOp &op, JsonEmitter &jemit) {
  return success();
}

//===----------------------------------------------------------------------===//
// Op contains Attr
//===----------------------------------------------------------------------===//

bool containsAttr(Operation &op, StringRef attrName) {
  for (NamedAttribute attr : op.getAttrs())
    if (attr.first == attrName)
      return true;
  return false;
}
