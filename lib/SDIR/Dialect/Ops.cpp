#include "SDIR/Dialect/Dialect.h"
#include "SDIR/Utils/Utils.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;
using namespace sdir;

//===----------------------------------------------------------------------===//
// InlineSymbol
//===----------------------------------------------------------------------===//

static ParseResult parseNumberList(OpAsmParser &parser, OperationState &result,
                                   StringRef attrName) {
  SmallVector<OpAsmParser::OperandType> opList;
  SmallVector<Attribute> attrList;
  SmallVector<Attribute> numList;
  int opIdx = result.operands.size();

  do {
    if (parser.parseOptionalKeyword("sym").succeeded()) {
      StringAttr stringAttr;
      if (parser.parseLParen() ||
          parser.parseAttribute(stringAttr,
                                parser.getBuilder().getNoneType()) ||
          parser.parseRParen())
        return failure();

      attrList.push_back(stringAttr);
      numList.push_back(parser.getBuilder().getI32IntegerAttr(-1));
      continue;
    }

    int32_t num;
    OptionalParseResult intOPR = parser.parseOptionalInteger(num);
    if (intOPR.hasValue() && intOPR.getValue().succeeded()) {
      Attribute intAttr = parser.getBuilder().getI32IntegerAttr(num);
      attrList.push_back(intAttr);
      numList.push_back(parser.getBuilder().getI32IntegerAttr(-1));
      continue;
    }

    OpAsmParser::OperandType op;
    OptionalParseResult opOPR = parser.parseOptionalOperand(op);
    if (opOPR.hasValue() && opOPR.getValue().succeeded()) {
      opList.push_back(op);
      numList.push_back(parser.getBuilder().getUI32IntegerAttr(opIdx++));
      continue;
    }

    if (parser.parseOptionalComma().succeeded())
      return failure();

  } while (parser.parseOptionalComma().succeeded());

  ArrayAttr attrArr = parser.getBuilder().getArrayAttr(attrList);
  result.addAttribute(attrName, attrArr);

  SmallVector<Value> valList;
  parser.resolveOperands(opList, parser.getBuilder().getIndexType(), valList);
  result.addOperands(valList);

  ArrayAttr numArr = parser.getBuilder().getArrayAttr(numList);
  result.addAttribute(attrName.str() + "_numList", numArr);

  return success();
}

static void printNumberList(OpAsmPrinter &p, Operation *op,
                            StringRef attrName) {
  ArrayAttr attrList = op->getAttr(attrName).cast<ArrayAttr>();
  ArrayAttr numList =
      op->getAttr(attrName.str() + "_numList").cast<ArrayAttr>();

  for (unsigned i = 0, attri = 0; i < numList.size(); ++i) {
    Attribute numAttr = numList[i];
    IntegerAttr num = numAttr.cast<IntegerAttr>();
    if (i > 0)
      p << ", ";

    if (num.getValue().isNegative()) {
      Attribute attr = attrList[attri++];

      if (attr.isa<StringAttr>()) {
        p << "sym(" << attr << ")";
      } else
        p.printAttributeWithoutType(attr);

    } else {
      unsigned idx = num.getUInt();
      Value val = op->getOperand(idx);
      p.printOperand(val);
    }
  }
}

static void
printOptionalAttrDictNoNumList(OpAsmPrinter &p, ArrayRef<NamedAttribute> attrs,
                               ArrayRef<StringRef> elidedAttrs = {}) {
  SmallVector<StringRef> numListAttrs(elidedAttrs.begin(), elidedAttrs.end());

  for (NamedAttribute na : attrs)
    if (na.getName().strref().endswith("numList"))
      numListAttrs.push_back(na.getName().strref());

  p.printOptionalAttrDict(attrs, /*elidedAttrs=*/numListAttrs);
}

static size_t getNumListSize(Operation *op, StringRef attrName) {
  ArrayAttr numList =
      op->getAttr(attrName.str() + "_numList").cast<ArrayAttr>();
  return numList.size();
}

//===----------------------------------------------------------------------===//
// SDFGNode
//===----------------------------------------------------------------------===//

SDFGNode SDFGNode::create(PatternRewriter &rewriter, Location loc,
                          FunctionType ft) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());
  build(builder, state, utils::generateID(), utils::generateName("sdfg"),
        "state_0", ft);
  SDFGNode sdfg = cast<SDFGNode>(rewriter.createOperation(state));
  rewriter.createBlock(&sdfg.getRegion(), {}, ft.getInputs());
  return sdfg;
}

SDFGNode SDFGNode::create(PatternRewriter &rewriter, Location loc,
                          FunctionType ft, StringRef name) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());
  build(builder, state, utils::generateID(), name, "state_0", ft);
  SDFGNode sdfg = cast<SDFGNode>(rewriter.createOperation(state));
  rewriter.createBlock(&sdfg.getRegion(), {}, ft.getInputs());
  return sdfg;
}

static ParseResult parseSDFGNode(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  IntegerAttr intAttr =
      parser.getBuilder().getI32IntegerAttr(utils::generateID());
  result.addAttribute("ID", intAttr);

  StringAttr sym_nameAttr;
  if (parser.parseSymbolName(sym_nameAttr, "sym_name", result.attributes))
    return failure();

  SmallVector<OpAsmParser::OperandType, 4> entryArgs;
  SmallVector<NamedAttrList, 4> argAttrs;
  SmallVector<NamedAttrList, 4> resultAttrs;
  SmallVector<Type, 4> argTypes;
  SmallVector<Type, 4> resultTypes;

  Region *body = result.addRegion();
  OptionalParseResult opr = parser.parseOptionalRegion(*body);
  if (opr.hasValue() && opr.getValue().succeeded()) {
    if (body->empty())
      body->emplaceBlock();

    Type type = parser.getBuilder().getFunctionType(argTypes, resultTypes);
    if (!type)
      return failure();
    result.addAttribute(SDFGNode::getTypeAttrName(), TypeAttr::get(type));

    return success();
  }

  bool isVariadic = false;
  if (function_like_impl::parseFunctionSignature(
          parser,
          /*allowVariadic=*/false, entryArgs, argTypes, argAttrs, isVariadic,
          resultTypes, resultAttrs))
    return failure();

  Type type = parser.getBuilder().getFunctionType(argTypes, resultTypes);
  if (!type)
    return failure();
  result.addAttribute(SDFGNode::getTypeAttrName(), TypeAttr::get(type));

  if (parser.parseRegion(*body, entryArgs,
                         entryArgs.empty() ? ArrayRef<Type>() : argTypes,
                         /*enableNameShadowing=*/false))
    return failure();

  if (body->empty())
    return parser.emitError(parser.getCurrentLocation(),
                            "expected non-empty function body");

  return success();
}

static void print(OpAsmPrinter &p, SDFGNode op) {
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{"ID", "sym_name", "type"});
  p << ' ';
  p.printSymbolName(op.sym_name());

  if (!op.getType().getInputs().empty() || !op.getType().getResults().empty())
    function_like_impl::printFunctionSignature(p, op, op.getType().getInputs(),
                                               /*isVariadic=*/false,
                                               op.getType().getResults());

  p.printRegion(op.body(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false, /*printEmptyBlock=*/true);
}

LogicalResult verify(SDFGNode op) {
  if (op.isExternal())
    return success();

  // Verify that the argument list of the function and the arg list of the
  // entry block line up. The trait already verified that the number of
  // arguments is the same between the signature and the block.
  ArrayRef<Type> fnInputTypes = op.getType().getInputs();
  Block &entryBlock = op.front();

  for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i)
    if (fnInputTypes[i] != entryBlock.getArgument(i).getType())
      return op.emitOpError("type of entry block argument #")
             << i << '(' << entryBlock.getArgument(i).getType()
             << ") must match the type of the corresponding argument in "
             << "function signature(" << fnInputTypes[i] << ')';

  // Verify that no other dialect is used in the body
  for (Operation &oper : op.body().getOps())
    if (oper.getDialect() != (*op).getDialect())
      return op.emitOpError("does not support other dialects");
  return success();
}

LogicalResult SDFGNode::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the entry attribute is specified.
  FlatSymbolRefAttr entryAttr =
      (*this)->getAttrOfType<FlatSymbolRefAttr>("entry");
  if (!entryAttr)
    return emitOpError("requires a 'src' symbol reference attribute");
  StateNode entry =
      symbolTable.lookupNearestSymbolFrom<StateNode>(*this, entryAttr);
  if (!entry)
    return emitOpError() << "'" << entryAttr.getValue()
                         << "' does not reference a valid state";

  return success();
}

unsigned SDFGNode::getIndexOfState(StateNode &node) {
  unsigned state_idx = 0;

  for (Operation &op : body().getOps()) {
    if (StateNode state = dyn_cast<StateNode>(op)) {
      if (state.sym_name() == node.sym_name())
        return state_idx;

      ++state_idx;
    }
  }

  return -1;
}

StateNode SDFGNode::getStateByIndex(unsigned idx) {
  unsigned state_idx = 0;

  for (Operation &op : body().getOps()) {
    if (StateNode state = dyn_cast<StateNode>(op)) {
      if (state_idx == idx)
        return state;

      ++state_idx;
    }
  }

  return nullptr;
}

StateNode SDFGNode::getStateBySymRef(StringRef symRef) {
  Operation *op = lookupSymbol(symRef);
  return dyn_cast<StateNode>(op);
}

bool SDFGNode::isNested() {
  Operation *parent = getOperation()->getParentOp();
  if (StateNode state = dyn_cast<StateNode>(parent))
    return true;
  return false;
}

void SDFGNode::setID(unsigned id) {
  Builder builder(*this);
  IntegerAttr intAttr = builder.getI32IntegerAttr(id);
  IDAttr(intAttr);
}

//===----------------------------------------------------------------------===//
// StateNode
//===----------------------------------------------------------------------===//

StateNode StateNode::create(PatternRewriter &rewriter, Location loc) {
  return create(rewriter, loc, "state");
}

StateNode StateNode::create(PatternRewriter &rewriter, Location loc,
                            StringRef name) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());
  build(builder, state, utils::generateID(), utils::generateName(name.str()));
  StateNode stateNode = cast<StateNode>(rewriter.createOperation(state));
  rewriter.createBlock(&stateNode.body());
  return stateNode;
}

StateNode StateNode::create(Location loc, StringRef name) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());
  build(builder, state, utils::generateID(), utils::generateName(name.str()));
  return cast<StateNode>(Operation::create(state));
}

static ParseResult parseStateNode(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  IntegerAttr intAttr =
      parser.getBuilder().getI32IntegerAttr(utils::generateID());
  result.addAttribute("ID", intAttr);

  StringAttr sym_nameAttr;
  if (parser.parseSymbolName(sym_nameAttr, "sym_name", result.attributes))
    return failure();

  Region *body = result.addRegion();
  if (parser.parseRegion(*body))
    return failure();

  if (body->empty())
    body->emplaceBlock();

  return success();
}

static void print(OpAsmPrinter &p, StateNode op) {
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"ID", "sym_name"});
  p << ' ';
  p.printSymbolName(op.sym_name());
  p.printRegion(op.body());
}

LogicalResult verify(StateNode op) {
  // Verify that no other dialect is used in the body
  // Except func operations
  for (Operation &oper : op.body().getOps())
    if (oper.getDialect() != (*op).getDialect() && !dyn_cast<FuncOp>(oper))
      return op.emitOpError("does not support other dialects");
  return success();
}

void StateNode::setID(unsigned id) {
  Builder builder(*this);
  IntegerAttr intAttr = builder.getI32IntegerAttr(id);
  IDAttr(intAttr);
}

//===----------------------------------------------------------------------===//
// TaskletNode
//===----------------------------------------------------------------------===//

static ParseResult parseTaskletNode(OpAsmParser &parser,
                                    OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  IntegerAttr intAttr =
      parser.getBuilder().getI32IntegerAttr(utils::generateID());
  result.addAttribute("ID", intAttr);

  StringAttr sym_nameAttr;
  if (parser.parseSymbolName(sym_nameAttr, "sym_name", result.attributes))
    return failure();

  SmallVector<OpAsmParser::OperandType, 4> args;
  SmallVector<Type, 4> argTypes;

  if (parser.parseLParen())
    return failure();

  while (parser.parseOptionalRParen().failed()) {
    if (result.operands.size() > 0)
      if (parser.parseComma())
        return failure();

    OpAsmParser::OperandType operand;
    OpAsmParser::OperandType arg;
    Type opType;

    if (parser.parseOperand(operand))
      return failure();

    if (parser.parseOptionalKeyword("as").succeeded())
      if (parser.parseRegionArgument(arg))
        return failure();

    if (parser.parseColonType(opType))
      return failure();

    if (parser.resolveOperand(operand, opType, result.operands))
      return failure();

    if (arg.location.isValid())
      args.push_back(arg);
    else
      args.push_back(operand);

    argTypes.push_back(opType);
  }

  Type retType;
  if (parser.parseArrow() || parser.parseType(retType))
    return failure();
  result.addTypes(retType);

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, args, argTypes, /*enableNameShadowing=*/true))
    return failure();

  if (body->empty())
    return parser.emitError(parser.getCurrentLocation(),
                            "expected non-empty tasklet body");

  return success();
}

static void print(OpAsmPrinter &p, TaskletNode op) {
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{"ID", "sym_name"});
  p << ' ';
  p.printSymbolName(op.sym_name());
  p << '(';

  for (unsigned i = 0; i < op.getNumOperands(); ++i) {
    if (i > 0)
      p << ", ";
    p << op.getOperand(i) << " as " << op.body().getArgument(i) << " : "
      << op.getOperandTypes()[i];
  }

  p << ") -> " << op.getType(0);
  p.printRegion(op.body(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true, /*printEmptyBlock=*/true);
}

LogicalResult verify(TaskletNode op) { return success(); }

TaskletNode TaskletNode::create(PatternRewriter &rewriter, Location location,
                                ValueRange operands, TypeRange results) {
  OpBuilder builder(location->getContext());
  OperationState state(location, getOperationName());
  build(builder, state, results, utils::generateID(),
        utils::generateName("task"), operands);

  TaskletNode task = cast<TaskletNode>(rewriter.createOperation(state));
  rewriter.createBlock(&task.getRegion(), {}, operands.getTypes());
  return task;
}

TaskletNode TaskletNode::create(Location location, StringRef name,
                                ValueRange operands, TypeRange results) {
  OpBuilder builder(location->getContext());
  OperationState state(location, getOperationName());
  build(builder, state, results, utils::generateID(),
        utils::generateName(name.str()), operands);

  TaskletNode task = cast<TaskletNode>(Operation::create(state));
  builder.createBlock(&task.body(), {}, operands.getTypes());
  return task;
}

void TaskletNode::setID(unsigned id) {
  Builder builder(*this);
  IntegerAttr intAttr = builder.getI32IntegerAttr(id);
  IDAttr(intAttr);
}

std::string TaskletNode::getInputName(unsigned idx) {
  AsmState state(*this);
  std::string name;
  llvm::raw_string_ostream nameStream(name);
  getOperand(idx).printAsOperand(nameStream, state);
  name.erase(0, 1); // Remove %-sign
  return getName().str() + "_" + name;
}

std::string TaskletNode::getOutputName(unsigned idx) {
  // TODO: Implement multiple return values
  return "__out";
}

//===----------------------------------------------------------------------===//
// MapNode
//===----------------------------------------------------------------------===//

static ParseResult parseMapNode(OpAsmParser &parser, OperationState &result) {
  Builder &builder = parser.getBuilder();
  IndexType indexType = builder.getIndexType();

  IntegerAttr intAttr =
      parser.getBuilder().getI32IntegerAttr(utils::generateID());
  result.addAttribute("entryID", intAttr);

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  SmallVector<OpAsmParser::OperandType, 4> ivs;
  if (parser.parseRegionArgumentList(ivs, OpAsmParser::Delimiter::Paren))
    return failure();

  if (parser.parseEqual())
    return failure();

  if (parser.parseLParen() || parseNumberList(parser, result, "lowerBounds") ||
      parser.parseRParen())
    return failure();

  if (parser.parseKeyword("to"))
    return failure();

  if (parser.parseLParen() || parseNumberList(parser, result, "upperBounds") ||
      parser.parseRParen())
    return failure();

  if (parser.parseKeyword("step"))
    return failure();

  if (parser.parseLParen() || parseNumberList(parser, result, "steps") ||
      parser.parseRParen())
    return failure();

  // Now parse the body.
  Region *body = result.addRegion();
  SmallVector<Type, 4> types(ivs.size(), indexType);
  if (parser.parseRegion(*body, ivs, types))
    return failure();

  intAttr = parser.getBuilder().getI32IntegerAttr(utils::generateID());
  result.addAttribute("exitID", intAttr);
  return success();
}

static void print(OpAsmPrinter &p, MapNode op) {
  printOptionalAttrDictNoNumList(
      p, op->getAttrs(),
      {"entryID", "exitID", "lowerBounds", "upperBounds", "steps"});

  p << " (" << op.getBody()->getArguments() << ") = (";

  printNumberList(p, op.getOperation(), "lowerBounds");

  p << ") to (";

  printNumberList(p, op.getOperation(), "upperBounds");

  p << ") step (";

  printNumberList(p, op.getOperation(), "steps");

  p << ")";

  p.printRegion(op.body(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

LogicalResult verify(MapNode op) {
  size_t var_count = op.getBody()->getArguments().size();

  if (getNumListSize(op, "lowerBounds") != var_count)
    return op.emitOpError("failed to verify that size of "
                          "lower bounds matches size of arguments");

  if (getNumListSize(op, "upperBounds") != var_count)
    return op.emitOpError("failed to verify that size of "
                          "upper bounds matches size of arguments");

  if (getNumListSize(op, "steps") != var_count)
    return op.emitOpError("failed to verify that size of "
                          "steps matches size of arguments");

  // Verify that no other dialect is used in the body
  for (Operation &oper : op.body().getOps())
    if (oper.getDialect() != (*op).getDialect())
      return op.emitOpError("does not support other dialects");

  return success();
}

bool MapNode::isDefinedOutsideOfLoop(Value value) {
  return !body().isAncestor(value.getParentRegion());
}

Region &MapNode::getLoopBody() { return body(); }

LogicalResult MapNode::moveOutOfLoop(ArrayRef<Operation *> ops) {
  return failure();
}

void MapNode::setEntryID(unsigned id) {
  Builder builder(*this);
  IntegerAttr intAttr = builder.getI32IntegerAttr(id);
  entryIDAttr(intAttr);
}

void MapNode::setExitID(unsigned id) {
  Builder builder(*this);
  IntegerAttr intAttr = builder.getI32IntegerAttr(id);
  exitIDAttr(intAttr);
}

//===----------------------------------------------------------------------===//
// ConsumeNode
//===----------------------------------------------------------------------===//

static ParseResult parseConsumeNode(OpAsmParser &parser,
                                    OperationState &result) {
  IntegerAttr intAttr =
      parser.getBuilder().getI32IntegerAttr(utils::generateID());
  result.addAttribute("entryID", intAttr);

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (parser.parseLParen())
    return failure();

  OpAsmParser::OperandType stream;
  Type streamType;
  if (parser.parseOperand(stream) || parser.parseColonType(streamType) ||
      parser.resolveOperand(stream, streamType, result.operands) ||
      !streamType.isa<StreamType>())
    return failure();

  if (parser.parseRParen() || parser.parseArrow() || parser.parseLParen())
    return failure();

  SmallVector<OpAsmParser::OperandType, 4> ivs;
  OpAsmParser::OperandType num_pes_op;
  if (parser.parseKeyword("pe") || parser.parseColon() ||
      parser.parseOperand(num_pes_op))
    return failure();
  ivs.push_back(num_pes_op);

  if (parser.parseComma())
    return failure();

  OpAsmParser::OperandType elem_op;
  if (parser.parseKeyword("elem") || parser.parseColon() ||
      parser.parseOperand(elem_op))
    return failure();
  ivs.push_back(elem_op);

  if (parser.parseRParen())
    return failure();

  // Now parse the body.
  Region *body = result.addRegion();
  SmallVector<Type, 4> types;
  types.push_back(parser.getBuilder().getIndexType());
  types.push_back(streamType.cast<StreamType>().getElementType());
  if (parser.parseRegion(*body, ivs, types))
    return failure();

  intAttr = parser.getBuilder().getI32IntegerAttr(utils::generateID());
  result.addAttribute("exitID", intAttr);
  return success();
}

static void print(OpAsmPrinter &p, ConsumeNode op) {
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{"entryID", "exitID"});
  p << " (" << op.stream() << " : " << op.stream().getType() << ")";
  p << " -> (pe: " << op.getBody()->getArgument(0);
  p << ", elem: " << op.getBody()->getArgument(1) << ")";
  p.printRegion(op.body(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

LogicalResult verify(ConsumeNode op) {
  if (op.num_pes().hasValue() && op.num_pes().getValue().isNonPositive())
    return op.emitOpError("failed to verify that number of "
                          "processing elements is at least one");

  // Verify that no other dialect is used in the body
  for (Operation &oper : op.body().getOps())
    if (oper.getDialect() != (*op).getDialect())
      return op.emitOpError("does not support other dialects");

  return success();
}

LogicalResult
ConsumeNode::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the condition attribute is specified.
  FlatSymbolRefAttr condAttr =
      (*this)->getAttrOfType<FlatSymbolRefAttr>("condition");
  if (!condAttr)
    return success();

  FuncOp cond = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, condAttr);
  if (!cond)
    return emitOpError() << "'" << condAttr.getValue()
                         << "' does not reference a valid func";

  if (cond.getArguments().size() != 1)
    return emitOpError() << "'" << condAttr.getValue()
                         << "' references a func with invalid signature";

  if (cond.getArgument(0).getType() != stream().getType())
    return emitOpError() << "'" << condAttr.getValue()
                         << "' references a func with invalid signature";

  return success();
}

bool ConsumeNode::isDefinedOutsideOfLoop(Value value) {
  return !body().isAncestor(value.getParentRegion());
}

Region &ConsumeNode::getLoopBody() { return body(); }

LogicalResult ConsumeNode::moveOutOfLoop(ArrayRef<Operation *> ops) {
  return failure();
}

void ConsumeNode::setEntryID(unsigned id) {
  Builder builder(*this);
  IntegerAttr intAttr = builder.getI32IntegerAttr(id);
  entryIDAttr(intAttr);
}

void ConsumeNode::setExitID(unsigned id) {
  Builder builder(*this);
  IntegerAttr intAttr = builder.getI32IntegerAttr(id);
  exitIDAttr(intAttr);
}

//===----------------------------------------------------------------------===//
// EdgeOp
//===----------------------------------------------------------------------===//

EdgeOp EdgeOp::create(PatternRewriter &rewriter, Location loc, StateNode &from,
                      StateNode &to, ArrayAttr &assign, StringAttr &condition,
                      Value ref) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());
  build(builder, state, from.sym_name(), to.sym_name(), assign, condition, ref);
  return cast<EdgeOp>(rewriter.createOperation(state));
}

EdgeOp EdgeOp::create(PatternRewriter &rewriter, Location loc, StateNode &from,
                      StateNode &to) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());
  build(builder, state, from.sym_name(), to.sym_name(), nullptr, nullptr,
        nullptr);
  return cast<EdgeOp>(rewriter.createOperation(state));
}

EdgeOp EdgeOp::create(Location loc, StateNode &from, StateNode &to,
                      ArrayAttr &assign, StringAttr &condition, Value ref) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());
  build(builder, state, from.sym_name(), to.sym_name(), assign, condition, ref);
  return cast<EdgeOp>(Operation::create(state));
}

static ParseResult parseEdgeOp(OpAsmParser &parser, OperationState &result) {
  FlatSymbolRefAttr srcAttr;
  FlatSymbolRefAttr destAttr;

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (parser.parseOptionalLParen().succeeded()) {
    OpAsmParser::OperandType op;
    SmallVector<Value> valList;
    Type t;

    if (parser.parseKeyword("ref") || parser.parseColon() ||
        parser.parseOperand(op) || parser.parseColon() || parser.parseType(t) ||
        parser.parseRParen() || parser.resolveOperands(op, t, valList))
      return failure();

    result.addOperands(valList);
  }

  if (parser.parseAttribute(srcAttr, "src", result.attributes))
    return failure();

  if (parser.parseArrow())
    return failure();

  if (parser.parseAttribute(destAttr, "dest", result.attributes))
    return failure();

  return success();
}

static void print(OpAsmPrinter &p, EdgeOp op) {
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"src", "dest"});
  p << ' ';
  if (!op.refMutable().empty())
    p << "(ref: " << op.ref() << ": " << op.ref().getType() << ") ";
  p.printAttributeWithoutType(op.srcAttr());
  p << " -> ";
  p.printAttributeWithoutType(op.destAttr());
}

LogicalResult verify(EdgeOp op) { return success(); }

LogicalResult EdgeOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the src/dest attributes are specified.
  FlatSymbolRefAttr srcAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("src");
  if (!srcAttr)
    return emitOpError("requires a 'src' symbol reference attribute");

  StateNode src =
      symbolTable.lookupNearestSymbolFrom<StateNode>(*this, srcAttr);
  if (!src)
    return emitOpError() << "'" << srcAttr.getValue()
                         << "' does not reference a valid state";

  FlatSymbolRefAttr destAttr =
      (*this)->getAttrOfType<FlatSymbolRefAttr>("dest");
  if (!destAttr)
    return emitOpError("requires a 'dest' symbol reference attribute");

  StateNode dest =
      symbolTable.lookupNearestSymbolFrom<StateNode>(*this, destAttr);
  if (!dest)
    return emitOpError() << "'" << destAttr.getValue()
                         << "' does not reference a valid state";

  return success();
}

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

static ParseResult parseAllocOp(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  SmallVector<OpAsmParser::OperandType, 4> paramsOperands;
  if (parser.parseOperandList(paramsOperands, OpAsmParser::Delimiter::Paren))
    return failure();

  if (parser.resolveOperands(paramsOperands, parser.getBuilder().getIndexType(),
                             result.operands))
    return failure();

  Type resultType;
  if (parser.parseColonType(resultType))
    return failure();
  result.addTypes(resultType);

  return success();
}

static void print(OpAsmPrinter &p, AllocOp op) {
  p.printOptionalAttrDict(op->getAttrs());
  p << "(";
  p.printOperands(op.params());
  p << ") : ";
  p << op.getOperation()->getResultTypes();
}

LogicalResult verify(AllocOp op) {
  if (ArrayType res = op.res().getType().dyn_cast<ArrayType>()) {
    if (res.getUndefRank() != op.params().size())
      return op.emitOpError("failed to verify that parameter size "
                            "matches undefined dimensions size");

    if (res.hasZeros())
      return op.emitOpError("failed to verify that return type "
                            "doesn't contain dimensions of size zero");
  }

  return success();
}

AllocOp AllocOp::create(PatternRewriter &rewriter, Location loc, Type res,
                        StringRef name, bool transient) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());
  StringAttr nameAttr = rewriter.getStringAttr(utils::generateName(name.str()));
  build(builder, state, res, {}, nameAttr, transient);
  return cast<AllocOp>(rewriter.createOperation(state));
}

AllocOp AllocOp::create(PatternRewriter &rewriter, Location loc, Type res,
                        bool transient) {
  return create(rewriter, loc, res, "arr", transient);
}

AllocOp AllocOp::create(Location loc, Type res, StringRef name,
                        bool transient) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());
  StringAttr nameAttr = builder.getStringAttr(name);

  if (!res.isa<ArrayType>()) {
    res = ArrayType::get(res.getContext(), res, {}, {}, {});
  }

  build(builder, state, res, {}, nameAttr, transient);
  return cast<AllocOp>(Operation::create(state));
}

SDFGNode AllocOp::getParentSDFG() {
  Operation *sdfgOrState = (*this)->getParentOp();

  if (SDFGNode sdfg = dyn_cast<SDFGNode>(sdfgOrState))
    return sdfg;

  Operation *sdfg = sdfgOrState->getParentOp();
  return dyn_cast<SDFGNode>(sdfg);
}

Type AllocOp::getElementType() {
  if (ArrayType t = getType().dyn_cast<ArrayType>())
    return t.getElementType();

  if (StreamType t = getType().dyn_cast<StreamType>())
    return t.getElementType();

  return Type();
}

bool AllocOp::isScalar() {
  if (ArrayType t = getType().dyn_cast<ArrayType>())
    return t.getShape().empty();

  if (StreamType t = getType().dyn_cast<StreamType>())
    return t.getShape().empty();

  return false;
}

bool AllocOp::isInState() {
  Operation *sdfgOrState = (*this)->getParentOp();
  if (StateNode state = dyn_cast<StateNode>(sdfgOrState))
    return true;
  return false;
}

std::string AllocOp::getName() {
  if ((*this)->hasAttr("name")) {
    Attribute nameAttr = (*this)->getAttr("name");
    if (StringAttr name = nameAttr.cast<StringAttr>())
      return name.getValue().str();
  }

  AsmState state(getParentSDFG());
  std::string name;
  llvm::raw_string_ostream nameStream(name);
  (*this)->getResult(0).printAsOperand(nameStream, state);
  utils::sanitizeName(name);

  return name;
}

//===----------------------------------------------------------------------===//
// GetAccessOp
//===----------------------------------------------------------------------===//

/* GetAccessOp GetAccessOp::create(PatternRewriter &rewriter, Location loc, Type
t, Value arr) { OpBuilder builder(loc->getContext()); OperationState state(loc,
getOperationName());

  if (ArrayType art = t.dyn_cast<ArrayType>()) {
    t = art.toMemlet();
  }

  build(builder, state, t, utils::generateID(), arr);
  return cast<GetAccessOp>(rewriter.createOperation(state));
}

GetAccessOp GetAccessOp::create(Location loc, Type t, Value arr) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());

  if (ArrayType art = t.dyn_cast<ArrayType>()) {
    t = art.toMemlet();
  }

  build(builder, state, t, utils::generateID(), arr);
  return cast<GetAccessOp>(Operation::create(state));
}

static ParseResult parseGetAccessOp(OpAsmParser &parser,
                                    OperationState &result) {
  IntegerAttr intAttr =
      parser.getBuilder().getI32IntegerAttr(utils::generateID());
  result.addAttribute("ID", intAttr);

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  OpAsmParser::OperandType arrStrOperand;
  if (parser.parseOperand(arrStrOperand))
    return failure();

  Type srcType;
  if (parser.parseColonType(srcType))
    return failure();

  if (parser.parseArrow())
    return failure();

  Type destType;
  if (parser.parseType(destType))
    return failure();
  result.addTypes(destType);

  if (parser.resolveOperand(arrStrOperand, srcType, result.operands))
    return failure();

  return success();
} */

/* static void print(OpAsmPrinter &p, GetAccessOp op) {
  p.printOptionalAttrDict(op->getAttrs(), {"ID"});
  p << ' ' << op.arr();
  p << " : ";
  p << ArrayRef<Type>(op.arr().getType());
  p << " -> ";
  p << ArrayRef<Type>(op.res().getType());
}

LogicalResult verify(GetAccessOp op) {
  Type arr = op.arr().getType();
  Type res = op.res().getType();

  if (arr.isa<ArrayType>() && res.isa<MemletType>())
    if (arr.cast<ArrayType>().getElementType() !=
        res.cast<MemletType>().getElementType())
      return op.emitOpError("failed to verify that result element type "
                            "matches element type of 'array'");

  if (arr.isa<StreamArrayType>() && res.isa<StreamType>())
    if (arr.cast<StreamArrayType>().getElementType() !=
        res.cast<StreamType>().getElementType())
      return op.emitOpError("failed to verify that result element type "
                            "matches element type of 'stream_array'");

  if (arr.isa<ArrayType>() && res.isa<StreamType>())
    return op.emitOpError("failed to verify that result type matches "
                          "derived type of 'array'");

  if (arr.isa<StreamArrayType>() && res.isa<MemletType>())
    return op.emitOpError("failed to verify that result type matches "
                          "derived type of 'stream_array'");

  return success();
}

Type GetAccessOp::getAllocType() {
  Operation *alloc = arr().getDefiningOp();

  if (AllocOp allocArr = dyn_cast<AllocOp>(alloc))
    return allocArr.getType();

  return nullptr;
}

ArrayRef<StringAttr> GetAccessOp::getSymbols() {
  Type t = getAllocType();

  if (ArrayType arr = t.dyn_cast<ArrayType>())
    return arr.getSymbols();

  if (MemletType arr = t.dyn_cast<MemletType>())
    return arr.getSymbols();

  if (StreamArrayType arr = t.dyn_cast<StreamArrayType>())
    return arr.getSymbols();

  if (StreamType arr = t.dyn_cast<StreamType>())
    return arr.getSymbols();

  return {};
}

ArrayRef<int64_t> GetAccessOp::getIntegers() {
  Type t = getAllocType();

  if (ArrayType arr = t.dyn_cast<ArrayType>())
    return arr.getIntegers();

  if (MemletType arr = t.dyn_cast<MemletType>())
    return arr.getIntegers();

  if (StreamArrayType arr = t.dyn_cast<StreamArrayType>())
    return arr.getIntegers();

  if (StreamType arr = t.dyn_cast<StreamType>())
    return arr.getIntegers();

  return {};
}

ArrayRef<bool> GetAccessOp::getShape() {
  Type t = getAllocType();

  if (ArrayType arr = t.dyn_cast<ArrayType>())
    return arr.getShape();

  if (MemletType arr = t.dyn_cast<MemletType>())
    return arr.getShape();

  if (StreamArrayType arr = t.dyn_cast<StreamArrayType>())
    return arr.getShape();

  if (StreamType arr = t.dyn_cast<StreamType>())
    return arr.getShape();

  return {};
}

SDFGNode GetAccessOp::getParentSDFG() {
  Operation *sdfg = (*this)->getParentOp()->getParentOp();
  return dyn_cast<SDFGNode>(sdfg);
}

std::string GetAccessOp::getName() {
  Operation *alloc = arr().getDefiningOp();

  if (AllocOp allocArr = dyn_cast<AllocOp>(alloc))
    return allocArr.getName();

  AsmState state(getParentSDFG());
  std::string name;
  llvm::raw_string_ostream nameStream(name);
  arr().printAsOperand(nameStream, state);
  utils::sanitizeName(name);

  return name;
}

void GetAccessOp::setID(unsigned id) {
  Builder builder(*this);
  IntegerAttr intAttr = builder.getI32IntegerAttr(id);
  IDAttr(intAttr);
} */

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

LoadOp LoadOp::create(PatternRewriter &rewriter, Location loc, AllocOp alloc,
                      ValueRange indices) {
  return create(rewriter, loc, alloc.getType(), alloc, indices);
}

LoadOp LoadOp::create(PatternRewriter &rewriter, Location loc, Type t,
                      Value mem, ValueRange indices) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());

  if (ArrayType arr = t.dyn_cast<ArrayType>())
    t = arr.getElementType();

  else if (StreamType arr = t.dyn_cast<StreamType>())
    t = arr.getElementType();

  SmallVector<Attribute> numList;
  for (size_t i = 0; i < indices.size(); ++i) {
    numList.push_back(builder.getUI32IntegerAttr(i));
  }
  ArrayAttr numArr = rewriter.getArrayAttr(numList);
  state.addAttribute("indices_numList", numArr);

  SmallVector<Attribute> attrList;
  ArrayAttr attrArr = rewriter.getArrayAttr(attrList);
  state.addAttribute("indices", attrArr);

  build(builder, state, t, indices, mem);
  return cast<LoadOp>(rewriter.createOperation(state));
}

LoadOp LoadOp::create(Location loc, AllocOp alloc, ValueRange indices) {
  return create(loc, alloc.getType(), alloc, indices);
}

LoadOp LoadOp::create(Location loc, Type t, Value mem, ValueRange indices) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());

  if (ArrayType arr = t.dyn_cast<ArrayType>())
    t = arr.getElementType();

  else if (StreamType arr = t.dyn_cast<StreamType>())
    t = arr.getElementType();

  SmallVector<Attribute> numList;
  for (size_t i = 0; i < indices.size(); ++i) {
    numList.push_back(builder.getUI32IntegerAttr(i));
  }
  ArrayAttr numArr = builder.getArrayAttr(numList);
  state.addAttribute("indices_numList", numArr);

  SmallVector<Attribute> attrList;
  ArrayAttr attrArr = builder.getArrayAttr(attrList);
  state.addAttribute("indices", attrArr);

  build(builder, state, t, indices, mem);
  return cast<LoadOp>(Operation::create(state));
}

static ParseResult parseLoadOp(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  OpAsmParser::OperandType memletOperand;
  if (parser.parseOperand(memletOperand))
    return failure();

  if (parser.parseLSquare())
    return failure();
  parseNumberList(parser, result, "indices");
  if (parser.parseRSquare())
    return failure();

  Type srcType;
  if (parser.parseColonType(srcType))
    return failure();

  if (parser.parseArrow())
    return failure();

  Type destType;
  if (parser.parseType(destType))
    return failure();
  result.addTypes(destType);

  if (parser.resolveOperand(memletOperand, srcType, result.operands))
    return failure();

  return success();
}

static void print(OpAsmPrinter &p, LoadOp op) {
  printOptionalAttrDictNoNumList(p, op->getAttrs(),
                                 /*elidedAttrs*/ {"indices"});
  p << ' ' << op.arr();
  p << "[";
  printNumberList(p, op.getOperation(), "indices");
  p << "]";
  p << " : ";
  p << ArrayRef<Type>(op.arr().getType());
  p << " -> ";
  p << ArrayRef<Type>(op.res().getType());
}

LogicalResult verify(LoadOp op) {
  size_t idx_size = getNumListSize(op.getOperation(), "indices");
  size_t mem_size = op.arr().getType().cast<ArrayType>().getRank();
  if (idx_size != mem_size)
    return op.emitOpError("incorrect number of indices for load");

  return success();
}

bool LoadOp::isIndirect() {
  for (Value v : indices())
    if (!isa<sdir::TaskletNode>(v.getDefiningOp()))
      return true;

  return false;
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

StoreOp StoreOp::create(PatternRewriter &rewriter, Location loc, Value val,
                        Value mem, ValueRange indices) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());

  SmallVector<Attribute> numList;
  for (size_t i = 0; i < indices.size(); ++i) {
    numList.push_back(builder.getUI32IntegerAttr(i));
  }
  ArrayAttr numArr = rewriter.getArrayAttr(numList);
  state.addAttribute("indices_numList", numArr);

  SmallVector<Attribute> attrList;
  ArrayAttr attrArr = rewriter.getArrayAttr(attrList);
  state.addAttribute("indices", attrArr);

  build(builder, state, indices, val, mem);
  return cast<StoreOp>(rewriter.createOperation(state));
}

StoreOp StoreOp::create(Location loc, Value val, Value mem,
                        ValueRange indices) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());

  SmallVector<Attribute> numList;
  for (size_t i = 0; i < indices.size(); ++i) {
    numList.push_back(builder.getUI32IntegerAttr(i));
  }
  ArrayAttr numArr = builder.getArrayAttr(numList);
  state.addAttribute("indices_numList", numArr);

  SmallVector<Attribute> attrList;
  ArrayAttr attrArr = builder.getArrayAttr(attrList);
  state.addAttribute("indices", attrArr);

  build(builder, state, indices, val, mem);
  return cast<StoreOp>(Operation::create(state));
}

StoreOp StoreOp::create(Location loc, Value val, Value mem,
                        ArrayRef<StringRef> indices) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());

  SmallVector<Attribute> numList;
  for (size_t i = 0; i < indices.size(); ++i) {
    numList.push_back(builder.getUI32IntegerAttr(-1));
  }
  ArrayAttr numArr = builder.getArrayAttr(numList);
  state.addAttribute("indices_numList", numArr);

  SmallVector<Attribute> attrList;
  for (StringRef str : indices)
    attrList.push_back(builder.getStringAttr(str));

  ArrayAttr attrArr = builder.getArrayAttr(attrList);
  state.addAttribute("indices", attrArr);

  build(builder, state, {}, val, mem);
  return cast<StoreOp>(Operation::create(state));
}

StoreOp StoreOp::create(Location loc, Value val, Value mem) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());

  SmallVector<Attribute> numList;
  ArrayAttr numArr = builder.getArrayAttr(numList);
  state.addAttribute("indices_numList", numArr);

  SmallVector<Attribute> attrList;
  ArrayAttr attrArr = builder.getArrayAttr(attrList);
  state.addAttribute("indices", attrArr);

  BoolAttr fullRange = builder.getBoolAttr(true);
  state.addAttribute("isFullRange", fullRange);

  build(builder, state, {}, val, mem);
  return cast<StoreOp>(Operation::create(state));
}

static ParseResult parseStoreOp(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  OpAsmParser::OperandType valOperand;
  if (parser.parseOperand(valOperand))
    return failure();
  if (parser.parseComma())
    return failure();

  OpAsmParser::OperandType memletOperand;
  if (parser.parseOperand(memletOperand))
    return failure();

  if (parser.parseLSquare())
    return failure();
  parseNumberList(parser, result, "indices");
  if (parser.parseRSquare())
    return failure();

  Type valType;
  if (parser.parseColonType(valType))
    return failure();

  if (parser.parseArrow())
    return failure();

  Type memletType;
  if (parser.parseType(memletType))
    return failure();

  if (parser.resolveOperand(valOperand, valType, result.operands))
    return failure();

  if (parser.resolveOperand(memletOperand, memletType, result.operands))
    return failure();

  return success();
}

static void print(OpAsmPrinter &p, StoreOp op) {
  printOptionalAttrDictNoNumList(p, op->getAttrs(),
                                 /*elidedAttrs=*/{"indices"});
  p << ' ' << op.val() << "," << ' ' << op.arr();
  p << "[";
  printNumberList(p, op.getOperation(), "indices");
  p << "]";
  p << " : ";
  p << ArrayRef<Type>(op.val().getType());
  p << " -> ";
  p << ArrayRef<Type>(op.arr().getType());
}

LogicalResult verify(StoreOp op) {
  size_t idx_size = getNumListSize(op.getOperation(), "indices");
  size_t mem_size = op.arr().getType().cast<ArrayType>().getRank();
  if (idx_size != mem_size)
    return op.emitOpError("incorrect number of indices for store");

  return success();
}

bool StoreOp::isIndirect() {
  for (Value v : indices())
    if (!isa<sdir::TaskletNode>(v.getDefiningOp()))
      return true;

  return false;
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

static ParseResult parseCopyOp(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  OpAsmParser::OperandType srcOperand;
  if (parser.parseOperand(srcOperand))
    return failure();

  if (parser.parseArrow())
    return failure();

  OpAsmParser::OperandType destOperand;
  if (parser.parseOperand(destOperand))
    return failure();

  Type opType;
  if (parser.parseColonType(opType))
    return failure();

  if (parser.resolveOperand(srcOperand, opType, result.operands))
    return failure();

  if (parser.resolveOperand(destOperand, opType, result.operands))
    return failure();

  return success();
}

static void print(OpAsmPrinter &p, CopyOp op) {
  p.printOptionalAttrDict(op->getAttrs());
  p << ' ' << op.src() << " -> " << op.dest();
  p << " : ";
  p << ArrayRef<Type>(op.src().getType());
}

LogicalResult verify(CopyOp op) { return success(); }

//===----------------------------------------------------------------------===//
// MemletCastOp
//===----------------------------------------------------------------------===//

static ParseResult parseMemletCastOp(OpAsmParser &parser,
                                     OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  OpAsmParser::OperandType memletOperand;
  if (parser.parseOperand(memletOperand))
    return failure();

  Type srcType;
  if (parser.parseColonType(srcType))
    return failure();

  if (parser.parseArrow())
    return failure();

  Type destType;
  if (parser.parseType(destType))
    return failure();
  result.addTypes(destType);

  if (parser.resolveOperands(memletOperand, srcType, result.operands))
    return failure();

  return success();
}

static void print(OpAsmPrinter &p, MemletCastOp op) {
  p.printOptionalAttrDict(op->getAttrs());
  p << ' ' << op.src();
  p << " : ";
  p << ArrayRef<Type>(op.src().getType());
  p << " -> ";
  p << op.getOperation()->getResultTypes();
}

LogicalResult verify(MemletCastOp op) {
  size_t src_size = op.src().getType().cast<ArrayType>().getRank();
  size_t res_size = op.res().getType().cast<ArrayType>().getRank();
  if (src_size != res_size)
    return op.emitOpError("incorrect rank for memlet_cast");

  return success();
}

//===----------------------------------------------------------------------===//
// ViewCastOp
//===----------------------------------------------------------------------===//

static ParseResult parseViewCastOp(OpAsmParser &parser,
                                   OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  OpAsmParser::OperandType memletOperand;
  if (parser.parseOperand(memletOperand))
    return failure();

  Type srcType;
  if (parser.parseColonType(srcType))
    return failure();

  if (parser.parseArrow())
    return failure();

  Type destType;
  if (parser.parseType(destType))
    return failure();
  result.addTypes(destType);

  if (parser.resolveOperands(memletOperand, srcType, result.operands))
    return failure();

  return success();
}

static void print(OpAsmPrinter &p, ViewCastOp op) {
  p.printOptionalAttrDict(op->getAttrs());
  p << ' ' << op.src();
  p << " : ";
  p << ArrayRef<Type>(op.src().getType());
  p << " -> ";
  p << op.getOperation()->getResultTypes();
}

LogicalResult verify(ViewCastOp op) {
  size_t src_size = op.src().getType().cast<ArrayType>().getRank();
  size_t res_size = op.res().getType().cast<ArrayType>().getRank();
  if (src_size != res_size)
    return op.emitOpError("incorrect rank for view_cast");

  return success();
}

//===----------------------------------------------------------------------===//
// SubviewOp
//===----------------------------------------------------------------------===//

static ParseResult parseSubviewOp(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  OpAsmParser::OperandType memletOperand;
  if (parser.parseOperand(memletOperand))
    return failure();

  if (parser.parseLSquare() || parseNumberList(parser, result, "offsets") ||
      parser.parseRSquare())
    return failure();

  if (parser.parseLSquare() || parseNumberList(parser, result, "sizes") ||
      parser.parseRSquare())
    return failure();

  if (parser.parseLSquare() || parseNumberList(parser, result, "strides") ||
      parser.parseRSquare())
    return failure();

  Type srcType;
  if (parser.parseColonType(srcType))
    return failure();

  if (parser.parseArrow())
    return failure();

  Type destType;
  if (parser.parseType(destType))
    return failure();
  result.addTypes(destType);

  if (parser.resolveOperands(memletOperand, srcType, result.operands))
    return failure();

  return success();
}

static void print(OpAsmPrinter &p, SubviewOp op) {
  printOptionalAttrDictNoNumList(p, op->getAttrs(),
                                 {"offsets", "sizes", "strides"});
  p << ' ' << op.src() << "[";
  printNumberList(p, op.getOperation(), "offsets");
  p << "][";
  printNumberList(p, op.getOperation(), "sizes");
  p << "][";
  printNumberList(p, op.getOperation(), "strides");
  p << "]";
  p << " : ";
  p << ArrayRef<Type>(op.src().getType());
  p << " -> ";
  p << op.getOperation()->getResultTypes();
}

LogicalResult verify(SubviewOp op) { return success(); }

//===----------------------------------------------------------------------===//
// StreamPopOp
//===----------------------------------------------------------------------===//

static ParseResult parseStreamPopOp(OpAsmParser &parser,
                                    OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  OpAsmParser::OperandType streamOperand;
  if (parser.parseOperand(streamOperand))
    return failure();

  Type streamType;
  if (parser.parseColonType(streamType))
    return failure();

  if (parser.parseArrow())
    return failure();

  Type resultType;
  if (parser.parseType(resultType))
    return failure();
  result.addTypes(resultType);

  if (parser.resolveOperand(streamOperand, streamType, result.operands))
    return failure();

  return success();
}

static void print(OpAsmPrinter &p, StreamPopOp op) {
  p.printOptionalAttrDict(op->getAttrs());
  p << ' ' << op.str();
  p << " : ";
  p << ArrayRef<Type>(op.str().getType());
  p << " -> ";
  p << ArrayRef<Type>(op.res().getType());
}

LogicalResult verify(StreamPopOp op) { return success(); }

//===----------------------------------------------------------------------===//
// StreamPushOp
//===----------------------------------------------------------------------===//

static ParseResult parseStreamPushOp(OpAsmParser &parser,
                                     OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  OpAsmParser::OperandType valOperand;
  if (parser.parseOperand(valOperand))
    return failure();
  if (parser.parseComma())
    return failure();

  OpAsmParser::OperandType streamOperand;
  if (parser.parseOperand(streamOperand))
    return failure();

  Type valType;
  if (parser.parseColonType(valType))
    return failure();

  if (parser.parseArrow())
    return failure();

  Type streamType;
  if (parser.parseType(streamType))
    return failure();

  if (parser.resolveOperand(valOperand, valType, result.operands))
    return failure();

  if (parser.resolveOperand(streamOperand, streamType, result.operands))
    return failure();

  return success();
}

static void print(OpAsmPrinter &p, StreamPushOp op) {
  p.printOptionalAttrDict(op->getAttrs());
  p << ' ' << op.val() << ", " << op.str();
  p << " : ";
  p << ArrayRef<Type>(op.val().getType());
  p << " -> ";
  p << ArrayRef<Type>(op.str().getType());
}

LogicalResult verify(StreamPushOp op) {
  if (op.val().getType() !=
      op.str().getType().cast<StreamType>().getElementType())
    op.emitOpError("failed to verify that value type "
                   "matches element type of 'stream'");

  return success();
}

//===----------------------------------------------------------------------===//
// StreamLengthOp
//===----------------------------------------------------------------------===//

static ParseResult parseStreamLengthOp(OpAsmParser &parser,
                                       OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  OpAsmParser::OperandType streamOperand;
  if (parser.parseOperand(streamOperand))
    return failure();

  Type streamType;
  if (parser.parseColonType(streamType))
    return failure();

  if (parser.parseArrow())
    return failure();

  Type resultType;
  if (parser.parseType(resultType))
    return failure();
  result.addTypes(resultType);

  if (parser.resolveOperands(streamOperand, streamType, result.operands))
    return failure();

  return success();
}

static void print(OpAsmPrinter &p, StreamLengthOp op) {
  p.printOptionalAttrDict(op->getAttrs());
  p << ' ' << op.str();
  p << " : ";
  p << ArrayRef<Type>(op.str().getType());
  p << " -> ";
  p << op.getOperation()->getResultTypes();
}

LogicalResult verify(StreamLengthOp op) { return success(); }

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

static ParseResult parseReturnOp(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  SmallVector<OpAsmParser::OperandType, 4> returnOperands;
  if (parser.parseOperandList(returnOperands))
    return failure();

  if (!returnOperands.empty()) {
    if (parser.parseColon())
      return failure();

    SmallVector<Type, 1> returnTypes;
    if (parser.parseTypeList(returnTypes))
      return failure();

    if (parser.resolveOperands(returnOperands, returnTypes,
                               parser.getCurrentLocation(), result.operands))
      return failure();
  }

  return success();
}

static void print(OpAsmPrinter &p, sdir::ReturnOp op) {
  p.printOptionalAttrDict(op->getAttrs());
  if (!op.input().empty()) {
    p << ' ' << op.input();
    p << " : ";
    p << op.input().getTypes();
  }
}

LogicalResult verify(sdir::ReturnOp op) {
  TaskletNode task = dyn_cast<TaskletNode>(op->getParentOp());

  if (task.getResultTypes() != op.getOperandTypes())
    return op.emitOpError("must match tasklet return types");

  return success();
}

sdir::ReturnOp sdir::ReturnOp::create(PatternRewriter &rewriter, Location loc,
                                      mlir::ValueRange input) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());
  build(builder, state, input);
  return cast<sdir::ReturnOp>(rewriter.createOperation(state));
}

sdir::ReturnOp sdir::ReturnOp::create(Location loc, mlir::ValueRange input) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());
  build(builder, state, input);
  return cast<sdir::ReturnOp>(Operation::create(state));
}

//===----------------------------------------------------------------------===//
// LibCallOp
//===----------------------------------------------------------------------===//

static ParseResult parseLibCallOp(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  Attribute calleeAttr;
  if (parser.parseAttribute(calleeAttr, parser.getBuilder().getNoneType(),
                            "callee", result.attributes))
    return failure();

  SmallVector<OpAsmParser::OperandType, 4> operandsOperands;
  if (parser.parseOperandList(operandsOperands, OpAsmParser::Delimiter::Paren))
    return failure();

  FunctionType func;
  if (parser.parseColonType(func))
    return failure();

  ArrayRef<Type> operandsTypes = func.getInputs();
  ArrayRef<Type> allResultTypes = func.getResults();
  result.addTypes(allResultTypes);

  if (parser.resolveOperands(operandsOperands, operandsTypes,
                             parser.getCurrentLocation(), result.operands))
    return failure();

  return success();
}

static void print(OpAsmPrinter &p, LibCallOp op) {
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"callee"});
  p << ' ';
  p.printAttributeWithoutType(op.calleeAttr());
  p << "(" << op.operands() << ")";
  p << " : ";
  p.printFunctionalType(op.operands().getTypes(),
                        op.getOperation()->getResultTypes());
}

LogicalResult verify(LibCallOp op) { return success(); }

//===----------------------------------------------------------------------===//
// AllocSymbolOp
//===----------------------------------------------------------------------===//

AllocSymbolOp AllocSymbolOp::create(PatternRewriter &rewriter, Location loc,
                                    StringRef sym) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());
  build(builder, state, sym);
  return cast<AllocSymbolOp>(rewriter.createOperation(state));
}

AllocSymbolOp AllocSymbolOp::create(Location loc, StringRef sym) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());
  build(builder, state, sym);
  return cast<AllocSymbolOp>(Operation::create(state));
}

static ParseResult parseAllocSymbolOp(OpAsmParser &parser,
                                      OperationState &result) {
  StringAttr symAttr;
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (parser.parseLParen())
    return failure();
  if (parser.parseAttribute(symAttr, parser.getBuilder().getNoneType(), "sym",
                            result.attributes))
    return failure();
  if (parser.parseRParen())
    return failure();

  return success();
}

static void print(OpAsmPrinter &p, AllocSymbolOp op) {
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"sym"});
  p << "(";
  p.printAttributeWithoutType(op.symAttr());
  p << ")";
}

LogicalResult verify(AllocSymbolOp op) {
  if (op.sym().empty())
    return op.emitOpError("failed to verify that input string is not empty");

  if (!isalpha(op.sym().front()))
    return op.emitOpError("failed to verify that input string starts with "
                          "an alphabetical character");

  for (auto c : op.sym())
    if (!isalnum(c) && c != '_')
      return op.emitOpError("failed to verify that input string only "
                            "contains alphanumeric characters");

  return success();
}

//===----------------------------------------------------------------------===//
// SymOp
//===----------------------------------------------------------------------===//

SymOp SymOp::create(PatternRewriter &rewriter, Location loc, Type type,
                    StringRef expr) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());
  build(builder, state, type, expr);
  return cast<SymOp>(rewriter.createOperation(state));
}

SymOp SymOp::create(Location loc, Type type, StringRef expr) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());
  build(builder, state, type, expr);
  return cast<SymOp>(Operation::create(state));
}

static ParseResult parseSymOp(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  StringAttr exprAttr;
  if (parser.parseLParen() ||
      parser.parseAttribute(exprAttr, "expr", result.attributes) ||
      parser.parseRParen())
    return failure();

  Type resType;
  if (parser.parseColonType(resType))
    return failure();
  result.addTypes(resType);

  return success();
}

static void print(OpAsmPrinter &p, SymOp op) {
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"expr"});
  p << "(";
  p.printAttributeWithoutType(op.exprAttr());
  p << ") : " << op->getResultTypes();
}

LogicalResult verify(SymOp op) { return success(); }

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "SDIR/Dialect/Ops.cpp.inc"
