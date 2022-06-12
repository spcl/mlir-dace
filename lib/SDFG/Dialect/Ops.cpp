#include "SDFG/Dialect/Dialect.h"
#include "SDFG/Utils/Utils.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;
using namespace sdfg;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static ParseResult parseRegion(OpAsmParser &parser, OperationState &result,
                               SmallVector<OpAsmParser::OperandType, 4> &args,
                               SmallVector<Type, 4> &argTypes,
                               bool enableShadowing) {
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, args, argTypes, enableShadowing))
    return failure();

  if (body->empty())
    return parser.emitError(parser.getCurrentLocation(),
                            "expected non-empty body");
  return success();
}

static ParseResult parseArgsList(OpAsmParser &parser,
                                 SmallVector<OpAsmParser::OperandType, 4> &args,
                                 SmallVector<Type, 4> &argTypes) {
  if (parser.parseLParen())
    return failure();

  for (unsigned i = 0; parser.parseOptionalRParen().failed(); ++i) {
    if (i > 0 && parser.parseComma())
      return failure();

    OpAsmParser::OperandType arg;
    Type type;

    if (parser.parseRegionArgument(arg) || parser.parseColonType(type))
      return failure();

    args.push_back(arg);
    argTypes.push_back(type);
  }

  return success();
}

static void printArgsList(OpAsmPrinter &p, Region::BlockArgListType args,
                          unsigned lb, unsigned ub) {
  p << " (";

  for (unsigned i = lb; i < ub; ++i) {
    if (i > lb)
      p << ", ";
    p << args[i] << ": " << args[i].getType();
  }

  p << ")";
}

static ParseResult parseAsArgs(OpAsmParser &parser, OperationState &result,
                               SmallVector<OpAsmParser::OperandType, 4> &args,
                               SmallVector<Type, 4> &argTypes) {
  if (parser.parseLParen())
    return failure();

  for (unsigned i = 0; parser.parseOptionalRParen().failed(); ++i) {
    if (i > 0 && parser.parseComma())
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

  return success();
}

static void printAsArgs(OpAsmPrinter &p, OperandRange opRange,
                        Region::BlockArgListType args, unsigned lb,
                        unsigned ub) {
  p << " (";

  for (unsigned i = lb; i < ub; ++i) {
    if (i > lb)
      p << ", ";
    p << opRange[i] << " as " << args[i] << ": " << opRange[i].getType();
  }

  p << ")";
}

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
      numList.push_back(parser.getBuilder().getI32IntegerAttr(opIdx++));
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
      unsigned idx = num.getInt();
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

SDFGNode SDFGNode::create(PatternRewriter &rewriter, Location loc) {
  return create(rewriter, loc, 0, {});
}

SDFGNode SDFGNode::create(PatternRewriter &rewriter, Location loc,
                          unsigned num_args, TypeRange args) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());
  build(builder, state, utils::generateID(), nullptr, num_args);
  SDFGNode sdfg = cast<SDFGNode>(rewriter.createOperation(state));
  rewriter.createBlock(&sdfg.getRegion(), {}, args);
  return sdfg;
}

static ParseResult parseSDFGNode(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  IntegerAttr intAttr =
      parser.getBuilder().getI32IntegerAttr(utils::generateID());
  result.addAttribute("ID", intAttr);

  SmallVector<OpAsmParser::OperandType, 4> args;
  SmallVector<Type, 4> argTypes;

  if (parseArgsList(parser, args, argTypes))
    return failure();

  result.addAttribute("num_args",
                      parser.getBuilder().getI32IntegerAttr(args.size()));

  if (parser.parseArrow() || parseArgsList(parser, args, argTypes))
    return failure();

  if (parseRegion(parser, result, args, argTypes, /*enableShadowing*/ true))
    return failure();

  return success();
}

static void print(OpAsmPrinter &p, SDFGNode op) {
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{"ID", "num_args"});

  printArgsList(p, op.body().getArguments(), 0, op.num_args());
  p << " ->";
  printArgsList(p, op.body().getArguments(), op.num_args(),
                op.body().getNumArguments());

  p.printRegion(op.body(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true, /*printEmptyBlock=*/true);
}

LogicalResult verify(SDFGNode op) {
  // Verify that no other dialect is used in the body
  for (Operation &oper : op.body().getOps())
    if (oper.getDialect() != (*op).getDialect())
      return op.emitOpError("does not support other dialects");

  // Verify that body contains at least one state
  if (op.body().getOps<StateNode>().empty())
    return op.emitOpError() << "must contain at least one state";

  return success();
}

LogicalResult SDFGNode::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the entry attribute references valid state
  FlatSymbolRefAttr entryAttr =
      (*this)->getAttrOfType<FlatSymbolRefAttr>("entry");

  if (!!entryAttr) {
    StateNode entry =
        symbolTable.lookupNearestSymbolFrom<StateNode>(*this, entryAttr);
    if (!entry)
      return emitOpError() << "'" << entryAttr.getValue()
                           << "' does not reference a valid state";
  }

  return success();
}

StateNode SDFGNode::getFirstState() {
  return *body().getOps<StateNode>().begin();
}

StateNode SDFGNode::getStateBySymRef(StringRef symRef) {
  Operation *op = lookupSymbol(symRef);
  return dyn_cast<StateNode>(op);
}

//===----------------------------------------------------------------------===//
// NestedSDFGNode
//===----------------------------------------------------------------------===//

static ParseResult parseNestedSDFGNode(OpAsmParser &parser,
                                       OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  IntegerAttr intAttr =
      parser.getBuilder().getI32IntegerAttr(utils::generateID());
  result.addAttribute("ID", intAttr);

  SmallVector<OpAsmParser::OperandType, 4> args;
  SmallVector<Type, 4> argTypes;

  if (parseAsArgs(parser, result, args, argTypes))
    return failure();

  size_t num_args = result.operands.size();
  result.addAttribute("num_args",
                      parser.getBuilder().getI32IntegerAttr(num_args));

  if (parser.parseArrow() || parseAsArgs(parser, result, args, argTypes))
    return failure();

  if (parseRegion(parser, result, args, argTypes, /*enableShadowing*/ true))
    return failure();

  return success();
}

static void print(OpAsmPrinter &p, NestedSDFGNode op) {
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{"ID", "num_args"});

  printAsArgs(p, op.getOperands(), op.body().getArguments(), 0, op.num_args());
  p << " ->";
  printAsArgs(p, op.getOperands(), op.body().getArguments(), op.num_args(),
              op.getNumOperands());

  p.printRegion(op.body(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true, /*printEmptyBlock=*/true);
}

LogicalResult verify(NestedSDFGNode op) {
  // Verify that no other dialect is used in the body
  for (Operation &oper : op.body().getOps())
    if (oper.getDialect() != (*op).getDialect())
      return op.emitOpError("does not support other dialects");

  // Verify that body contains at least one state
  if (op.body().getOps<StateNode>().empty())
    return op.emitOpError() << "must contain at least one state";

  // Verify that operands and arguments line up
  if (op.getNumOperands() != op.body().getNumArguments())
    op.emitOpError() << "must have matching amount of operands and arguments";

  return success();
}

LogicalResult
NestedSDFGNode::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the entry attribute references valid state
  FlatSymbolRefAttr entryAttr =
      (*this)->getAttrOfType<FlatSymbolRefAttr>("entry");

  if (!!entryAttr) {
    StateNode entry =
        symbolTable.lookupNearestSymbolFrom<StateNode>(*this, entryAttr);
    if (!entry)
      return emitOpError() << "'" << entryAttr.getValue()
                           << "' does not reference a valid state";
  }

  return success();
}

StateNode NestedSDFGNode::getFirstState() {
  return *body().getOps<StateNode>().begin();
}

StateNode NestedSDFGNode::getStateBySymRef(StringRef symRef) {
  Operation *op = lookupSymbol(symRef);
  return dyn_cast<StateNode>(op);
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

  SmallVector<OpAsmParser::OperandType, 4> args;
  SmallVector<Type, 4> argTypes;

  if (parseAsArgs(parser, result, args, argTypes))
    return failure();

  if (parser.parseOptionalArrowTypeList(result.types))
    return failure();

  if (parseRegion(parser, result, args, argTypes, /*enableShadowing*/ true))
    return failure();

  return success();
}

static void print(OpAsmPrinter &p, TaskletNode op) {
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"ID"});
  printAsArgs(p, op.getOperands(), op.body().getArguments(), 0,
              op.getNumOperands());
  p << " -> (" << op.getResultTypes() << ")";
  p.printRegion(op.body(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true, /*printEmptyBlock=*/true);
}

LogicalResult verify(TaskletNode op) {
  // Verify that operands and arguments line up
  if (op.getNumOperands() != op.body().getNumArguments())
    op.emitOpError() << "must have matching amount of operands and arguments";

  return success();
}

TaskletNode TaskletNode::create(PatternRewriter &rewriter, Location location,
                                ValueRange operands, TypeRange results) {
  OpBuilder builder(location->getContext());
  OperationState state(location, getOperationName());
  build(builder, state, results, utils::generateID(), operands);

  TaskletNode task = cast<TaskletNode>(rewriter.createOperation(state));
  rewriter.createBlock(&task.getRegion(), {}, operands.getTypes());
  return task;
}

TaskletNode TaskletNode::create(Location location, ValueRange operands,
                                TypeRange results) {
  OpBuilder builder(location->getContext());
  OperationState state(location, getOperationName());
  build(builder, state, results, utils::generateID(), operands);

  TaskletNode task = cast<TaskletNode>(Operation::create(state));
  builder.createBlock(&task.body(), {}, operands.getTypes());
  return task;
}

std::string TaskletNode::getInputName(unsigned idx) {
  return utils::valueToString(body().getArgument(idx), *getOperation());
}

std::string TaskletNode::getOutputName(unsigned idx) {
  // TODO: Implement multiple return values
  return "__out" + std::to_string(idx);
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
  types.push_back(utils::getSizedType(streamType).getElementType());
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

BlockArgument ConsumeNode::pe() { return body().getArgument(0); }
BlockArgument ConsumeNode::elem() { return body().getArgument(1); }

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
  build(builder, state, from.sym_name(), to.sym_name(),
        rewriter.getStrArrayAttr({}), "1", nullptr);
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

LogicalResult verify(EdgeOp op) {
  // Check that condition is non-empty
  if (op.condition().empty())
    return op.emitOpError() << "condition must be non-empty or omitted";

  return success();
}

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
  p << " (";
  p.printOperands(op.params());
  p << ") : ";
  p << op.getOperation()->getResultTypes();
}

LogicalResult verify(AllocOp op) {
  SizedType res = utils::getSizedType(op.res().getType());

  if (res.getUndefRank() != op.params().size())
    return op.emitOpError("failed to verify that parameter size "
                          "matches undefined dimensions size");

  if (res.hasZeros())
    return op.emitOpError("failed to verify that return type "
                          "doesn't contain dimensions of size zero");

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
    SizedType sized = SizedType::get(res.getContext(), res, {}, {}, {});
    res = ArrayType::get(res.getContext(), sized);
  }

  build(builder, state, res, {}, nameAttr, transient);
  return cast<AllocOp>(Operation::create(state));
}

Type AllocOp::getElementType() {
  return utils::getSizedType(getType()).getElementType();
}

bool AllocOp::isScalar() {
  return utils::getSizedType(getType()).getShape().empty();
}

bool AllocOp::isStream() { return getType().isa<StreamType>(); }

bool AllocOp::isInState() {
  return utils::getParentState(*this->getOperation()) != nullptr;
}

std::string AllocOp::getName() {
  if ((*this)->hasAttr("name")) {
    Attribute nameAttr = (*this)->getAttr("name");
    if (StringAttr name = nameAttr.cast<StringAttr>()) {
      std::string str = name.getValue().str();
      utils::sanitizeName(str);
      return str;
    }
  }

  return utils::valueToString(getResult(), *getOperation());
}

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

  if (utils::isSizedType(t))
    t = utils::getSizedType(t).getElementType();

  SmallVector<Attribute> numList;
  for (size_t i = 0; i < indices.size(); ++i) {
    numList.push_back(builder.getI32IntegerAttr(i));
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
  t = utils::getSizedType(t).getElementType();

  SmallVector<Attribute> numList;
  for (size_t i = 0; i < indices.size(); ++i) {
    numList.push_back(builder.getI32IntegerAttr(i));
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
  size_t mem_size = utils::getSizedType(op.arr().getType()).getRank();

  if (idx_size != mem_size)
    return op.emitOpError("incorrect number of indices for load");

  return success();
}

bool LoadOp::isIndirect() { return !indices().empty(); }

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

StoreOp StoreOp::create(PatternRewriter &rewriter, Location loc, Value val,
                        Value mem, ValueRange indices) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());

  SmallVector<Attribute> numList;
  for (size_t i = 0; i < indices.size(); ++i) {
    numList.push_back(builder.getI32IntegerAttr(i));
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
    numList.push_back(builder.getI32IntegerAttr(i));
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
    numList.push_back(builder.getI32IntegerAttr(-1));
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
  size_t mem_size = utils::getSizedType(op.arr().getType()).getRank();

  if (idx_size != mem_size)
    return op.emitOpError("incorrect number of indices for store");

  return success();
}

bool StoreOp::isIndirect() { return !indices().empty(); }

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
  size_t src_size = utils::getSizedType(op.src().getType()).getRank();
  size_t res_size = utils::getSizedType(op.res().getType()).getRank();

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
  size_t src_size = utils::getSizedType(op.src().getType()).getRank();
  size_t res_size = utils::getSizedType(op.res().getType()).getRank();

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

LogicalResult verify(StreamPushOp op) { return success(); }

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

  SmallVector<Type, 1> returnTypes;
  if (parser.parseOptionalColonTypeList(returnTypes))
    return failure();

  if (parser.resolveOperands(returnOperands, returnTypes,
                             parser.getCurrentLocation(), result.operands))
    return failure();

  return success();
}

static void print(OpAsmPrinter &p, sdfg::ReturnOp op) {
  p.printOptionalAttrDict(op->getAttrs());
  if (op.getNumOperands() > 0)
    p << ' ' << op.input() << " : " << op.input().getTypes();
}

LogicalResult verify(sdfg::ReturnOp op) {
  TaskletNode task = dyn_cast<TaskletNode>(op->getParentOp());

  if (task.getResultTypes() != op.getOperandTypes())
    return op.emitOpError("must match tasklet return types");

  return success();
}

sdfg::ReturnOp sdfg::ReturnOp::create(PatternRewriter &rewriter, Location loc,
                                      mlir::ValueRange input) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());
  build(builder, state, input);
  return cast<sdfg::ReturnOp>(rewriter.createOperation(state));
}

sdfg::ReturnOp sdfg::ReturnOp::create(Location loc, mlir::ValueRange input) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());
  build(builder, state, input);
  return cast<sdfg::ReturnOp>(Operation::create(state));
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
  p << " (" << op.operands() << ")";
  p << " : ";
  p.printFunctionalType(op.operands().getTypes(),
                        op.getOperation()->getResultTypes());
}

LogicalResult verify(LibCallOp op) { return success(); }

std::string LibCallOp::getInputName(unsigned idx) {
  return utils::valueToString(getOperand(idx), *getOperation());
}

std::string LibCallOp::getOutputName(unsigned idx) {
  // TODO: Implement multiple return values
  return "__out" + std::to_string(idx);
}

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
  p << " (";
  p.printAttributeWithoutType(op.symAttr());
  p << ")";
}

LogicalResult verify(AllocSymbolOp op) {
  if (op.sym().empty())
    return op.emitOpError("failed to verify that input string is not empty");

  if (!isalpha(op.sym().front()))
    return op.emitOpError("failed to verify that input string starts with "
                          "an alphabetical character");

  for (char c : op.sym())
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
  p << " (";
  p.printAttributeWithoutType(op.exprAttr());
  p << ") : " << op->getResultTypes();
}

LogicalResult verify(SymOp op) { return success(); }

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "SDFG/Dialect/Ops.cpp.inc"
