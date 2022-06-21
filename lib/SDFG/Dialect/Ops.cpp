#include "SDFG/Dialect/Dialect.h"
#include "SDFG/Utils/Utils.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
                               SmallVector<OpAsmParser::Argument, 4> &args,
                               bool enableShadowing) {
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, args, enableShadowing))
    return failure();

  if (body->empty())
    return parser.emitError(parser.getCurrentLocation(),
                            "expected non-empty body");
  return success();
}

static ParseResult parseArgsList(OpAsmParser &parser,
                                 SmallVector<OpAsmParser::Argument, 4> &args) {
  if (parser.parseLParen())
    return failure();

  for (unsigned i = 0; parser.parseOptionalRParen().failed(); ++i) {
    if (i > 0 && parser.parseComma())
      return failure();

    OpAsmParser::Argument arg;

    if (parser.parseArgument(arg, /*allowType=*/true))
      return failure();

    args.push_back(arg);
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
                               SmallVector<OpAsmParser::Argument, 4> &args) {
  if (parser.parseLParen())
    return failure();

  for (unsigned i = 0; parser.parseOptionalRParen().failed(); ++i) {
    if (i > 0 && parser.parseComma())
      return failure();

    OpAsmParser::UnresolvedOperand operand;
    OpAsmParser::Argument arg;

    if (parser.parseOperand(operand))
      return failure();

    if (parser.parseOptionalKeyword("as").succeeded()) {
      if (parser.parseArgument(arg, /*allowType=*/true))
        return failure();

    } else {
      Type type;

      if (parser.parseColonType(type))
        return failure();

      arg.type = type;
      arg.ssaName = operand;
    }

    if (parser.resolveOperand(operand, arg.type, result.operands))
      return failure();

    args.push_back(arg);
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
  SmallVector<OpAsmParser::UnresolvedOperand> opList;
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

    OpAsmParser::UnresolvedOperand op;
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

  if (parser.resolveOperands(opList, parser.getBuilder().getIndexType(),
                             result.operands))
    return failure();

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
  SDFGNode sdfg = cast<SDFGNode>(rewriter.create(state));

  std::vector<Location> locs = {};
  for (unsigned i = 0; i < args.size(); ++i)
    locs.push_back(loc);

  rewriter.createBlock(&sdfg.getRegion(), {}, args, locs);
  return sdfg;
}

ParseResult SDFGNode::parse(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  IntegerAttr intAttr =
      parser.getBuilder().getI32IntegerAttr(utils::generateID());
  result.addAttribute("ID", intAttr);

  SmallVector<OpAsmParser::Argument, 4> args;

  if (parseArgsList(parser, args))
    return failure();

  result.addAttribute("num_args",
                      parser.getBuilder().getI32IntegerAttr(args.size()));

  if (parser.parseArrow() || parseArgsList(parser, args))
    return failure();

  if (parseRegion(parser, result, args, /*enableShadowing*/ true))
    return failure();

  return success();
}

void SDFGNode::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"ID", "num_args"});

  printArgsList(p, body().getArguments(), 0, num_args());
  p << " ->";
  printArgsList(p, body().getArguments(), num_args(), body().getNumArguments());

  p.printRegion(body(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true, /*printEmptyBlock=*/true);
}

LogicalResult SDFGNode::verify() {
  // Verify that no other dialect is used in the body
  for (Operation &oper : body().getOps())
    if (oper.getDialect() != (*this)->getDialect())
      return emitOpError("does not support other dialects");

  // Verify that body contains at least one state
  if (body().getOps<StateNode>().empty())
    return emitOpError() << "must contain at least one state";

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

ParseResult NestedSDFGNode::parse(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  IntegerAttr intAttr =
      parser.getBuilder().getI32IntegerAttr(utils::generateID());
  result.addAttribute("ID", intAttr);

  SmallVector<OpAsmParser::Argument, 4> args;

  if (parseAsArgs(parser, result, args))
    return failure();

  size_t num_args = result.operands.size();
  result.addAttribute("num_args",
                      parser.getBuilder().getI32IntegerAttr(num_args));

  if (parser.parseArrow() || parseAsArgs(parser, result, args))
    return failure();

  if (parseRegion(parser, result, args, /*enableShadowing*/ true))
    return failure();

  return success();
}

void NestedSDFGNode::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"ID", "num_args"});

  printAsArgs(p, getOperands(), body().getArguments(), 0, num_args());
  p << " ->";
  printAsArgs(p, getOperands(), body().getArguments(), num_args(),
              getNumOperands());

  p.printRegion(body(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true, /*printEmptyBlock=*/true);
}

LogicalResult NestedSDFGNode::verify() {
  // Verify that no other dialect is used in the body
  for (Operation &oper : body().getOps())
    if (oper.getDialect() != (*this)->getDialect())
      return emitOpError("does not support other dialects");

  // Verify that body contains at least one state
  if (body().getOps<StateNode>().empty())
    return emitOpError() << "must contain at least one state";

  // Verify that operands and arguments line up
  if (getNumOperands() != body().getNumArguments())
    emitOpError() << "must have matching amount of operands and arguments";

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
  StateNode stateNode = cast<StateNode>(rewriter.create(state));
  rewriter.createBlock(&stateNode.body());
  return stateNode;
}

StateNode StateNode::create(Location loc, StringRef name) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());
  build(builder, state, utils::generateID(), utils::generateName(name.str()));
  return cast<StateNode>(Operation::create(state));
}

ParseResult StateNode::parse(OpAsmParser &parser, OperationState &result) {
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

void StateNode::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"ID", "sym_name"});
  p << ' ';
  p.printSymbolName(sym_name());
  p.printRegion(body());
}

LogicalResult StateNode::verify() {
  // Verify that no other dialect is used in the body
  // Except func operations
  for (Operation &oper : body().getOps())
    if (oper.getDialect() != (*this)->getDialect() &&
        !dyn_cast<func::FuncOp>(oper))
      return emitOpError("does not support other dialects");
  return success();
}

//===----------------------------------------------------------------------===//
// TaskletNode
//===----------------------------------------------------------------------===//

ParseResult TaskletNode::parse(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  IntegerAttr intAttr =
      parser.getBuilder().getI32IntegerAttr(utils::generateID());
  result.addAttribute("ID", intAttr);

  SmallVector<OpAsmParser::Argument, 4> args;

  if (parseAsArgs(parser, result, args))
    return failure();

  if (parser.parseOptionalArrowTypeList(result.types))
    return failure();

  if (parseRegion(parser, result, args, /*enableShadowing*/ true))
    return failure();

  return success();
}

void TaskletNode::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"ID"});
  printAsArgs(p, getOperands(), body().getArguments(), 0, getNumOperands());
  p << " -> (" << getResultTypes() << ")";
  p.printRegion(body(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true, /*printEmptyBlock=*/true);
}

LogicalResult TaskletNode::verify() {
  // Verify that operands and arguments line up
  if (getNumOperands() != body().getNumArguments())
    emitOpError() << "must have matching amount of operands and arguments";

  return success();
}

TaskletNode TaskletNode::create(PatternRewriter &rewriter, Location location,
                                ValueRange operands, TypeRange results) {
  OpBuilder builder(location->getContext());
  OperationState state(location, getOperationName());
  build(builder, state, results, utils::generateID(), operands);

  TaskletNode task = cast<TaskletNode>(rewriter.create(state));

  std::vector<Location> locs = {};
  for (unsigned i = 0; i < operands.size(); ++i)
    locs.push_back(location);

  rewriter.createBlock(&task.getRegion(), {}, operands.getTypes(), locs);
  return task;
}

TaskletNode TaskletNode::create(Location location, ValueRange operands,
                                TypeRange results) {
  OpBuilder builder(location->getContext());
  OperationState state(location, getOperationName());
  build(builder, state, results, utils::generateID(), operands);

  TaskletNode task = cast<TaskletNode>(Operation::create(state));

  std::vector<Location> locs = {};
  for (unsigned i = 0; i < operands.size(); ++i)
    locs.push_back(location);

  builder.createBlock(&task.body(), {}, operands.getTypes(), locs);
  return task;
}

std::string TaskletNode::getInputName(unsigned idx) {
  return utils::valueToString(body().getArgument(idx), *getOperation());
}

std::string TaskletNode::getOutputName(unsigned idx) {
  return "__out" + std::to_string(idx);
}

//===----------------------------------------------------------------------===//
// MapNode
//===----------------------------------------------------------------------===//

ParseResult MapNode::parse(OpAsmParser &parser, OperationState &result) {
  IntegerAttr intAttr =
      parser.getBuilder().getI32IntegerAttr(utils::generateID());
  result.addAttribute("entryID", intAttr);

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  SmallVector<OpAsmParser::Argument, 4> ivs;
  if (parser.parseArgumentList(ivs, OpAsmParser::Delimiter::Paren))
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

  for (unsigned i = 0; i < ivs.size(); ++i)
    ivs[i].type = parser.getBuilder().getIndexType();

  // Now parse the body.
  if (parseRegion(parser, result, ivs, /*enableShadowing=*/false))
    return failure();

  intAttr = parser.getBuilder().getI32IntegerAttr(utils::generateID());
  result.addAttribute("exitID", intAttr);
  return success();
}

void MapNode::print(OpAsmPrinter &p) {
  printOptionalAttrDictNoNumList(
      p, (*this)->getAttrs(),
      {"entryID", "exitID", "lowerBounds", "upperBounds", "steps"});

  p << " (" << getBody()->getArguments() << ") = (";

  printNumberList(p, getOperation(), "lowerBounds");

  p << ") to (";

  printNumberList(p, getOperation(), "upperBounds");

  p << ") step (";

  printNumberList(p, getOperation(), "steps");

  p << ")";

  p.printRegion(body(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

LogicalResult MapNode::verify() {
  size_t var_count = getBody()->getArguments().size();

  if (getNumListSize(*this, "lowerBounds") != var_count)
    return emitOpError("failed to verify that size of "
                       "lower bounds matches size of arguments");

  if (getNumListSize(*this, "upperBounds") != var_count)
    return emitOpError("failed to verify that size of "
                       "upper bounds matches size of arguments");

  if (getNumListSize(*this, "steps") != var_count)
    return emitOpError("failed to verify that size of "
                       "steps matches size of arguments");

  // Verify that no other dialect is used in the body
  for (Operation &oper : body().getOps())
    if (oper.getDialect() != (*this)->getDialect())
      return emitOpError("does not support other dialects");

  return success();
}

Region &MapNode::getLoopBody() { return body(); }

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

ParseResult ConsumeNode::parse(OpAsmParser &parser, OperationState &result) {
  IntegerAttr intAttr =
      parser.getBuilder().getI32IntegerAttr(utils::generateID());
  result.addAttribute("entryID", intAttr);

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (parser.parseLParen())
    return failure();

  OpAsmParser::UnresolvedOperand stream;
  Type streamType;
  if (parser.parseOperand(stream) || parser.parseColonType(streamType) ||
      parser.resolveOperand(stream, streamType, result.operands) ||
      !streamType.isa<StreamType>())
    return failure();

  if (parser.parseRParen() || parser.parseArrow() || parser.parseLParen())
    return failure();

  SmallVector<OpAsmParser::Argument, 4> ivs;
  OpAsmParser::Argument num_pes_op;
  if (parser.parseKeyword("pe") || parser.parseColon() ||
      parser.parseArgument(num_pes_op))
    return failure();
  num_pes_op.type = parser.getBuilder().getIndexType();
  ivs.push_back(num_pes_op);

  if (parser.parseComma())
    return failure();

  OpAsmParser::Argument elem_op;
  if (parser.parseKeyword("elem") || parser.parseColon() ||
      parser.parseArgument(elem_op))
    return failure();
  elem_op.type = utils::getSizedType(streamType).getElementType();
  ivs.push_back(elem_op);

  if (parser.parseRParen())
    return failure();

  // Now parse the body.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, ivs))
    return failure();

  intAttr = parser.getBuilder().getI32IntegerAttr(utils::generateID());
  result.addAttribute("exitID", intAttr);
  return success();
}

void ConsumeNode::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"entryID", "exitID"});
  p << " (" << stream() << " : " << stream().getType() << ")";
  p << " -> (pe: " << getBody()->getArgument(0);
  p << ", elem: " << getBody()->getArgument(1) << ")";
  p.printRegion(body(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

LogicalResult ConsumeNode::verify() {
  if (num_pes().hasValue() && num_pes().getValue().isNonPositive())
    return emitOpError("failed to verify that number of "
                       "processing elements is at least one");

  // Verify that no other dialect is used in the body
  for (Operation &oper : body().getOps())
    if (oper.getDialect() != (*this)->getDialect())
      return emitOpError("does not support other dialects");

  return success();
}

LogicalResult
ConsumeNode::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the condition attribute is specified.
  FlatSymbolRefAttr condAttr =
      (*this)->getAttrOfType<FlatSymbolRefAttr>("condition");
  if (!condAttr)
    return success();

  func::FuncOp cond =
      symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, condAttr);
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

Region &ConsumeNode::getLoopBody() { return body(); }

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
  return cast<EdgeOp>(rewriter.create(state));
}

EdgeOp EdgeOp::create(PatternRewriter &rewriter, Location loc, StateNode &from,
                      StateNode &to) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());
  build(builder, state, from.sym_name(), to.sym_name(),
        rewriter.getStrArrayAttr({}), "1", nullptr);
  return cast<EdgeOp>(rewriter.create(state));
}

EdgeOp EdgeOp::create(Location loc, StateNode &from, StateNode &to,
                      ArrayAttr &assign, StringAttr &condition, Value ref) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());
  build(builder, state, from.sym_name(), to.sym_name(), assign, condition, ref);
  return cast<EdgeOp>(Operation::create(state));
}

ParseResult EdgeOp::parse(OpAsmParser &parser, OperationState &result) {
  FlatSymbolRefAttr srcAttr;
  FlatSymbolRefAttr destAttr;

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (parser.parseOptionalLParen().succeeded()) {
    OpAsmParser::UnresolvedOperand op;
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

void EdgeOp::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"src", "dest"});
  p << ' ';
  if (!refMutable().empty())
    p << "(ref: " << ref() << ": " << ref().getType() << ") ";
  p.printAttributeWithoutType(srcAttr());
  p << " -> ";
  p.printAttributeWithoutType(destAttr());
}

LogicalResult EdgeOp::verify() {
  // Check that condition is non-empty
  if (condition().empty())
    return emitOpError() << "condition must be non-empty or omitted";

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

ParseResult AllocOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  SmallVector<OpAsmParser::UnresolvedOperand, 4> paramsOperands;
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

void AllocOp::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " (";
  p.printOperands(params());
  p << ") : ";
  p << getOperation()->getResultTypes();
}

LogicalResult AllocOp::verify() {
  SizedType result = utils::getSizedType(res().getType());

  if (result.getUndefRank() != params().size())
    return emitOpError("failed to verify that parameter size "
                       "matches undefined dimensions size");

  if (result.hasZeros())
    return emitOpError("failed to verify that return type "
                       "doesn't contain dimensions of size zero");

  return success();
}

AllocOp AllocOp::create(PatternRewriter &rewriter, Location loc, Type res,
                        StringRef name, bool transient) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());
  StringAttr nameAttr = rewriter.getStringAttr(utils::generateName(name.str()));
  build(builder, state, res, {}, nameAttr, transient);
  return cast<AllocOp>(rewriter.create(state));
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
  return cast<LoadOp>(rewriter.create(state));
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

ParseResult LoadOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  OpAsmParser::UnresolvedOperand memletOperand;
  if (parser.parseOperand(memletOperand))
    return failure();

  if (parser.parseLSquare())
    return failure();

  if (parseNumberList(parser, result, "indices"))
    return failure();

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

void LoadOp::print(OpAsmPrinter &p) {
  printOptionalAttrDictNoNumList(p, (*this)->getAttrs(),
                                 /*elidedAttrs*/ {"indices"});
  p << ' ' << arr();
  p << "[";
  printNumberList(p, getOperation(), "indices");
  p << "]";
  p << " : ";
  p << ArrayRef<Type>(arr().getType());
  p << " -> ";
  p << ArrayRef<Type>(res().getType());
}

LogicalResult LoadOp::verify() {
  size_t idx_size = getNumListSize(getOperation(), "indices");
  size_t mem_size = utils::getSizedType(arr().getType()).getRank();

  if (idx_size != mem_size)
    return emitOpError("incorrect number of indices for load");

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
  return cast<StoreOp>(rewriter.create(state));
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

ParseResult StoreOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  OpAsmParser::UnresolvedOperand valOperand;
  if (parser.parseOperand(valOperand))
    return failure();
  if (parser.parseComma())
    return failure();

  OpAsmParser::UnresolvedOperand memletOperand;
  if (parser.parseOperand(memletOperand))
    return failure();

  if (parser.parseLSquare())
    return failure();

  if (parseNumberList(parser, result, "indices"))
    return failure();

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

void StoreOp::print(OpAsmPrinter &p) {
  printOptionalAttrDictNoNumList(p, (*this)->getAttrs(),
                                 /*elidedAttrs=*/{"indices"});
  p << ' ' << val() << "," << ' ' << arr();
  p << "[";
  printNumberList(p, getOperation(), "indices");
  p << "]";
  p << " : ";
  p << ArrayRef<Type>(val().getType());
  p << " -> ";
  p << ArrayRef<Type>(arr().getType());
}

LogicalResult StoreOp::verify() {
  size_t idx_size = getNumListSize(getOperation(), "indices");
  size_t mem_size = utils::getSizedType(arr().getType()).getRank();

  if (idx_size != mem_size)
    return emitOpError("incorrect number of indices for store");

  return success();
}

bool StoreOp::isIndirect() { return !indices().empty(); }

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

ParseResult CopyOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  OpAsmParser::UnresolvedOperand srcOperand;
  if (parser.parseOperand(srcOperand))
    return failure();

  if (parser.parseArrow())
    return failure();

  OpAsmParser::UnresolvedOperand destOperand;
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

void CopyOp::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs());
  p << ' ' << src() << " -> " << dest();
  p << " : ";
  p << ArrayRef<Type>(src().getType());
}

LogicalResult CopyOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// MemletCastOp
//===----------------------------------------------------------------------===//

ParseResult MemletCastOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  OpAsmParser::UnresolvedOperand memletOperand;
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

void MemletCastOp::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs());
  p << ' ' << src();
  p << " : ";
  p << ArrayRef<Type>(src().getType());
  p << " -> ";
  p << getOperation()->getResultTypes();
}

LogicalResult MemletCastOp::verify() {
  size_t src_size = utils::getSizedType(src().getType()).getRank();
  size_t res_size = utils::getSizedType(res().getType()).getRank();

  if (src_size != res_size)
    return emitOpError("incorrect rank for memlet_cast");

  return success();
}

//===----------------------------------------------------------------------===//
// ViewCastOp
//===----------------------------------------------------------------------===//

ParseResult ViewCastOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  OpAsmParser::UnresolvedOperand memletOperand;
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

void ViewCastOp::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs());
  p << ' ' << src();
  p << " : ";
  p << ArrayRef<Type>(src().getType());
  p << " -> ";
  p << getOperation()->getResultTypes();
}

LogicalResult ViewCastOp::verify() {
  size_t src_size = utils::getSizedType(src().getType()).getRank();
  size_t res_size = utils::getSizedType(res().getType()).getRank();

  if (src_size != res_size)
    return emitOpError("incorrect rank for view_cast");

  return success();
}

//===----------------------------------------------------------------------===//
// SubviewOp
//===----------------------------------------------------------------------===//

ParseResult SubviewOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  OpAsmParser::UnresolvedOperand memletOperand;
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

void SubviewOp::print(OpAsmPrinter &p) {
  printOptionalAttrDictNoNumList(p, (*this)->getAttrs(),
                                 {"offsets", "sizes", "strides"});
  p << ' ' << src() << "[";
  printNumberList(p, getOperation(), "offsets");
  p << "][";
  printNumberList(p, getOperation(), "sizes");
  p << "][";
  printNumberList(p, getOperation(), "strides");
  p << "]";
  p << " : ";
  p << ArrayRef<Type>(src().getType());
  p << " -> ";
  p << getOperation()->getResultTypes();
}

LogicalResult SubviewOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// StreamPopOp
//===----------------------------------------------------------------------===//

ParseResult StreamPopOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  OpAsmParser::UnresolvedOperand streamOperand;
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

void StreamPopOp::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs());
  p << ' ' << str();
  p << " : ";
  p << ArrayRef<Type>(str().getType());
  p << " -> ";
  p << ArrayRef<Type>(res().getType());
}

LogicalResult StreamPopOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// StreamPushOp
//===----------------------------------------------------------------------===//

ParseResult StreamPushOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  OpAsmParser::UnresolvedOperand valOperand;
  if (parser.parseOperand(valOperand))
    return failure();
  if (parser.parseComma())
    return failure();

  OpAsmParser::UnresolvedOperand streamOperand;
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

void StreamPushOp::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs());
  p << ' ' << val() << ", " << str();
  p << " : ";
  p << ArrayRef<Type>(val().getType());
  p << " -> ";
  p << ArrayRef<Type>(str().getType());
}

LogicalResult StreamPushOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// StreamLengthOp
//===----------------------------------------------------------------------===//

ParseResult StreamLengthOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  OpAsmParser::UnresolvedOperand streamOperand;
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

void StreamLengthOp::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs());
  p << ' ' << str();
  p << " : ";
  p << ArrayRef<Type>(str().getType());
  p << " -> ";
  p << getOperation()->getResultTypes();
}

LogicalResult StreamLengthOp::verify() {
  Operation *parent = (*this)->getParentOp();
  if (parent == nullptr)
    return emitOpError("must be in a StateNode, MapNode or FuncOp");

  if (isa<StateNode>(parent) || isa<MapNode>(parent) ||
      isa<func::FuncOp>(parent)) {
    return success();
  }

  return emitOpError("must be in a StateNode, MapNode or FuncOp");
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

ParseResult sdfg::ReturnOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  SmallVector<OpAsmParser::UnresolvedOperand, 4> returnOperands;
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

void sdfg::ReturnOp::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs());
  if (getNumOperands() > 0)
    p << ' ' << input() << " : " << input().getTypes();
}

LogicalResult sdfg::ReturnOp::verify() {
  TaskletNode task = dyn_cast<TaskletNode>((*this)->getParentOp());

  if (task.getResultTypes() != getOperandTypes())
    return emitOpError("must match tasklet return types");

  return success();
}

sdfg::ReturnOp sdfg::ReturnOp::create(PatternRewriter &rewriter, Location loc,
                                      mlir::ValueRange input) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());
  build(builder, state, input);
  return cast<sdfg::ReturnOp>(rewriter.create(state));
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

ParseResult LibCallOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  Attribute calleeAttr;
  if (parser.parseAttribute(calleeAttr, parser.getBuilder().getNoneType(),
                            "callee", result.attributes))
    return failure();

  SmallVector<OpAsmParser::UnresolvedOperand, 4> operandsOperands;
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

void LibCallOp::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict(getOperation()->getAttrs(),
                          /*elidedAttrs=*/{"callee"});
  p << ' ';
  p.printAttributeWithoutType(calleeAttr());
  p << " (" << operands() << ")";
  p << " : ";
  p.printFunctionalType(operands().getTypes(),
                        getOperation()->getResultTypes());
}

LogicalResult LibCallOp::verify() { return success(); }

std::string LibCallOp::getInputName(unsigned idx) {
  if (getOperation()->hasAttr("inputs")) {
    if (ArrayAttr inputs =
            getOperation()->getAttr("inputs").dyn_cast<ArrayAttr>()) {
      if (idx <= inputs.size() && inputs[idx].isa<StringAttr>()) {
        return inputs[idx].cast<StringAttr>().getValue().str();
      }
    }
  }

  return utils::valueToString(getOperand(idx), *getOperation());
}

std::string LibCallOp::getOutputName(unsigned idx) {
  if (getOperation()->hasAttr("outputs")) {
    if (ArrayAttr outputs =
            getOperation()->getAttr("outputs").dyn_cast<ArrayAttr>()) {
      if (idx <= outputs.size() && outputs[idx].isa<StringAttr>()) {
        return outputs[idx].cast<StringAttr>().getValue().str();
      }
    }
  }

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
  return cast<AllocSymbolOp>(rewriter.create(state));
}

AllocSymbolOp AllocSymbolOp::create(Location loc, StringRef sym) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());
  build(builder, state, sym);
  return cast<AllocSymbolOp>(Operation::create(state));
}

ParseResult AllocSymbolOp::parse(OpAsmParser &parser, OperationState &result) {
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

void AllocSymbolOp::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict(getOperation()->getAttrs(), /*elidedAttrs=*/{"sym"});
  p << " (";
  p.printAttributeWithoutType(symAttr());
  p << ")";
}

LogicalResult AllocSymbolOp::verify() {
  if (sym().empty())
    return emitOpError("failed to verify that input string is not empty");

  if (!isalpha(sym().front()))
    return emitOpError("failed to verify that input string starts with "
                       "an alphabetical character");

  for (char c : sym())
    if (!isalnum(c) && c != '_')
      return emitOpError("failed to verify that input string only "
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
  return cast<SymOp>(rewriter.create(state));
}

SymOp SymOp::create(Location loc, Type type, StringRef expr) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());
  build(builder, state, type, expr);
  return cast<SymOp>(Operation::create(state));
}

ParseResult SymOp::parse(OpAsmParser &parser, OperationState &result) {
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

void SymOp::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict(getOperation()->getAttrs(), /*elidedAttrs=*/{"expr"});
  p << " (";
  p.printAttributeWithoutType(exprAttr());
  p << ") : " << getOperation()->getResultTypes();
}

LogicalResult SymOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "SDFG/Dialect/Ops.cpp.inc"
