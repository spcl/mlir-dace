// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

#include "SDFG/Dialect/Dialect.h"
#include "SDFG/Utils/Utils.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;
using namespace sdfg;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Parses a non-empty region.
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

/// Parses a list of arguments.
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

/// Prints a list of arguments in human-readable form.
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

/// Parses arguments with an optional "as" keyword to compactly represent
/// arguments and parameters.
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

/// Prints a list of arguments with an optional "as" keyword in human-readable
/// form.
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

/// There are 3 possible values that can be used as a number: symbols, integers
/// and operands. Operands are stored as regular operands. Symbols as stringAttr
/// and integers as int32 Attr. In order to encode the correct order of values
/// we use an auxiliary attr called [attrName]_numList.
/// The numList contains int32 Attrs with the following encoding:
/// Positive int n: nth operand
/// Negative int n: -nth - 1 Attribute (symbol or integer) in [attrName]
static ParseResult parseNumberList(OpAsmParser &parser, OperationState &result,
                                   StringRef attrName) {
  SmallVector<OpAsmParser::UnresolvedOperand> opList;
  SmallVector<Attribute> attrList;
  SmallVector<Attribute> numList;
  int opIdx = result.operands.size();
  int attrIdx = 1;

  do {
    if (parser.parseOptionalKeyword("sym").succeeded()) {
      StringAttr stringAttr;
      if (parser.parseLParen() ||
          parser.parseAttribute(stringAttr,
                                parser.getBuilder().getNoneType()) ||
          parser.parseRParen())
        return failure();

      attrList.push_back(stringAttr);
      numList.push_back(parser.getBuilder().getI32IntegerAttr(-attrIdx));
      attrIdx++;
      continue;
    }

    int32_t num = -1;
    OptionalParseResult intOPR = parser.parseOptionalInteger(num);
    if (intOPR.has_value() && intOPR.value().succeeded()) {
      Attribute intAttr = parser.getBuilder().getI32IntegerAttr(num);
      attrList.push_back(intAttr);
      numList.push_back(parser.getBuilder().getI32IntegerAttr(-attrIdx));
      attrIdx++;
      continue;
    }

    OpAsmParser::UnresolvedOperand op;
    OptionalParseResult opOPR = parser.parseOptionalOperand(op);
    if (opOPR.has_value() && opOPR.value().succeeded()) {
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

/// Prints a list of number arguments in human-readable form.
static void printNumberList(OpAsmPrinter &p, Operation *op,
                            StringRef attrName) {
  ArrayAttr attrList = op->getAttr(attrName).cast<ArrayAttr>();
  ArrayAttr numList =
      op->getAttr(attrName.str() + "_numList").cast<ArrayAttr>();

  for (unsigned i = 0; i < numList.size(); ++i) {
    Attribute numAttr = numList[i];
    IntegerAttr num = numAttr.cast<IntegerAttr>();
    if (i > 0)
      p << ", ";

    if (num.getValue().isNegative()) {
      Attribute attr = attrList[-num.getInt() - 1];

      if (attr.isa<StringAttr>()) {
        p << "sym(" << attr << ")";
      } else {
        p.printAttributeWithoutType(attr);
      }

    } else {
      Value val = op->getOperand(num.getInt());
      p.printOperand(val);
    }
  }
}

/// Prints a list of optional attributes excluding the number list in
/// human-readable form.
static void
printOptionalAttrDictNoNumList(OpAsmPrinter &p, ArrayRef<NamedAttribute> attrs,
                               ArrayRef<StringRef> elidedAttrs = {}) {
  SmallVector<StringRef> numListAttrs(elidedAttrs.begin(), elidedAttrs.end());

  for (NamedAttribute na : attrs)
    if (na.getName().strref().endswith("numList"))
      numListAttrs.push_back(na.getName().strref());

  p.printOptionalAttrDict(attrs, /*elidedAttrs=*/numListAttrs);
}

/// Returns the length of the number list, which is equivalent to the number of
/// numeric arguments.
static size_t getNumListSize(Operation *op, StringRef attrName) {
  ArrayAttr numList =
      op->getAttr(attrName.str() + "_numList").cast<ArrayAttr>();
  return numList.size();
}

//===----------------------------------------------------------------------===//
// SDFGNode
//===----------------------------------------------------------------------===//

/// Builds, creates and inserts a SDFG node using the provided PatternRewriter.
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

/// Builds, creates and inserts a SDFG node using the provided PatternRewriter.
SDFGNode SDFGNode::create(PatternRewriter &rewriter, Location loc) {
  return create(rewriter, loc, 0, {});
}

/// Attempts to parse a SDFG node.
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

/// Prints a SDFG node in human-readable form.
void SDFGNode::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"ID", "num_args"});

  printArgsList(p, getBody().getArguments(), 0, getNumArgs());
  p << " ->";
  printArgsList(p, getBody().getArguments(), getNumArgs(),
                getBody().getNumArguments());

  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true, /*printEmptyBlock=*/true);
}

/// Verifies the correct structure of a SDFG node.
LogicalResult SDFGNode::verify() {
  // Verify that no other dialect is used in the body
  for (Operation &oper : getBody().getOps())
    if (oper.getDialect() != (*this)->getDialect())
      return emitOpError("does not support other dialects");

  // Verify that body contains at least one state
  if (getBody().getOps<StateNode>().empty())
    return emitOpError() << "must contain at least one state";

  return success();
}

/// Verifies the correct structure of symbols in a SDFG node.
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

/// Returns the first state in the SDFG node.
StateNode SDFGNode::getFirstState() {
  return *getBody().getOps<StateNode>().begin();
}

/// Returns the state with the provided name (symbol) in the SDFG node.
StateNode SDFGNode::getStateBySymRef(StringRef symRef) {
  Operation *op = lookupSymbol(symRef);
  return dyn_cast<StateNode>(op);
}

/// Returns the entry state of the SDFG node.
StateNode SDFGNode::getEntryState() {
  if (this->getEntry().has_value())
    return getStateBySymRef(this->getEntry().value());

  return this->getFirstState();
}

/// Returns the list of arguments in the SDFG node.
Block::BlockArgListType SDFGNode::getArgs() {
  return this->getBody().getArguments().take_front(getNumArgs());
}

/// Returns a list of argument types in the SDFG node.
TypeRange SDFGNode::getArgTypes() {
  SmallVector<Type> types = {};
  for (BlockArgument BArg : getArgs()) {
    types.push_back(BArg.getType());
  }
  return TypeRange(types);
}

/// Returns the list of results in the SDFG node.
Block::BlockArgListType SDFGNode::getResults() {
  return this->getBody().getArguments().drop_front(getNumArgs());
}

/// Returns a list of result types in the SDFG node.
TypeRange SDFGNode::getResultTypes() {
  SmallVector<Type> types = {};
  for (BlockArgument BArg : getResults()) {
    types.push_back(BArg.getType());
  }
  return TypeRange(types);
}

//===----------------------------------------------------------------------===//
// NestedSDFGNode
//===----------------------------------------------------------------------===//

/// Builds, creates and inserts a nested SDFG node using the provided
/// PatternRewriter.
NestedSDFGNode NestedSDFGNode::create(PatternRewriter &rewriter, Location loc,
                                      unsigned num_args, ValueRange args) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());

  build(builder, state, utils::generateID(), nullptr, num_args, args);
  NestedSDFGNode sdfg = cast<NestedSDFGNode>(rewriter.create(state));

  std::vector<Location> locs = {};
  for (unsigned i = 0; i < args.size(); ++i)
    locs.push_back(args[i].getLoc());

  rewriter.createBlock(&sdfg.getRegion(), {}, args.getTypes(), locs);
  return sdfg;
}

/// Builds, creates and inserts a nested SDFG node using the provided
/// PatternRewriter.
NestedSDFGNode NestedSDFGNode::create(PatternRewriter &rewriter, Location loc) {
  return create(rewriter, loc, 0, {});
}

/// Attempts to parse a nested SDFG node.
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

/// Prints a nested SDFG node in human-readable form.
void NestedSDFGNode::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"ID", "num_args"});

  printAsArgs(p, getOperands(), getBody().getArguments(), 0, getNumArgs());
  p << " ->";
  printAsArgs(p, getOperands(), getBody().getArguments(), getNumArgs(),
              getNumOperands());

  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true, /*printEmptyBlock=*/true);
}

/// Verifies the correct structure of a nested SDFG node.
LogicalResult NestedSDFGNode::verify() {
  // Verify that no other dialect is used in the body
  for (Operation &oper : getBody().getOps())
    if (oper.getDialect() != (*this)->getDialect())
      return emitOpError("does not support other dialects");

  // Verify that body contains at least one state
  if (getBody().getOps<StateNode>().empty())
    return emitOpError() << "must contain at least one state";

  // Verify that operands and arguments line up
  if (getNumOperands() != getBody().getNumArguments())
    emitOpError() << "must have matching amount of operands and arguments";

  return success();
}

/// Verifies the correct structure of symbols in a nested SDFG node.
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

/// Returns the first state in the nested SDFG node.
StateNode NestedSDFGNode::getFirstState() {
  return *getBody().getOps<StateNode>().begin();
}

/// Returns the state with the provided name (symbol) in the nested SDFG node.
StateNode NestedSDFGNode::getStateBySymRef(StringRef symRef) {
  Operation *op = lookupSymbol(symRef);
  return dyn_cast<StateNode>(op);
}

/// Returns the entry state of the nested SDFG node.
StateNode NestedSDFGNode::getEntryState() {
  if (this->getEntry().has_value())
    return getStateBySymRef(this->getEntry().value());

  return this->getFirstState();
}

/// Returns the list of arguments in the nested SDFG node.
ValueRange NestedSDFGNode::getArgs() {
  return this->getOperands().take_front(getNumArgs());
}

/// Returns the list of results in the nested SDFG node.
ValueRange NestedSDFGNode::getResults() {
  return this->getOperands().drop_front(getNumArgs());
}

//===----------------------------------------------------------------------===//
// StateNode
//===----------------------------------------------------------------------===//

/// Builds, creates and inserts a state node using the provided PatternRewriter.
StateNode StateNode::create(PatternRewriter &rewriter, Location loc,
                            StringRef name) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());
  build(builder, state, utils::generateID(), utils::generateName(name.str()));
  StateNode stateNode = cast<StateNode>(rewriter.create(state));
  rewriter.createBlock(&stateNode.getBody());
  return stateNode;
}

/// Builds, creates and inserts a state node using the provided PatternRewriter.
StateNode StateNode::create(PatternRewriter &rewriter, Location loc) {
  return create(rewriter, loc, "state");
}

/// Builds, creates and inserts a state node using Operation::create.
StateNode StateNode::create(Location loc, StringRef name) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());
  build(builder, state, utils::generateID(), utils::generateName(name.str()));
  return cast<StateNode>(Operation::create(state));
}

/// Attempts to parse a state node.
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

/// Prints a state node in human-readable form.
void StateNode::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"ID", "sym_name"});
  p << ' ';
  p.printSymbolName(getSymName());
  p.printRegion(getBody());
}

/// Verifies the correct structure of a state node.
LogicalResult StateNode::verify() {
  // Verify that no other dialect is used in the body
  // Except func operations
  for (Operation &oper : getBody().getOps())
    if (oper.getDialect() != (*this)->getDialect() &&
        !dyn_cast<func::FuncOp>(oper))
      return emitOpError("does not support other dialects");
  return success();
}

//===----------------------------------------------------------------------===//
// TaskletNode
//===----------------------------------------------------------------------===//

/// Builds, creates and inserts a tasklet node using the provided
/// PatternRewriter.
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

/// Builds, creates and inserts a tasklet node using Operation::create.
TaskletNode TaskletNode::create(Location location, ValueRange operands,
                                TypeRange results) {
  OpBuilder builder(location->getContext());
  OperationState state(location, getOperationName());
  build(builder, state, results, utils::generateID(), operands);

  TaskletNode task = cast<TaskletNode>(Operation::create(state));

  std::vector<Location> locs = {};
  for (unsigned i = 0; i < operands.size(); ++i)
    locs.push_back(location);

  builder.createBlock(&task.getBody(), {}, operands.getTypes(), locs);
  return task;
}

/// Attempts to parse a tasklet node.
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

/// Prints a tasklet node in human-readable form.
void TaskletNode::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"ID"});
  printAsArgs(p, getOperands(), getBody().getArguments(), 0, getNumOperands());
  p << " -> (" << getResultTypes() << ")";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true, /*printEmptyBlock=*/true);
}

/// Verifies the correct structure of a tasklet node.
LogicalResult TaskletNode::verify() {
  // Verify that operands and arguments line up
  if (getNumOperands() != getBody().getNumArguments())
    emitOpError() << "must have matching amount of operands and arguments";

  return success();
}

/// Returns the input name of the provided index.
std::string TaskletNode::getInputName(unsigned idx) {
  return utils::valueToString(getBody().getArgument(idx), *getOperation());
}

/// Returns the output name of the provided index.
std::string TaskletNode::getOutputName(unsigned idx) {
  return "__out" + std::to_string(idx);
}

//===----------------------------------------------------------------------===//
// MapNode
//===----------------------------------------------------------------------===//

/// Attempts to parse a map node.
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

/// Prints a map node in human-readable form.
void MapNode::print(OpAsmPrinter &p) {
  printOptionalAttrDictNoNumList(
      p, (*this)->getAttrs(),
      {"entryID", "exitID", "lowerBounds", "upperBounds", "steps"});

  p << " (" << getBody().getArguments() << ") = (";

  printNumberList(p, getOperation(), "lowerBounds");

  p << ") to (";

  printNumberList(p, getOperation(), "upperBounds");

  p << ") step (";

  printNumberList(p, getOperation(), "steps");

  p << ")";

  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

/// Verifies the correct structure of a map node.
LogicalResult MapNode::verify() {
  size_t var_count = getBody().getArguments().size();

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
  for (Operation &oper : getBody().getOps())
    if (oper.getDialect() != (*this)->getDialect())
      return emitOpError("does not support other dialects");

  return success();
}

/// Returns the body of the map node.
Region &MapNode::getLoopBody() { return getBody(); }

//===----------------------------------------------------------------------===//
// ConsumeNode
//===----------------------------------------------------------------------===//

/// Attempts to parse a consume node.
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

/// Prints a consume node in human-readable form.
void ConsumeNode::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"entryID", "exitID"});
  p << " (" << getStream() << " : " << getStream().getType() << ")";
  p << " -> (pe: " << getBody().getArgument(0);
  p << ", elem: " << getBody().getArgument(1) << ")";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

/// Verifies the correct structure of a consume node.
LogicalResult ConsumeNode::verify() {
  if (getNumPes().has_value() && getNumPes().value().isNonPositive())
    return emitOpError("failed to verify that number of "
                       "processing elements is at least one");

  // Verify that no other dialect is used in the body
  for (Operation &oper : getBody().getOps())
    if (oper.getDialect() != (*this)->getDialect())
      return emitOpError("does not support other dialects");

  return success();
}

/// Verifies the correct structure of symbols in a consume node.
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

  if (cond.getArgument(0).getType() != getStream().getType())
    return emitOpError() << "'" << condAttr.getValue()
                         << "' references a func with invalid signature";

  return success();
}

/// Returns the body of the consume node.
Region &ConsumeNode::getLoopBody() { return getBody(); }
/// Returns the argument corresponding to the processing element.
BlockArgument ConsumeNode::pe() { return getBody().getArgument(0); }
/// Returns the argument corresponding to the popped element.
BlockArgument ConsumeNode::elem() { return getBody().getArgument(1); }

//===----------------------------------------------------------------------===//
// EdgeOp
//===----------------------------------------------------------------------===//

/// Builds, creates and inserts an edge using the provided PatternRewriter.
EdgeOp EdgeOp::create(PatternRewriter &rewriter, Location loc, StateNode &from,
                      StateNode &to, ArrayAttr &assign, StringAttr &condition,
                      Value ref) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());
  build(builder, state, from.getSymName(), to.getSymName(), assign, condition,
        ref);
  return cast<EdgeOp>(rewriter.create(state));
}

/// Builds, creates and inserts an edge using the provided PatternRewriter.
EdgeOp EdgeOp::create(PatternRewriter &rewriter, Location loc, StateNode &from,
                      StateNode &to) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());
  build(builder, state, from.getSymName(), to.getSymName(),
        rewriter.getStrArrayAttr({}), "1", nullptr);
  return cast<EdgeOp>(rewriter.create(state));
}

/// Builds, creates and inserts an edge using Operation::create.
EdgeOp EdgeOp::create(Location loc, StateNode &from, StateNode &to,
                      ArrayAttr &assign, StringAttr &condition, Value ref) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());
  build(builder, state, from.getSymName(), to.getSymName(), assign, condition,
        ref);
  return cast<EdgeOp>(Operation::create(state));
}

/// Attempts to parse a edge operation.
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
        parser.parseRParen() || parser.resolveOperand(op, t, valList))
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

/// Prints a edge operation in human-readable form.
void EdgeOp::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"src", "dest"});
  p << ' ';
  if (!getRefMutable().empty())
    p << "(ref: " << getRef() << ": " << getRef().getType() << ") ";
  p.printAttributeWithoutType(getSrcAttr());
  p << " -> ";
  p.printAttributeWithoutType(getDestAttr());
}

/// Verifies the correct structure of an edge operation.
LogicalResult EdgeOp::verify() {
  // Check that condition is non-empty
  if (getCondition().empty())
    return emitOpError() << "condition must be non-empty or omitted";

  return success();
}

/// Verifies the correct structure of symbols in an edge operation.
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
  p.printOperands(getParams());
  p << ") : ";
  p << getOperation()->getResultTypes();
}

LogicalResult AllocOp::verify() {
  SizedType result = utils::getSizedType(getRes().getType());

  if (result.getUndefRank() != getParams().size())
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

std::string AllocOp::getContainerName() {
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
  p << ' ' << getArr();
  p << "[";
  printNumberList(p, getOperation(), "indices");
  p << "]";
  p << " : ";
  p << ArrayRef<Type>(getArr().getType());
  p << " -> ";
  p << ArrayRef<Type>(getRes().getType());
}

LogicalResult LoadOp::verify() {
  size_t idx_size = getNumListSize(getOperation(), "indices");
  size_t mem_size = utils::getSizedType(getArr().getType()).getRank();

  if (idx_size != mem_size)
    return emitOpError("incorrect number of indices for load");

  return success();
}

bool LoadOp::isIndirect() { return !getIndices().empty(); }

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
  p << ' ' << getVal() << "," << ' ' << getArr();
  p << "[";
  printNumberList(p, getOperation(), "indices");
  p << "]";
  p << " : ";
  p << ArrayRef<Type>(getVal().getType());
  p << " -> ";
  p << ArrayRef<Type>(getArr().getType());
}

LogicalResult StoreOp::verify() {
  size_t idx_size = getNumListSize(getOperation(), "indices");
  size_t mem_size = utils::getSizedType(getArr().getType()).getRank();

  if (idx_size != mem_size)
    return emitOpError("incorrect number of indices for store");

  return success();
}

bool StoreOp::isIndirect() { return !getIndices().empty(); }

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

CopyOp CopyOp::create(PatternRewriter &rewriter, Location loc, Value src,
                      Value dst) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());

  // Makes sure that src and destination type match (reduces symbols)
  dst.setType(src.getType());

  build(builder, state, src, dst);
  return cast<CopyOp>(rewriter.create(state));
}

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
  p << ' ' << getSrc() << " -> " << getDest();
  p << " : ";
  p << ArrayRef<Type>(getSrc().getType());
}

LogicalResult CopyOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// ViewCastOp
//===----------------------------------------------------------------------===//

ViewCastOp ViewCastOp::create(PatternRewriter &rewriter, Location loc,
                              Value array, Type type) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());
  build(builder, state, type, array);
  return cast<ViewCastOp>(rewriter.create(state));
}

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

  if (parser.resolveOperand(memletOperand, srcType, result.operands))
    return failure();

  return success();
}

void ViewCastOp::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs());
  p << ' ' << getSrc();
  p << " : ";
  p << ArrayRef<Type>(getSrc().getType());
  p << " -> ";
  p << getOperation()->getResultTypes();
}

LogicalResult ViewCastOp::verify() {
  size_t src_size = utils::getSizedType(getSrc().getType()).getRank();
  size_t res_size = utils::getSizedType(getRes().getType()).getRank();

  if (src_size != res_size)
    return emitOpError("incorrect rank for view_cast");

  return success();
}

//===----------------------------------------------------------------------===//
// SubviewOp
//===----------------------------------------------------------------------===//

SubviewOp SubviewOp::create(PatternRewriter &rewriter, Location loc, Type res,
                            Value src, ArrayAttr offsets, ArrayAttr sizes,
                            ArrayAttr strides) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());

  ArrayAttr numArr = rewriter.getArrayAttr({});

  state.addAttribute("offsets_numList", numArr);
  state.addAttribute("sizes_numList", numArr);
  state.addAttribute("strides_numList", numArr);

  build(builder, state, res, src, offsets, sizes, strides);
  return cast<SubviewOp>(rewriter.create(state));
}

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

  if (parser.resolveOperand(memletOperand, srcType, result.operands))
    return failure();

  return success();
}

void SubviewOp::print(OpAsmPrinter &p) {
  printOptionalAttrDictNoNumList(p, (*this)->getAttrs(),
                                 {"offsets", "sizes", "strides"});
  p << ' ' << getSrc() << "[";
  printNumberList(p, getOperation(), "offsets");
  p << "][";
  printNumberList(p, getOperation(), "sizes");
  p << "][";
  printNumberList(p, getOperation(), "strides");
  p << "]";
  p << " : ";
  p << ArrayRef<Type>(getSrc().getType());
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
  p << ' ' << getStr();
  p << " : ";
  p << ArrayRef<Type>(getStr().getType());
  p << " -> ";
  p << ArrayRef<Type>(getRes().getType());
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
  p << ' ' << getVal() << ", " << getStr();
  p << " : ";
  p << ArrayRef<Type>(getVal().getType());
  p << " -> ";
  p << ArrayRef<Type>(getStr().getType());
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

  if (parser.resolveOperand(streamOperand, streamType, result.operands))
    return failure();

  return success();
}

void StreamLengthOp::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs());
  p << ' ' << getStr();
  p << " : ";
  p << ArrayRef<Type>(getStr().getType());
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
    p << ' ' << getInput() << " : " << getInput().getTypes();
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

LibCallOp LibCallOp::create(PatternRewriter &rewriter, Location loc,
                            TypeRange result, StringRef callee,
                            ValueRange operands) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());
  build(builder, state, result, rewriter.getStringAttr(callee), operands);
  return cast<LibCallOp>(rewriter.create(state));
}

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
  p.printAttributeWithoutType(getCalleeAttr());
  p << " (" << getOperands() << ")";
  p << " : ";
  p.printFunctionalType(getOperands().getTypes(),
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
  p.printAttributeWithoutType(getSymAttr());
  p << ")";
}

LogicalResult AllocSymbolOp::verify() {
  if (getSym().empty())
    return emitOpError("failed to verify that input string is not empty");

  if (!isalpha(getSym().front()) && getSym().front() != '_')
    return emitOpError("failed to verify that input string starts with "
                       "an alphabetical character");

  for (char c : getSym())
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
  p.printAttributeWithoutType(getExprAttr());
  p << ") : " << getOperation()->getResultTypes();
}

LogicalResult SymOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

/// Generate the code for operation definitions.
#define GET_OP_CLASSES
#include "SDFG/Dialect/Ops.cpp.inc"
