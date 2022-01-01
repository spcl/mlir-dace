#include "SDIR/Dialect/Dialect.h"
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
    if (na.first.strref().endswith("numList"))
      numListAttrs.push_back(na.first.strref());

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

static ParseResult parseSDFGNode(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  IntegerAttr intAttr =
      parser.getBuilder().getI32IntegerAttr(SDIRDialect::getNextID());
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

static ParseResult parseStateNode(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  IntegerAttr intAttr =
      parser.getBuilder().getI32IntegerAttr(SDIRDialect::getNextID());
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
      parser.getBuilder().getI32IntegerAttr(SDIRDialect::getNextID());
  result.addAttribute("ID", intAttr);

  auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes,
                          ArrayRef<Type> results,
                          function_like_impl::VariadicFlag, std::string &) {
    return builder.getFunctionType(argTypes, results);
  };

  return function_like_impl::parseFunctionLikeOp(parser, result,
                                                 /*allowVariadic=*/false,
                                                 buildFuncType);
}

static void print(OpAsmPrinter &p, TaskletNode op) {
  FunctionType fnType = op.getType();
  ArrayRef<Type> argTypes = fnType.getInputs();
  ArrayRef<Type> resultTypes = fnType.getResults();
  bool isVariadic = false;
  StringRef visibilityAttrName = SymbolTable::getVisibilityAttrName();

  p << ' ';
  p.printSymbolName(op.sym_name());

  function_like_impl::printFunctionSignature(p, op, argTypes, isVariadic,
                                             resultTypes);
  function_like_impl::printFunctionAttributes(
      p, op, argTypes.size(), resultTypes.size(), {"ID", visibilityAttrName});
  // Print the body if this is not an external function.
  Region &body = op->getRegion(0);
  if (!body.empty())
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
}

LogicalResult verify(TaskletNode op) {
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

  return success();
}

TaskletNode TaskletNode::create(Location location, StringRef name,
                                FunctionType type,
                                ArrayRef<NamedAttribute> attrs) {
  OpBuilder builder(location->getContext());
  OperationState state(location, getOperationName());
  build(builder, state, name, type, attrs);
  return cast<TaskletNode>(Operation::create(state));
}

TaskletNode TaskletNode::create(Location location, StringRef name,
                                FunctionType type,
                                Operation::dialect_attr_range attrs) {
  SmallVector<NamedAttribute, 8> attrRef(attrs);
  return create(location, name, type, makeArrayRef(attrRef));
}

TaskletNode TaskletNode::create(Location location, StringRef name,
                                FunctionType type,
                                ArrayRef<NamedAttribute> attrs,
                                ArrayRef<DictionaryAttr> argAttrs) {
  TaskletNode func = create(location, name, type, attrs);
  func.setAllArgAttrs(argAttrs);
  return func;
}

void TaskletNode::build(OpBuilder &builder, OperationState &state,
                        StringRef name, FunctionType type,
                        ArrayRef<NamedAttribute> attrs,
                        ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getTypeAttrName(), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
  function_like_impl::addArgAndResultAttrs(builder, state, argAttrs,
                                           /*resultAttrs=*/None);
}

void TaskletNode::cloneInto(TaskletNode dest, BlockAndValueMapping &mapper) {
  llvm::MapVector<Identifier, Attribute> newAttrs;
  for (const NamedAttribute &attr : dest->getAttrs())
    newAttrs.insert(attr);
  for (const NamedAttribute &attr : (*this)->getAttrs())
    newAttrs.insert(attr);
  dest->setAttrs(DictionaryAttr::get(getContext(), newAttrs.takeVector()));

  getBody().cloneInto(&dest.getBody(), mapper);
}

TaskletNode TaskletNode::clone(BlockAndValueMapping &mapper) {
  TaskletNode newFunc =
      cast<TaskletNode>(getOperation()->cloneWithoutRegions());

  if (!isExternal()) {
    FunctionType oldType = getType();

    unsigned oldNumArgs = oldType.getNumInputs();
    SmallVector<Type, 4> newInputs;
    newInputs.reserve(oldNumArgs);

    for (unsigned i = 0; i < oldNumArgs; ++i)
      if (!mapper.contains(getArgument(i)))
        newInputs.push_back(oldType.getInput(i));

    if (newInputs.size() != oldNumArgs) {
      newFunc.setType(FunctionType::get(oldType.getContext(), newInputs,
                                        oldType.getResults()));

      if (ArrayAttr argAttrs = getAllArgAttrs()) {
        SmallVector<Attribute> newArgAttrs;
        newArgAttrs.reserve(newInputs.size());
        for (unsigned i = 0; i < oldNumArgs; ++i)
          if (!mapper.contains(getArgument(i)))
            newArgAttrs.push_back(argAttrs[i]);
        newFunc.setAllArgAttrs(newArgAttrs);
      }
    }
  }

  cloneInto(newFunc, mapper);
  return newFunc;
}

TaskletNode TaskletNode::clone() {
  BlockAndValueMapping mapper;
  return clone(mapper);
}

void TaskletNode::setID(unsigned id) {
  Builder builder(*this);
  IntegerAttr intAttr = builder.getI32IntegerAttr(id);
  IDAttr(intAttr);
}

//===----------------------------------------------------------------------===//
// MapNode
//===----------------------------------------------------------------------===//

static ParseResult parseMapNode(OpAsmParser &parser, OperationState &result) {
  Builder &builder = parser.getBuilder();
  IndexType indexType = builder.getIndexType();

  IntegerAttr intAttr =
      parser.getBuilder().getI32IntegerAttr(SDIRDialect::getNextID());
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

  intAttr = parser.getBuilder().getI32IntegerAttr(SDIRDialect::getNextID());
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
      parser.getBuilder().getI32IntegerAttr(SDIRDialect::getNextID());
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

  intAttr = parser.getBuilder().getI32IntegerAttr(SDIRDialect::getNextID());
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

static ParseResult parseEdgeOp(OpAsmParser &parser, OperationState &result) {
  FlatSymbolRefAttr srcAttr;
  FlatSymbolRefAttr destAttr;

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

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
  ArrayType res = op.res().getType().cast<ArrayType>();

  if (res.getUndefRank() != op.params().size())
    return op.emitOpError("failed to verify that parameter size "
                          "matches undefined dimensions size");

  if (res.hasZeros())
    return op.emitOpError("failed to verify that return type "
                          "doesn't contain dimensions of size zero");

  return success();
}

SDFGNode AllocOp::getParentSDFG() {
  Operation *sdfgOrState = (*this)->getParentOp();

  if (SDFGNode sdfg = dyn_cast<SDFGNode>(sdfgOrState))
    return sdfg;

  Operation *sdfg = sdfgOrState->getParentOp();
  return dyn_cast<SDFGNode>(sdfg);
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

  return name;
}

//===----------------------------------------------------------------------===//
// AllocTransientOp
//===----------------------------------------------------------------------===//

static ParseResult parseAllocTransientOp(OpAsmParser &parser,
                                         OperationState &result) {
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

static void print(OpAsmPrinter &p, AllocTransientOp op) {
  p.printOptionalAttrDict(op->getAttrs());
  p << "(";
  p.printOperands(op.params());
  p << ") : ";
  p << op.getOperation()->getResultTypes();
}

LogicalResult verify(AllocTransientOp op) {
  ArrayType res = op.res().getType().cast<ArrayType>();

  if (res.getUndefRank() != op.params().size())
    return op.emitOpError("failed to verify that parameter size matches "
                          "undefined dimensions size");

  if (res.hasZeros())
    return op.emitOpError("failed to verify that return type doesn't "
                          "contain dimensions of size zero");

  return success();
}

SDFGNode AllocTransientOp::getParentSDFG() {
  Operation *sdfgOrState = (*this)->getParentOp();

  if (SDFGNode sdfg = dyn_cast<SDFGNode>(sdfgOrState))
    return sdfg;

  Operation *sdfg = sdfgOrState->getParentOp();
  return dyn_cast<SDFGNode>(sdfg);
}

bool AllocTransientOp::isInState() {
  Operation *sdfgOrState = (*this)->getParentOp();
  if (StateNode state = dyn_cast<StateNode>(sdfgOrState))
    return true;
  return false;
}

//===----------------------------------------------------------------------===//
// GetAccessOp
//===----------------------------------------------------------------------===//

static ParseResult parseGetAccessOp(OpAsmParser &parser,
                                    OperationState &result) {
  IntegerAttr intAttr =
      parser.getBuilder().getI32IntegerAttr(SDIRDialect::getNextID());
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
}

static void print(OpAsmPrinter &p, GetAccessOp op) {
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"ID"});
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

  return name;
}

void GetAccessOp::setID(unsigned id) {
  Builder builder(*this);
  IntegerAttr intAttr = builder.getI32IntegerAttr(id);
  IDAttr(intAttr);
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

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
  size_t mem_size = op.arr().getType().cast<MemletType>().getRank();
  if (idx_size != mem_size)
    return op.emitOpError("incorrect number of indices for load");

  return success();
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

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
  size_t mem_size = op.arr().getType().cast<MemletType>().getRank();
  if (idx_size != mem_size)
    return op.emitOpError("incorrect number of indices for store");

  return success();
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
  size_t src_size = op.src().getType().cast<MemletType>().getRank();
  size_t res_size = op.res().getType().cast<MemletType>().getRank();
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
  size_t src_size = op.src().getType().cast<MemletType>().getRank();
  size_t res_size = op.res().getType().cast<MemletType>().getRank();
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
// AllocStreamOp
//===----------------------------------------------------------------------===//

static ParseResult parseAllocStreamOp(OpAsmParser &parser,
                                      OperationState &result) {
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

static void print(OpAsmPrinter &p, AllocStreamOp op) {
  p.printOptionalAttrDict(op->getAttrs());
  p << "() : ";
  p << op.getOperation()->getResultTypes();
}

LogicalResult verify(AllocStreamOp op) { return success(); }

SDFGNode AllocStreamOp::getParentSDFG() {
  Operation *sdfgOrState = (*this)->getParentOp();

  if (SDFGNode sdfg = dyn_cast<SDFGNode>(sdfgOrState))
    return sdfg;

  Operation *sdfg = sdfgOrState->getParentOp();
  return dyn_cast<SDFGNode>(sdfg);
}

bool AllocStreamOp::isInState() {
  Operation *sdfgOrState = (*this)->getParentOp();
  if (StateNode state = dyn_cast<StateNode>(sdfgOrState))
    return true;
  return false;
}

//===----------------------------------------------------------------------===//
// AllocTransientStreamOp
//===----------------------------------------------------------------------===//

static ParseResult parseAllocTransientStreamOp(OpAsmParser &parser,
                                               OperationState &result) {
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

static void print(OpAsmPrinter &p, AllocTransientStreamOp op) {
  p.printOptionalAttrDict(op->getAttrs());
  p << "() : ";
  p << op.getOperation()->getResultTypes();
}

LogicalResult verify(AllocTransientStreamOp op) { return success(); }

SDFGNode AllocTransientStreamOp::getParentSDFG() {
  Operation *sdfgOrState = (*this)->getParentOp();

  if (SDFGNode sdfg = dyn_cast<SDFGNode>(sdfgOrState))
    return sdfg;

  Operation *sdfg = sdfgOrState->getParentOp();
  return dyn_cast<SDFGNode>(sdfg);
}

bool AllocTransientStreamOp::isInState() {
  Operation *sdfgOrState = (*this)->getParentOp();
  if (StateNode state = dyn_cast<StateNode>(sdfgOrState))
    return true;
  return false;
}

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

LogicalResult verify(sdir::ReturnOp op) { return success(); }

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

static ParseResult parseCallOp(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  Attribute calleeAttr;
  if (parser.parseAttribute(calleeAttr, parser.getBuilder().getNoneType(),
                            "callee", result.attributes))
    return failure();

  SmallVector<OpAsmParser::OperandType, 4> operands;
  if (parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren))
    return failure();

  FunctionType func;
  if (parser.parseColonType(func))
    return failure();

  ArrayRef<Type> operandsTypes = func.getInputs();
  ArrayRef<Type> resultTypes = func.getResults();
  result.addTypes(resultTypes);

  if (parser.resolveOperands(operands, operandsTypes,
                             parser.getCurrentLocation(), result.operands))
    return failure();

  return success();
}

static void print(OpAsmPrinter &p, sdir::CallOp op) {
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"callee"});
  p << ' ';
  p.printAttributeWithoutType(op.calleeAttr());
  p << "(" << op.operands() << ")";
  p << " : ";
  p.printFunctionalType(op.operands().getTypes(),
                        op.getOperation()->getResultTypes());
}

LogicalResult verify(sdir::CallOp op) { return success(); }

LogicalResult
sdir::CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  FlatSymbolRefAttr fnAttr =
      (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");

  TaskletNode fnT =
      symbolTable.lookupNearestSymbolFrom<TaskletNode>(*this, fnAttr);
  SDFGNode fnS = symbolTable.lookupNearestSymbolFrom<SDFGNode>(*this, fnAttr);

  if (!fnT && !fnS) {
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid tasklet or SDFG";
  }
  // Verify that the operand and result types match the callee.
  FunctionType fnType = !fnT ? fnS.getType() : fnT.getType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0; i < fnType.getNumInputs(); ++i)
    if (getOperand(i).getType() != fnType.getInput(i))
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;

  if (fnType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for callee");

  for (unsigned i = 0; i < fnType.getNumResults(); ++i)
    if (getResult(i).getType() != fnType.getResult(i)) {
      InFlightDiagnostic diag = emitOpError("result type mismatch at index ")
                                << i;
      diag.attachNote() << "      op result types: " << getResultTypes();
      diag.attachNote() << "function result types: " << fnType.getResults();
      return diag;
    }

  return success();
}

StateNode sdir::CallOp::getParentState() {
  Operation *stateOrMapConsume = (*this)->getParentOp();

  if (StateNode state = dyn_cast<StateNode>(stateOrMapConsume))
    return state;

  Operation *state = stateOrMapConsume->getParentOp();
  return dyn_cast<StateNode>(state);
}

TaskletNode sdir::CallOp::getTasklet() {
  StateNode state = getParentState();
  Operation *task = state.lookupSymbol(callee());
  TaskletNode tasklet = dyn_cast<TaskletNode>(task);
  return tasklet;
}

SDFGNode sdir::CallOp::getSDFG() {
  StateNode state = getParentState();
  Operation *op = state.lookupSymbol(callee());
  SDFGNode sdfg = dyn_cast<SDFGNode>(op);
  return sdfg;
}

bool sdir::CallOp::callsTasklet() { return getTasklet() != nullptr; }

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
    if (!isalnum(c))
      return op.emitOpError("failed to verify that input string only "
                            "contains alphanumeric characters");

  return success();
}

//===----------------------------------------------------------------------===//
// SymOp
//===----------------------------------------------------------------------===//

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
