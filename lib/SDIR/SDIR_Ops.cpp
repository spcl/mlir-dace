#include "SDIR/SDIR_Dialect.h"

using namespace mlir;
using namespace mlir::sdir;

//===----------------------------------------------------------------------===//
// SDFGNode
//===----------------------------------------------------------------------===//

static ParseResult parseSDFGNode(OpAsmParser &parser, OperationState &result) {
    if (parser.parseOptionalAttrDict(result.attributes))
        return failure();

    StringAttr sym_nameAttr;
    if (parser.parseSymbolName(sym_nameAttr, "sym_name", result.attributes))
        return failure();

    Region *body = result.addRegion();
    if (parser.parseRegion(*body))
        return failure();

    if(body->empty()) body->emplaceBlock();

    return success();
}

static void print(OpAsmPrinter &p, SDFGNode op) {
    p.printNewline();
    p << op.getOperationName();
    p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"sym_name"});
    p << ' ';
    p.printSymbolName(op.sym_name());
    p.printRegion(op.region());
}

LogicalResult SDFGNode::verifySymbolUses(SymbolTableCollection &symbolTable) {
    // Check that the entry attribute is specified.
    auto entryAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("entry");
    if (!entryAttr)
        return emitOpError("requires a 'src' symbol reference attribute");
    StateNode entry = symbolTable.lookupNearestSymbolFrom<StateNode>(*this, entryAttr);
    if (!entry)
        return emitOpError() << "'" << entryAttr.getValue()
                            << "' does not reference a valid state";

    return success();
}

//===----------------------------------------------------------------------===//
// StateNode
//===----------------------------------------------------------------------===//

static ParseResult parseStateNode(OpAsmParser &parser, OperationState &result) {
    if (parser.parseOptionalAttrDict(result.attributes))
        return failure();

    StringAttr sym_nameAttr;
    if (parser.parseSymbolName(sym_nameAttr, "sym_name", result.attributes))
        return failure();

    Region *body = result.addRegion();
    if (parser.parseRegion(*body))
        return failure();

    if(body->empty()) body->emplaceBlock();

    return success();
}

static void print(OpAsmPrinter &p, StateNode op) {
    p.printNewline();
    p << op.getOperationName();
    p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"sym_name"});
    p << ' ';
    p.printSymbolName(op.sym_name());
    p.printRegion(op.region());
}

//===----------------------------------------------------------------------===//
// TaskletNode
//===----------------------------------------------------------------------===//

TaskletNode TaskletNode::create(Location location, StringRef name, FunctionType type, 
                                        ArrayRef<NamedAttribute> attrs) {
    OpBuilder builder(location->getContext());
    OperationState state(location, getOperationName());
    build(builder, state, name, type, attrs);
    return cast<TaskletNode>(Operation::create(state));
}

TaskletNode TaskletNode::create(Location location, StringRef name, FunctionType type, 
                                        Operation::dialect_attr_range attrs) {
    SmallVector<NamedAttribute, 8> attrRef(attrs);
    return create(location, name, type, llvm::makeArrayRef(attrRef));
}

TaskletNode TaskletNode::create(Location location, StringRef name, FunctionType type, 
                                        ArrayRef<NamedAttribute> attrs, 
                                        ArrayRef<DictionaryAttr> argAttrs) {
    TaskletNode func = create(location, name, type, attrs);
    func.setAllArgAttrs(argAttrs);
    return func;
}

void TaskletNode::build(OpBuilder &builder, OperationState &state, StringRef name,
                FunctionType type, ArrayRef<NamedAttribute> attrs,
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
                                            /*resultAttrs=*/llvm::None);
}

void TaskletNode::cloneInto(TaskletNode dest, BlockAndValueMapping &mapper) {
    llvm::MapVector<Identifier, Attribute> newAttrs;
    for (const auto &attr : dest->getAttrs())
        newAttrs.insert(attr);
    for (const auto &attr : (*this)->getAttrs())
        newAttrs.insert(attr);
    dest->setAttrs(DictionaryAttr::get(getContext(), newAttrs.takeVector()));

    getBody().cloneInto(&dest.getBody(), mapper);
}

TaskletNode TaskletNode::clone(BlockAndValueMapping &mapper) {
    TaskletNode newFunc = cast<TaskletNode>(getOperation()->cloneWithoutRegions());
    
    if (!isExternal()) {
        FunctionType oldType = getType();

        unsigned oldNumArgs = oldType.getNumInputs();
        SmallVector<Type, 4> newInputs;
        newInputs.reserve(oldNumArgs);

        for (unsigned i = 0; i != oldNumArgs; ++i)
            if (!mapper.contains(getArgument(i)))
                newInputs.push_back(oldType.getInput(i));

        if (newInputs.size() != oldNumArgs) {
            newFunc.setType(FunctionType::get(oldType.getContext(), newInputs,
                                            oldType.getResults()));

            if (ArrayAttr argAttrs = getAllArgAttrs()) {
                SmallVector<Attribute> newArgAttrs;
                newArgAttrs.reserve(newInputs.size());
                for (unsigned i = 0; i != oldNumArgs; ++i)
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

//===----------------------------------------------------------------------===//
// MapNode
//===----------------------------------------------------------------------===//

static ParseResult parseMapNode(OpAsmParser &parser, OperationState &result) {
    auto &builder = parser.getBuilder();
    auto indexType = builder.getIndexType();

    if(parser.parseOptionalAttrDict(result.attributes))
        return failure();

    SmallVector<OpAsmParser::OperandType, 4> ivs;
    if (parser.parseRegionArgumentList(ivs, OpAsmParser::Delimiter::Paren))
        return failure();

    if(parser.parseEqual()) return failure();

    AffineMapAttr lbMapAttr;
    NamedAttrList lbAttrs;
    SmallVector<OpAsmParser::OperandType, 4> lbMapOperands;
    if(parser.parseAffineMapOfSSAIds(lbMapOperands, lbMapAttr,
                                     "lowerBounds", lbAttrs,
                                      OpAsmParser::Delimiter::Paren)) 
        return failure();

    SmallVector<int64_t, 4> lb;
    auto lbMap = lbMapAttr.getValue();
    for (const auto &result : lbMap.getResults()) {
      auto constExpr = result.dyn_cast<AffineConstantExpr>();
      if (!constExpr)
        return parser.emitError(parser.getNameLoc(),
                                "lower bound must be constant integers");
      lb.push_back(constExpr.getValue());
    }

    result.addAttribute("lowerBounds", builder.getI64ArrayAttr(lb));

    if(parser.parseKeyword("to")) return failure();

    AffineMapAttr ubMapAttr;
    NamedAttrList ubAttrs;
    SmallVector<OpAsmParser::OperandType, 4> ubMapOperands;
    if(parser.parseAffineMapOfSSAIds(ubMapOperands, ubMapAttr,
                                      "upperBounds", ubAttrs,
                                      OpAsmParser::Delimiter::Paren)) 
        return failure();

    SmallVector<int64_t, 4> ub;
    auto ubMap = ubMapAttr.getValue();
    for (const auto &result : ubMap.getResults()) {
      auto constExpr = result.dyn_cast<AffineConstantExpr>();
      if (!constExpr)
        return parser.emitError(parser.getNameLoc(),
                                "upper bound must be constant integers");
      ub.push_back(constExpr.getValue());
    }

    
    result.addAttribute("upperBounds", builder.getI64ArrayAttr(ub));

    if(parser.parseKeyword("step")) return failure();

    AffineMapAttr stepsMapAttr;
    NamedAttrList stepsAttrs;
    SmallVector<OpAsmParser::OperandType, 4> stepsMapOperands;
    if(parser.parseAffineMapOfSSAIds(stepsMapOperands, stepsMapAttr,
                                      "steps", stepsAttrs,
                                      OpAsmParser::Delimiter::Paren)) 
        return failure();

    SmallVector<int64_t, 4> steps;
    auto stepsMap = stepsMapAttr.getValue();
    for (const auto &result : stepsMap.getResults()) {
      auto constExpr = result.dyn_cast<AffineConstantExpr>();
      if (!constExpr)
        return parser.emitError(parser.getNameLoc(),
                                "steps must be constant integers");
      steps.push_back(constExpr.getValue());
    }

    result.addAttribute("steps", builder.getI64ArrayAttr(steps));

    // Now parse the body.
    Region *body = result.addRegion();
    SmallVector<Type, 4> types(ivs.size(), indexType);
    if (parser.parseRegion(*body, ivs, types))
        return failure();

    return success();
}

static void print(OpAsmPrinter &p, MapNode op) {
    p.printNewline();
    p << op.getOperationName();
    p.printOptionalAttrDict(op->getAttrs(), 
                /*elidedAttrs=*/{"lowerBounds", "upperBounds", "steps"}); 
    p << " (" << op.getBody()->getArguments() << ") = (";

    SmallVector<int64_t, 8> lbresult;
    for (Attribute attr : op.lowerBounds()) {
        lbresult.push_back(attr.cast<IntegerAttr>().getInt());
    }
    llvm::interleaveComma(lbresult, p);

    p << ") to (";

    SmallVector<int64_t, 8> ubresult;
    for (Attribute attr : op.upperBounds()) {
        ubresult.push_back(attr.cast<IntegerAttr>().getInt());
    }
    llvm::interleaveComma(ubresult, p);

    p << ") step (";

    SmallVector<int64_t, 8> stepresult;
    for (Attribute attr : op.steps()) {
        stepresult.push_back(attr.cast<IntegerAttr>().getInt());
    }
    llvm::interleaveComma(stepresult, p);

    p << ")";

    p.printRegion(op.region(), /*printEntryBlockArgs=*/false, 
        /*printBlockTerminators=*/false);
}

LogicalResult verify(MapNode op){
    size_t var_count = op.getBody()->getArguments().size();

    if(op.lowerBounds().size() != var_count)
        return op.emitOpError("failed to verify that size of lower bounds matches size of arguments");
    
    if(op.upperBounds().size() != var_count)
        return op.emitOpError("failed to verify that size of upper bounds matches size of arguments");
    
    if(op.steps().size() != var_count)
        return op.emitOpError("failed to verify that size of steps matches size of arguments");

    return success();
}

bool MapNode::isDefinedOutsideOfLoop(Value value){
    return !region().isAncestor(value.getParentRegion());
}

Region &MapNode::getLoopBody(){
    return region();
}

LogicalResult MapNode::moveOutOfLoop(ArrayRef<Operation *> ops){
    return failure();
}

//===----------------------------------------------------------------------===//
// ConsumeNode
//===----------------------------------------------------------------------===//

static ParseResult parseConsumeNode(OpAsmParser &parser, OperationState &result) {
    if(parser.parseOptionalAttrDict(result.attributes))
        return failure();

    if(parser.parseLParen()) return failure();

    OpAsmParser::OperandType stream;
    Type streamType;
    if(parser.parseOperand(stream) || parser.parseColonType(streamType)
            || parser.resolveOperand(stream, streamType, result.operands)
            || !streamType.isa<StreamType>())
        return failure();

    if(parser.parseRParen() || parser.parseArrow() || parser.parseLParen()) 
        return failure();

    SmallVector<OpAsmParser::OperandType, 4> ivs;
    OpAsmParser::OperandType num_pes_op;
    if(parser.parseKeyword("pe") || parser.parseColon()
            || parser.parseOperand(num_pes_op)) 
        return failure();
    ivs.push_back(num_pes_op);

    if(parser.parseComma()) return failure();

    OpAsmParser::OperandType elem_op;
    if(parser.parseKeyword("elem") || parser.parseColon()
            || parser.parseOperand(elem_op)) 
        return failure();
    ivs.push_back(elem_op);

    if(parser.parseRParen())
        return failure();

    // Now parse the body.
    Region *body = result.addRegion();
    SmallVector<Type, 4> types;
    types.push_back(parser.getBuilder().getIndexType());
    types.push_back(streamType.cast<StreamType>().getElementType());
    if (parser.parseRegion(*body, ivs, types))
        return failure();

    return success();
}

static void print(OpAsmPrinter &p, ConsumeNode op) {
    p.printNewline();
    p << op.getOperationName();
    p.printOptionalAttrDict(op->getAttrs()); 
    p << " (" << op.stream() << " : " << op.stream().getType() << ")";
    p << " -> (pe: " << op.getBody()->getArgument(0) << ", elem: " << op.getBody()->getArgument(1) << ")";
    p.printRegion(op.region(), /*printEntryBlockArgs=*/false, 
        /*printBlockTerminators=*/false);
}

LogicalResult verify(ConsumeNode op){
    if(op.num_pes().hasValue() && op.num_pes().getValue().isNonPositive())
        return op.emitOpError("failed to verify that number of processing elements is at least one");

    return success();
}

LogicalResult ConsumeNode::verifySymbolUses(SymbolTableCollection &symbolTable) {
    // Check that the condition attributes are specified.
    auto condAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("condition");
    if (!condAttr)
        return success();

    FuncOp cond = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, condAttr);
    if (!cond)
        return emitOpError() << "'" << condAttr.getValue()
                            << "' does not reference a valid func";
    
    if(cond.getArguments().size() != 1)
        return emitOpError() << "'" << condAttr.getValue()
                            << "' references a func with invalid signature";

    if(cond.getArgument(0).getType() != stream().getType())
        return emitOpError() << "'" << condAttr.getValue()
                            << "' references a func with invalid signature";
    
    return success();
}

bool ConsumeNode::isDefinedOutsideOfLoop(Value value){
    return !region().isAncestor(value.getParentRegion());
}

Region &ConsumeNode::getLoopBody(){
    return region();
}

LogicalResult ConsumeNode::moveOutOfLoop(ArrayRef<Operation *> ops){
    return failure();
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
    // Check that the callee attribute was specified.
    auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
    if (!fnAttr)
        return emitOpError("requires a 'callee' symbol reference attribute");
    TaskletNode fn = symbolTable.lookupNearestSymbolFrom<TaskletNode>(*this, fnAttr);
    if (!fn)
        return emitOpError() << "'" << fnAttr.getValue()
                            << "' does not reference a valid tasklet";

    // Verify that the operand and result types match the callee.
    auto fnType = fn.getType();
    if (fnType.getNumInputs() != getNumOperands())
        return emitOpError("incorrect number of operands for callee");

    for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (getOperand(i).getType() != fnType.getInput(i))
        return emitOpError("operand type mismatch: expected operand type ")
            << fnType.getInput(i) << ", but provided "
            << getOperand(i).getType() << " for operand number " << i;

    if (fnType.getNumResults() != getNumResults())
        return emitOpError("incorrect number of results for callee");

    for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
    if (getResult(i).getType() != fnType.getResult(i)) {
        auto diag = emitOpError("result type mismatch at index ") << i;
        diag.attachNote() << "      op result types: " << getResultTypes();
        diag.attachNote() << "function result types: " << fnType.getResults();
        return diag;
    }

    return success();
}

//===----------------------------------------------------------------------===//
// EdgeOp
//===----------------------------------------------------------------------===//

LogicalResult EdgeOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
    // Check that the src/dest attributes are specified.
    auto srcAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("src");
    if (!srcAttr)
        return emitOpError("requires a 'src' symbol reference attribute");
    StateNode src = symbolTable.lookupNearestSymbolFrom<StateNode>(*this, srcAttr);
    if (!src)
        return emitOpError() << "'" << srcAttr.getValue()
                            << "' does not reference a valid state";
    
    auto destAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("dest");
    if (!destAttr)
        return emitOpError("requires a 'dest' symbol reference attribute");
    StateNode dest = symbolTable.lookupNearestSymbolFrom<StateNode>(*this, destAttr);
    if (!dest)
        return emitOpError() << "'" << destAttr.getValue()
                            << "' does not reference a valid state";

    return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "SDIR/SDIR_Ops.cpp.inc"
