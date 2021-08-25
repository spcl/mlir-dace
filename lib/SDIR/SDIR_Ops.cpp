#include "SDIR/SDIR_Dialect.h"

#define GET_OP_CLASSES
#include "SDIR/SDIR_Ops.cpp.inc"

using namespace mlir;

//===----------------------------------------------------------------------===//
// TaskletNode
//===----------------------------------------------------------------------===//

sdir::TaskletNode sdir::TaskletNode::create(Location location, StringRef name, FunctionType type, 
                                        ArrayRef<NamedAttribute> attrs) {
    OpBuilder builder(location->getContext());
    OperationState state(location, getOperationName());
    build(builder, state, name, type, attrs);
    return cast<TaskletNode>(Operation::create(state));
}

sdir::TaskletNode sdir::TaskletNode::create(Location location, StringRef name, FunctionType type, 
                                        Operation::dialect_attr_range attrs) {
    SmallVector<NamedAttribute, 8> attrRef(attrs);
    return create(location, name, type, llvm::makeArrayRef(attrRef));
}

sdir::TaskletNode sdir::TaskletNode::create(Location location, StringRef name, FunctionType type, 
                                        ArrayRef<NamedAttribute> attrs, 
                                        ArrayRef<DictionaryAttr> argAttrs) {
    TaskletNode func = create(location, name, type, attrs);
    func.setAllArgAttrs(argAttrs);
    return func;
}

void sdir::TaskletNode::build(OpBuilder &builder, OperationState &state, StringRef name,
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

void sdir::TaskletNode::cloneInto(TaskletNode dest, BlockAndValueMapping &mapper) {
    llvm::MapVector<Identifier, Attribute> newAttrs;
    for (const auto &attr : dest->getAttrs())
        newAttrs.insert(attr);
    for (const auto &attr : (*this)->getAttrs())
        newAttrs.insert(attr);
    dest->setAttrs(DictionaryAttr::get(getContext(), newAttrs.takeVector()));

    getBody().cloneInto(&dest.getBody(), mapper);
}

sdir::TaskletNode sdir::TaskletNode::clone(BlockAndValueMapping &mapper) {
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

sdir::TaskletNode sdir::TaskletNode::clone() {
    BlockAndValueMapping mapper;
    return clone(mapper);
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult sdir::CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
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
// SDFGNode
//===----------------------------------------------------------------------===//

LogicalResult sdir::SDFGNode::verifySymbolUses(SymbolTableCollection &symbolTable) {
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
// EdgeOp
//===----------------------------------------------------------------------===//

LogicalResult sdir::EdgeOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
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
// MapNode
//===----------------------------------------------------------------------===//

ParseResult sdir::MapNode::parseMapOp(OpAsmParser &parser, OperationState &result) {
    auto &builder = parser.getBuilder();
    auto indexType = builder.getIndexType();

    SmallVector<OpAsmParser::OperandType, 4> ivs;
    if (parser.parseRegionArgumentList(ivs, OpAsmParser::Delimiter::Paren))
        return failure();

    if(parser.parseEqual()) return failure();

    AffineMapAttr lbMapAttr;
    NamedAttrList lbAttrs;
    SmallVector<OpAsmParser::OperandType, 4> lbMapOperands;
    if(parser.parseAffineMapOfSSAIds(lbMapOperands, lbMapAttr,
                                      MapNode::getLBAttrName(),
                                      lbAttrs,
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

    result.addAttribute(MapNode::getLBAttrName(),
                        builder.getI64ArrayAttr(lb));

    if(parser.parseKeyword("to")) return failure();

    AffineMapAttr ubMapAttr;
    NamedAttrList ubAttrs;
    SmallVector<OpAsmParser::OperandType, 4> ubMapOperands;
    if(parser.parseAffineMapOfSSAIds(ubMapOperands, ubMapAttr,
                                      MapNode::getUBAttrName(),
                                      ubAttrs,
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

    
    result.addAttribute(MapNode::getUBAttrName(),
                        builder.getI64ArrayAttr(ub));

    if(parser.parseKeyword("step")) return failure();

    AffineMapAttr stepsMapAttr;
    NamedAttrList stepsAttrs;
    SmallVector<OpAsmParser::OperandType, 4> stepsMapOperands;
    if(parser.parseAffineMapOfSSAIds(stepsMapOperands, stepsMapAttr,
                                      MapNode::getStepsAttrName(),
                                      stepsAttrs,
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

    result.addAttribute(MapNode::getStepsAttrName(),
                        builder.getI64ArrayAttr(steps));

    // Now parse the body.
    Region *body = result.addRegion();
    SmallVector<Type, 4> types(ivs.size(), indexType);
    if (parser.parseRegion(*body, ivs, types) ||
            parser.parseOptionalAttrDict(result.attributes))
        return failure();

    return success();
}

void sdir::MapNode::printMapOp(OpAsmPrinter &p) {
    p << getOperationName() << " (" << getBody()->getArguments() << ") = (";
    
    SmallVector<int64_t, 8> lbresult;
    for (Attribute attr : lowerBoundsGroups()) {
        lbresult.push_back(attr.cast<IntegerAttr>().getInt());
    }
    llvm::interleaveComma(lbresult, p);

    p << ") to (";

    SmallVector<int64_t, 8> ubresult;
    for (Attribute attr : upperBoundsGroups()) {
        ubresult.push_back(attr.cast<IntegerAttr>().getInt());
    }
    llvm::interleaveComma(ubresult, p);

    p << ") step (";

    SmallVector<int64_t, 8> stepresult;
    for (Attribute attr : steps()) {
        stepresult.push_back(attr.cast<IntegerAttr>().getInt());
    }
    llvm::interleaveComma(stepresult, p);

    p << ")";

    p.printRegion(region(), /*printEntryBlockArgs=*/false, 
        /*printBlockTerminators=*/false);
}

bool sdir::MapNode::isDefinedOutsideOfLoop(Value value){
    return !region().isAncestor(value.getParentRegion());
}

Region &sdir::MapNode::getLoopBody(){
    return region();
}

LogicalResult sdir::MapNode::moveOutOfLoop(ArrayRef<Operation *> ops){
    return failure();
}
