#include "SDIR/SDIR_Dialect.h"

#define GET_OP_CLASSES
#include "SDIR/SDIR_Ops.cpp.inc"

using namespace mlir;

//===----------------------------------------------------------------------===//
// TaskletOp
//===----------------------------------------------------------------------===//

sdir::TaskletOp sdir::TaskletOp::create(Location location, StringRef name, FunctionType type, 
                                        ArrayRef<NamedAttribute> attrs) {
    OpBuilder builder(location->getContext());
    OperationState state(location, getOperationName());
    build(builder, state, name, type, attrs);
    return cast<TaskletOp>(Operation::create(state));
}

sdir::TaskletOp sdir::TaskletOp::create(Location location, StringRef name, FunctionType type, 
                                        Operation::dialect_attr_range attrs) {
    SmallVector<NamedAttribute, 8> attrRef(attrs);
    return create(location, name, type, llvm::makeArrayRef(attrRef));
}

sdir::TaskletOp sdir::TaskletOp::create(Location location, StringRef name, FunctionType type, 
                                        ArrayRef<NamedAttribute> attrs, 
                                        ArrayRef<DictionaryAttr> argAttrs) {
    TaskletOp func = create(location, name, type, attrs);
    func.setAllArgAttrs(argAttrs);
    return func;
}

void sdir::TaskletOp::build(OpBuilder &builder, OperationState &state, StringRef name,
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

void sdir::TaskletOp::cloneInto(TaskletOp dest, BlockAndValueMapping &mapper) {
    llvm::MapVector<Identifier, Attribute> newAttrs;
    for (const auto &attr : dest->getAttrs())
        newAttrs.insert(attr);
    for (const auto &attr : (*this)->getAttrs())
        newAttrs.insert(attr);
    dest->setAttrs(DictionaryAttr::get(getContext(), newAttrs.takeVector()));

    getBody().cloneInto(&dest.getBody(), mapper);
}

sdir::TaskletOp sdir::TaskletOp::clone(BlockAndValueMapping &mapper) {
    TaskletOp newFunc = cast<TaskletOp>(getOperation()->cloneWithoutRegions());
    
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

sdir::TaskletOp sdir::TaskletOp::clone() {
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
    TaskletOp fn = symbolTable.lookupNearestSymbolFrom<TaskletOp>(*this, fnAttr);
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
// SDFGOp
//===----------------------------------------------------------------------===//

LogicalResult sdir::SDFGOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
    // Check that the entry attribute is specified.
    auto entryAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("entry");
    if (!entryAttr)
        return emitOpError("requires a 'src' symbol reference attribute");
    StateOp entry = symbolTable.lookupNearestSymbolFrom<StateOp>(*this, entryAttr);
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
    StateOp src = symbolTable.lookupNearestSymbolFrom<StateOp>(*this, srcAttr);
    if (!src)
        return emitOpError() << "'" << srcAttr.getValue()
                            << "' does not reference a valid state";
    
    auto destAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("dest");
    if (!destAttr)
        return emitOpError("requires a 'dest' symbol reference attribute");
    StateOp dest = symbolTable.lookupNearestSymbolFrom<StateOp>(*this, destAttr);
    if (!dest)
        return emitOpError() << "'" << destAttr.getValue()
                            << "' does not reference a valid state";

    return success();
}
