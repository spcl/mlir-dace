#include "SDIR/SDIR_Dialect.h"

using namespace mlir;
using namespace mlir::sdir;

//===----------------------------------------------------------------------===//
// SDFGNode
//===----------------------------------------------------------------------===//

static ParseResult parseSDFGNode(OpAsmParser &parser, OperationState &result) {
    if(parser.parseOptionalAttrDict(result.attributes))
        return failure();

    StringAttr sym_nameAttr;
    if(parser.parseSymbolName(sym_nameAttr, "sym_name", result.attributes))
        return failure();

    Region *body = result.addRegion();
    if(parser.parseRegion(*body))
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

LogicalResult verify(SDFGNode op){
    return success();
}

LogicalResult SDFGNode::verifySymbolUses(SymbolTableCollection &symbolTable) {
    // Check that the entry attribute is specified.
    FlatSymbolRefAttr entryAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("entry");
    if(!entryAttr)
        return emitOpError("requires a 'src' symbol reference attribute");
    StateNode entry = symbolTable.lookupNearestSymbolFrom<StateNode>(*this, entryAttr);
    if(!entry)
        return emitOpError() << "'" << entryAttr.getValue()
                            << "' does not reference a valid state";

    return success();
}

//===----------------------------------------------------------------------===//
// StateNode
//===----------------------------------------------------------------------===//

static ParseResult parseStateNode(OpAsmParser &parser, OperationState &result) {
    if(parser.parseOptionalAttrDict(result.attributes))
        return failure();

    StringAttr sym_nameAttr;
    if(parser.parseSymbolName(sym_nameAttr, "sym_name", result.attributes))
        return failure();

    Region *body = result.addRegion();
    if(parser.parseRegion(*body))
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

LogicalResult verify(StateNode op){
    return success();
}

//===----------------------------------------------------------------------===//
// TaskletNode
//===----------------------------------------------------------------------===//

static ParseResult parseTaskletNode(OpAsmParser &parser, OperationState &result) {
    auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
                            function_like_impl::VariadicFlag, std::string &) {
        return builder.getFunctionType(argTypes, results);
    };

    return function_like_impl::parseFunctionLikeOp(parser, result, /*allowVariadic=*/false, buildFuncType);
}

static void print(OpAsmPrinter &p, TaskletNode op) {
    p.printNewline();
    FunctionType fnType = op.getType();
    function_like_impl::printFunctionLikeOp(p, op, fnType.getInputs(), /*isVariadic=*/false, fnType.getResults());
}

LogicalResult verify(TaskletNode op){
    if (op.isExternal())
        return success();

    // Verify that the argument list of the function and the arg list of the entry
    // block line up.  The trait already verified that the number of arguments is
    // the same between the signature and the block.
    ArrayRef<Type> fnInputTypes = op.getType().getInputs();
    Block &entryBlock = op.front();

    for (int i = 0; i < entryBlock.getNumArguments(); i++)
        if (fnInputTypes[i] != entryBlock.getArgument(i).getType())
            return op.emitOpError("type of entry block argument #")
                    << i << '(' << entryBlock.getArgument(i).getType()
                    << ") must match the type of the corresponding argument in "
                    << "function signature(" << fnInputTypes[i] << ')';

    return success();
}

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

    if(argAttrs.empty())
        return;
    assert(type.getNumInputs() == argAttrs.size());
    function_like_impl::addArgAndResultAttrs(builder, state, argAttrs,
                                            /*resultAttrs=*/llvm::None);
}

void TaskletNode::cloneInto(TaskletNode dest, BlockAndValueMapping &mapper) {
    llvm::MapVector<Identifier, Attribute> newAttrs;
    for(const NamedAttribute &attr : dest->getAttrs())
        newAttrs.insert(attr);
    for(const NamedAttribute &attr : (*this)->getAttrs())
        newAttrs.insert(attr);
    dest->setAttrs(DictionaryAttr::get(getContext(), newAttrs.takeVector()));

    getBody().cloneInto(&dest.getBody(), mapper);
}

TaskletNode TaskletNode::clone(BlockAndValueMapping &mapper) {
    TaskletNode newFunc = cast<TaskletNode>(getOperation()->cloneWithoutRegions());
    
    if(!isExternal()) {
        FunctionType oldType = getType();

        unsigned oldNumArgs = oldType.getNumInputs();
        SmallVector<Type, 4> newInputs;
        newInputs.reserve(oldNumArgs);

        for(int i = 0; i < oldNumArgs; i++)
            if(!mapper.contains(getArgument(i)))
                newInputs.push_back(oldType.getInput(i));

        if(newInputs.size() != oldNumArgs) {
            newFunc.setType(FunctionType::get(oldType.getContext(), newInputs,
                                            oldType.getResults()));

            if(ArrayAttr argAttrs = getAllArgAttrs()) {
                SmallVector<Attribute> newArgAttrs;
                newArgAttrs.reserve(newInputs.size());
                for(int i = 0; i < oldNumArgs; i++)
                    if(!mapper.contains(getArgument(i)))
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
    Builder &builder = parser.getBuilder();
    IndexType indexType = builder.getIndexType();

    if(parser.parseOptionalAttrDict(result.attributes))
        return failure();

    SmallVector<OpAsmParser::OperandType, 4> ivs;
    if(parser.parseRegionArgumentList(ivs, OpAsmParser::Delimiter::Paren))
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
    AffineMap lbMap = lbMapAttr.getValue();
    for(const AffineExpr &result : lbMap.getResults()) {
      AffineConstantExpr constExpr = result.dyn_cast<AffineConstantExpr>();
      if(!constExpr)
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
    AffineMap ubMap = ubMapAttr.getValue();
    for(const AffineExpr &result : ubMap.getResults()) {
      AffineConstantExpr constExpr = result.dyn_cast<AffineConstantExpr>();
      if(!constExpr)
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
    AffineMap stepsMap = stepsMapAttr.getValue();
    for(const AffineExpr &result : stepsMap.getResults()) {
      AffineConstantExpr constExpr = result.dyn_cast<AffineConstantExpr>();
      if (!constExpr)
        return parser.emitError(parser.getNameLoc(),
                                "steps must be constant integers");
      steps.push_back(constExpr.getValue());
    }

    result.addAttribute("steps", builder.getI64ArrayAttr(steps));

    // Now parse the body.
    Region *body = result.addRegion();
    SmallVector<Type, 4> types(ivs.size(), indexType);
    if(parser.parseRegion(*body, ivs, types))
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
    for(Attribute attr : op.lowerBounds()) {
        lbresult.push_back(attr.cast<IntegerAttr>().getInt());
    }
    llvm::interleaveComma(lbresult, p);

    p << ") to (";

    SmallVector<int64_t, 8> ubresult;
    for(Attribute attr : op.upperBounds()) {
        ubresult.push_back(attr.cast<IntegerAttr>().getInt());
    }
    llvm::interleaveComma(ubresult, p);

    p << ") step (";

    SmallVector<int64_t, 8> stepresult;
    for(Attribute attr : op.steps()) {
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
    if(parser.parseRegion(*body, ivs, types))
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
    // Check that the condition attribute is specified.
    FlatSymbolRefAttr condAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("condition");
    if(!condAttr)
        return success();

    FuncOp cond = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, condAttr);
    if(!cond)
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
// EdgeOp
//===----------------------------------------------------------------------===//

static ParseResult parseEdgeOp(OpAsmParser &parser, OperationState &result) {
    FlatSymbolRefAttr srcAttr;
    FlatSymbolRefAttr destAttr;

    if(parser.parseOptionalAttrDict(result.attributes))
        return failure();

    if(parser.parseAttribute(srcAttr, "src", result.attributes))
        return failure();

    if(parser.parseArrow())
        return failure();

    if(parser.parseAttribute(destAttr, "dest", result.attributes))
        return failure();

    return success();
}

static void print(OpAsmPrinter &p, EdgeOp op) {
    p << op.getOperationName();
    p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"src", "dest"});
    p << ' ';
    p.printAttributeWithoutType(op.srcAttr());
    p << " -> ";
    p.printAttributeWithoutType(op.destAttr());
}

LogicalResult verify(EdgeOp op){
    return success();
}

LogicalResult EdgeOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
    // Check that the src/dest attributes are specified.
    FlatSymbolRefAttr srcAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("src");
    if(!srcAttr)
        return emitOpError("requires a 'src' symbol reference attribute");

    StateNode src = symbolTable.lookupNearestSymbolFrom<StateNode>(*this, srcAttr);
    if(!src)
        return emitOpError() << "'" << srcAttr.getValue()
                            << "' does not reference a valid state";
    
    FlatSymbolRefAttr destAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("dest");
    if(!destAttr)
        return emitOpError("requires a 'dest' symbol reference attribute");

    StateNode dest = symbolTable.lookupNearestSymbolFrom<StateNode>(*this, destAttr);
    if(!dest)
        return emitOpError() << "'" << destAttr.getValue()
                            << "' does not reference a valid state";

    return success();
}

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

static ParseResult parseAllocOp(OpAsmParser &parser, OperationState &result) {
    if(parser.parseOptionalAttrDict(result.attributes))
        return failure();

    SmallVector<OpAsmParser::OperandType, 4> paramsOperands;
    if(parser.parseOperandList(paramsOperands, OpAsmParser::Delimiter::Paren))
        return failure();

    // IDEA: Try multiple integer types before failing
    if(parser.resolveOperands(paramsOperands, parser.getBuilder().getI32Type(), result.operands))
        return failure();

    Type resultType;
    if(parser.parseColonType(resultType))
        return failure();
    result.addTypes(resultType);

    return success();
}

static void print(OpAsmPrinter &p, AllocOp op) {
    p << op.getOperationName();
    p.printOptionalAttrDict(op->getAttrs());
    p << "(";
    p.printOperands(op.params());
    p << ") : ";
    p << op.getOperation()->getResultTypes();
}

LogicalResult verify(AllocOp op){
    ArrayType res = op.res().getType().cast<ArrayType>();

    if(res.getUndefRank() != op.params().size())
        return op.emitOpError("failed to verify that parameter size matches undefined dimensions size");

    if(res.hasZeros())
        return op.emitOpError("failed to verify that return type doesn't contain dimensions of size zero");

    return success();
}

//===----------------------------------------------------------------------===//
// AllocTransientOp
//===----------------------------------------------------------------------===//

static ParseResult parseAllocTransientOp(OpAsmParser &parser, OperationState &result) {
    if(parser.parseOptionalAttrDict(result.attributes))
        return failure();

    SmallVector<OpAsmParser::OperandType, 4> paramsOperands;
    if(parser.parseOperandList(paramsOperands, OpAsmParser::Delimiter::Paren))
        return failure();

    // IDEA: Try multiple integer types before failing
    if(parser.resolveOperands(paramsOperands, parser.getBuilder().getI32Type(), result.operands))
        return failure();
    
    Type resultType;
    if(parser.parseColonType(resultType))
        return failure();
    result.addTypes(resultType);

    return success();
}

static void print(OpAsmPrinter &p, AllocTransientOp op) {
    p << op.getOperationName();
    p.printOptionalAttrDict(op->getAttrs());
    p << "(";
    p.printOperands(op.params());
    p << ") : ";
    p << op.getOperation()->getResultTypes();
}

LogicalResult verify(AllocTransientOp op){
    ArrayType res = op.res().getType().cast<ArrayType>();

    if(res.getUndefRank() != op.params().size())
        return op.emitOpError("failed to verify that parameter size matches undefined dimensions size");

    if(res.hasZeros())
        return op.emitOpError("failed to verify that return type doesn't contain dimensions of size zero");

    return success();
}

//===----------------------------------------------------------------------===//
// GetAccessOp
//===----------------------------------------------------------------------===//

static ParseResult parseGetAccessOp(OpAsmParser &parser, OperationState &result) {
    if(parser.parseOptionalAttrDict(result.attributes))
        return failure();

    OpAsmParser::OperandType arrStrOperand;
    if(parser.parseOperand(arrStrOperand))
        return failure();

    Type srcType;
    if(parser.parseColonType(srcType))
        return failure();

    if(parser.parseArrow())
        return failure();

    Type destType;
    if(parser.parseType(destType))
        return failure();
    result.addTypes(destType);

    if(parser.resolveOperand(arrStrOperand, srcType, result.operands))
        return failure();

    return success();
}

static void print(OpAsmPrinter &p, GetAccessOp op) {
    p << op.getOperationName();
    p.printOptionalAttrDict(op->getAttrs());
    p << ' ' << op.arr();
    p << " : ";
    p << ArrayRef<Type>(op.arr().getType());
    p << " -> ";
    p << ArrayRef<Type>(op.res().getType());
}

LogicalResult verify(GetAccessOp op){
    Type arr = op.arr().getType();
    Type res = op.res().getType();

    if(arr.isa<ArrayType>() && res.isa<MemletType>())
        if(arr.cast<ArrayType>().getElementType() != res.cast<MemletType>().getElementType())
            return op.emitOpError("failed to verify that result element type matches element type of 'array'");

    if(arr.isa<StreamArrayType>() && res.isa<StreamType>())
        if(arr.cast<StreamArrayType>().getElementType() != res.cast<StreamType>().getElementType())
            return op.emitOpError("failed to verify that result element type matches element type of 'stream_array'");

    if(arr.isa<ArrayType>() && res.isa<StreamType>())
        return op.emitOpError("failed to verify that result type matches derived type of 'array'");

    if(arr.isa<StreamArrayType>() && res.isa<MemletType>())
        return op.emitOpError("failed to verify that result type matches derived type of 'stream_array'");

    return success();
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

static ParseResult parseLoadOp(OpAsmParser &parser, OperationState &result) {
    if(parser.parseOptionalAttrDict(result.attributes))
        return failure();

    OpAsmParser::OperandType memletOperand;
    if(parser.parseOperand(memletOperand))
        return failure();

    SmallVector<OpAsmParser::OperandType, 4> indicesOperands;
    if(parser.parseOperandList(indicesOperands, OpAsmParser::Delimiter::Square))
        return failure();

    Type srcType;
    if(parser.parseColonType(srcType))
        return failure();

    if(parser.parseArrow())
        return failure();

    Type destType;
    if(parser.parseType(destType))
        return failure();
    result.addTypes(destType);

    if(parser.resolveOperand(memletOperand, srcType, result.operands))
        return failure();

    if(parser.resolveOperands(indicesOperands, parser.getBuilder().getIndexType(), result.operands))
        return failure();

    return success();
}

static void print(OpAsmPrinter &p, LoadOp op) {
    p << op.getOperationName();
    p.printOptionalAttrDict(op->getAttrs());
    p << ' ' << op.arr();
    p << "[" << op.indices() << "]";
    p << " : ";
    p << ArrayRef<Type>(op.arr().getType());
    p << " -> ";
    p << ArrayRef<Type>(op.res().getType());
}

LogicalResult verify(LoadOp op){
    size_t idx_size = op.indices().size();
    size_t mem_size = op.arr().getType().cast<MemletType>().getRank();
    if(idx_size != mem_size)
      return op.emitOpError("incorrect number of indices for load");

    return success();
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

static ParseResult parseStoreOp(OpAsmParser &parser, OperationState &result) {
    if(parser.parseOptionalAttrDict(result.attributes))
        return failure();
    
    OpAsmParser::OperandType valOperand;
    if(parser.parseOperand(valOperand))
        return failure();
    if(parser.parseComma())
        return failure();

    OpAsmParser::OperandType memletOperand;
    if(parser.parseOperand(memletOperand))
        return failure();

    SmallVector<OpAsmParser::OperandType, 4> indicesOperands;
    if(parser.parseOperandList(indicesOperands, OpAsmParser::Delimiter::Square))
        return failure();

    Type valType;
    if(parser.parseColonType(valType))
        return failure();

    if(parser.parseArrow())
        return failure();

    Type memletType;
    if(parser.parseType(memletType))
        return failure();

    if(parser.resolveOperand(valOperand, valType, result.operands))
        return failure();

    if(parser.resolveOperand(memletOperand, memletType, result.operands))
        return failure();

    if(parser.resolveOperands(indicesOperands, parser.getBuilder().getIndexType(), result.operands))
        return failure();

    return success();
}

static void print(OpAsmPrinter &p, StoreOp op) {
    p << op.getOperationName();
    p.printOptionalAttrDict(op->getAttrs());
    p << ' ' << op.val() << "," << ' ' << op.arr();
    p << "[" << op.indices() << "]";
    p << " : ";
    p << ArrayRef<Type>(op.val().getType());
    p << " -> ";
    p << ArrayRef<Type>(op.arr().getType());
}

LogicalResult verify(StoreOp op){
    size_t idx_size = op.indices().size();
    size_t mem_size = op.arr().getType().cast<MemletType>().getRank();
    if(idx_size != mem_size)
      return op.emitOpError("incorrect number of indices for store");

    return success();
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

static ParseResult parseCopyOp(OpAsmParser &parser, OperationState &result) {
    if(parser.parseOptionalAttrDict(result.attributes))
        return failure();

    OpAsmParser::OperandType srcOperand;
    if(parser.parseOperand(srcOperand))
        return failure();

    if(parser.parseArrow())
        return failure();

    OpAsmParser::OperandType destOperand;
    if(parser.parseOperand(destOperand))
        return failure();

    Type opType;
    if(parser.parseColonType(opType))
        return failure();

    if(parser.resolveOperand(srcOperand, opType, result.operands))
        return failure();

    if(parser.resolveOperand(destOperand, opType, result.operands))
        return failure();
    
    return success();
}

static void print(OpAsmPrinter &p, CopyOp op) {
    p << op.getOperationName();
    p.printOptionalAttrDict(op->getAttrs());
    p << ' ' << op.src() << " -> " << op.dest();
    p << " : ";
    p << ArrayRef<Type>(op.src().getType());
}

LogicalResult verify(CopyOp op){
    return success();
}

//===----------------------------------------------------------------------===//
// MemletCastOp
//===----------------------------------------------------------------------===//

static ParseResult parseMemletCastOp(OpAsmParser &parser, OperationState &result) {
    if(parser.parseOptionalAttrDict(result.attributes))
        return failure();

    OpAsmParser::OperandType memletOperand;
    if(parser.parseOperand(memletOperand))
        return failure();

    Type srcType;
    if(parser.parseColonType(srcType))
        return failure();

    if(parser.parseArrow())
        return failure();

    Type destType;
    if(parser.parseType(destType))
        return failure();
    result.addTypes(destType);
  
    if(parser.resolveOperands(memletOperand, srcType, result.operands))
        return failure();
  
    return success();
}

static void print(OpAsmPrinter &p, MemletCastOp op) {
    p << op.getOperationName();
    p.printOptionalAttrDict(op->getAttrs());
    p << ' ' << op.src();
    p << " : ";
    p << ArrayRef<Type>(op.src().getType());
    p << " -> ";
    p << op.getOperation()->getResultTypes();
}

LogicalResult verify(MemletCastOp op){
    size_t src_size = op.src().getType().cast<MemletType>().getRank();
    size_t res_size = op.res().getType().cast<MemletType>().getRank();
    if(src_size != res_size)
        return op.emitOpError("incorrect rank for memlet_cast");

    return success();
}

//===----------------------------------------------------------------------===//
// ViewCastOp
//===----------------------------------------------------------------------===//

static ParseResult parseViewCastOp(OpAsmParser &parser, OperationState &result) {
    if(parser.parseOptionalAttrDict(result.attributes))
        return failure();

    OpAsmParser::OperandType memletOperand;
    if(parser.parseOperand(memletOperand))
        return failure();

    Type srcType;
    if(parser.parseColonType(srcType))
        return failure();

    if(parser.parseArrow())
        return failure();

    Type destType;
    if(parser.parseType(destType))
        return failure();
    result.addTypes(destType);
  
    if(parser.resolveOperands(memletOperand, srcType, result.operands))
        return failure();
  
    return success();
}

static void print(OpAsmPrinter &p, ViewCastOp op) {
    p << op.getOperationName();
    p.printOptionalAttrDict(op->getAttrs());
    p << ' ' << op.src();
    p << " : ";
    p << ArrayRef<Type>(op.src().getType());
    p << " -> ";
    p << op.getOperation()->getResultTypes();
}

LogicalResult verify(ViewCastOp op){
    size_t src_size = op.src().getType().cast<MemletType>().getRank();
    size_t res_size = op.res().getType().cast<MemletType>().getRank();
    if (src_size != res_size)
        return op.emitOpError("incorrect rank for view_cast");
    
    return success();
}

//===----------------------------------------------------------------------===//
// AllocStreamOp
//===----------------------------------------------------------------------===//

static ParseResult parseAllocStreamOp(OpAsmParser &parser, OperationState &result) {
    if(parser.parseOptionalAttrDict(result.attributes))
        return failure();

    SmallVector<OpAsmParser::OperandType, 4> paramsOperands;
    if(parser.parseOperandList(paramsOperands, OpAsmParser::Delimiter::Paren))
        return failure();
    
    // IDEA: Try multiple integer types before failing
    if(parser.resolveOperands(paramsOperands, parser.getBuilder().getI32Type(), result.operands))
        return failure();

    Type resultType;
    if(parser.parseColonType(resultType))
        return failure();
    result.addTypes(resultType);
    
    return success();
}

static void print(OpAsmPrinter &p, AllocStreamOp op) {
    p << op.getOperationName();
    p.printOptionalAttrDict(op->getAttrs());
    p << "() : ";
    p << op.getOperation()->getResultTypes();
}

LogicalResult verify(AllocStreamOp op){
    return success();
}

//===----------------------------------------------------------------------===//
// AllocTransientStreamOp
//===----------------------------------------------------------------------===//

static ParseResult parseAllocTransientStreamOp(OpAsmParser &parser, OperationState &result) {
    if(parser.parseOptionalAttrDict(result.attributes))
        return failure();

    SmallVector<OpAsmParser::OperandType, 4> paramsOperands;
    if(parser.parseOperandList(paramsOperands, OpAsmParser::Delimiter::Paren))
        return failure();
    
    // IDEA: Try multiple integer types before failing
    if(parser.resolveOperands(paramsOperands, parser.getBuilder().getI32Type(), result.operands))
        return failure();

    Type resultType;
    if(parser.parseColonType(resultType))
        return failure();
    result.addTypes(resultType);
    
    return success();
}

static void print(OpAsmPrinter &p, AllocTransientStreamOp op) {
    p << op.getOperationName();
    p.printOptionalAttrDict(op->getAttrs());
    p << "() : ";
    p << op.getOperation()->getResultTypes();
}

LogicalResult verify(AllocTransientStreamOp op){
    return success();
}

//===----------------------------------------------------------------------===//
// StreamPopOp
//===----------------------------------------------------------------------===//

static ParseResult parseStreamPopOp(OpAsmParser &parser, OperationState &result) {
    if(parser.parseOptionalAttrDict(result.attributes))
        return failure();

    OpAsmParser::OperandType streamOperand;
    if(parser.parseOperand(streamOperand))
        return failure();

    Type streamType;
    if(parser.parseColonType(streamType))
        return failure();

    if(parser.parseArrow())
        return failure();

    Type resultType;
    if(parser.parseType(resultType))
        return failure();
    result.addTypes(resultType);

    if(parser.resolveOperand(streamOperand, streamType, result.operands))
        return failure();

    return success();
}

static void print(OpAsmPrinter &p, StreamPopOp op) {
    p << op.getOperationName();
    p.printOptionalAttrDict(op->getAttrs());
    p << ' ' << op.str();
    p << " : ";
    p << ArrayRef<Type>(op.str().getType());
    p << " -> ";
    p << ArrayRef<Type>(op.res().getType());
}

LogicalResult verify(StreamPopOp op){
    return success();
}

//===----------------------------------------------------------------------===//
// StreamPushOp
//===----------------------------------------------------------------------===//

static ParseResult parseStreamPushOp(OpAsmParser &parser, OperationState &result) {
    if(parser.parseOptionalAttrDict(result.attributes))
        return failure();

    OpAsmParser::OperandType valOperand;
    if(parser.parseOperand(valOperand))
        return failure();
    if(parser.parseComma())
        return failure();

    OpAsmParser::OperandType streamOperand;
    if(parser.parseOperand(streamOperand))
        return failure();

    Type valType;
    if(parser.parseColonType(valType))
        return failure();
    
    if(parser.parseArrow())
        return failure();
    
    Type streamType;
    if(parser.parseType(streamType))
        return failure();

    if(parser.resolveOperand(valOperand, valType, result.operands))
        return failure();

    if(parser.resolveOperand(streamOperand, streamType, result.operands))
        return failure();

    return success();
}

static void print(OpAsmPrinter &p, StreamPushOp op) {
    p << op.getOperationName();
    p.printOptionalAttrDict(op->getAttrs());
    p << ' ' << op.val() << ", " << op.str();
    p << " : ";
    p << ArrayRef<Type>(op.val().getType());
    p << " -> ";
    p << ArrayRef<Type>(op.str().getType());
}

LogicalResult verify(StreamPushOp op){
    if(op.val().getType() != op.str().getType().cast<StreamType>().getElementType())
        op.emitOpError("failed to verify that value type matches element type of 'stream'");

    return success();
}

//===----------------------------------------------------------------------===//
// StreamLengthOp
//===----------------------------------------------------------------------===//

static ParseResult parseStreamLengthOp(OpAsmParser &parser, OperationState &result) {
    if(parser.parseOptionalAttrDict(result.attributes))
        return failure();

    OpAsmParser::OperandType streamOperand;
    if(parser.parseOperand(streamOperand))
        return failure();

    Type streamType;
    if(parser.parseColonType(streamType))
        return failure();

    if(parser.parseArrow())
        return failure();

    Type resultType;
    if(parser.parseType(resultType))
        return failure();
    result.addTypes(resultType);

    if(parser.resolveOperands(streamOperand, streamType, result.operands))
        return failure();

    return success();
}

static void print(OpAsmPrinter &p, StreamLengthOp op) {
    p << op.getOperationName();
    p.printOptionalAttrDict(op->getAttrs());
    p << ' ' << op.str();
    p << " : ";
    p << ArrayRef<Type>(op.str().getType());
    p << " -> ";
    p << op.getOperation()->getResultTypes();
}

LogicalResult verify(StreamLengthOp op){
    return success();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

static ParseResult parseReturnOp(OpAsmParser &parser, OperationState &result) {
    if(parser.parseOptionalAttrDict(result.attributes))
        return failure();

    SmallVector<OpAsmParser::OperandType, 4> returnOperands;
    if(parser.parseOperandList(returnOperands))
        return failure();

    if(!returnOperands.empty()) {
        if(parser.parseColon())
            return failure();

        SmallVector<Type, 1> returnTypes;
        if(parser.parseTypeList(returnTypes))
            return failure();
        
        if(parser.resolveOperands(returnOperands, returnTypes, parser.getCurrentLocation() , result.operands))
            return failure();
    }
   
    return success();
}

static void print(OpAsmPrinter &p, ReturnOp op) {
    p << op.getOperationName();
    p.printOptionalAttrDict(op->getAttrs());
    if(!op.input().empty()) {
        p << ' ' << op.input();
        p << " : ";
        p << op.input().getTypes();
    }
}

LogicalResult verify(ReturnOp op){
    return success();
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

static ParseResult parseCallOp(OpAsmParser &parser, OperationState &result) {
    if(parser.parseOptionalAttrDict(result.attributes))
        return failure();

    Attribute calleeAttr;
    if(parser.parseAttribute(calleeAttr, parser.getBuilder().getNoneType(), "callee", result.attributes))
        return failure();

    SmallVector<OpAsmParser::OperandType, 4> operands;
    if(parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren))
        return failure();
  
    FunctionType func;
    if(parser.parseColonType(func))
        return failure();

    ArrayRef<Type> operandsTypes = func.getInputs();
    ArrayRef<Type> resultTypes = func.getResults();
    result.addTypes(resultTypes);

    if(parser.resolveOperands(operands, operandsTypes, parser.getCurrentLocation(), result.operands))
        return failure();

    return success();
}

static void print(OpAsmPrinter &p, CallOp op) {
    p << op.getOperationName();
    p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"callee"});
    p << ' ';
    p.printAttributeWithoutType(op.calleeAttr());
    p << "(" << op.operands() << ")";
    p << " : ";
    p.printFunctionalType(op.operands().getTypes(), op.getOperation()->getResultTypes());
}

LogicalResult verify(CallOp op){
    return success();
}

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
    // Check that the callee attribute was specified.
    FlatSymbolRefAttr fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
    if(!fnAttr)
        return emitOpError("requires a 'callee' symbol reference attribute");

    TaskletNode fn = symbolTable.lookupNearestSymbolFrom<TaskletNode>(*this, fnAttr);
    if(!fn)
        return emitOpError() << "'" << fnAttr.getValue()
                            << "' does not reference a valid tasklet";

    // Verify that the operand and result types match the callee.
    FunctionType fnType = fn.getType();
    if(fnType.getNumInputs() != getNumOperands())
        return emitOpError("incorrect number of operands for callee");

    for(int i = 0; i < fnType.getNumInputs(); i++)
        if(getOperand(i).getType() != fnType.getInput(i))
            return emitOpError("operand type mismatch: expected operand type ")
                << fnType.getInput(i) << ", but provided "
                << getOperand(i).getType() << " for operand number " << i;

    if(fnType.getNumResults() != getNumResults())
        return emitOpError("incorrect number of results for callee");

    for(int i = 0; i < fnType.getNumResults(); i++)
        if(getResult(i).getType() != fnType.getResult(i)) {
            InFlightDiagnostic diag = emitOpError("result type mismatch at index ") << i;
            diag.attachNote() << "      op result types: " << getResultTypes();
            diag.attachNote() << "function result types: " << fnType.getResults();
            return diag;
        }

    return success();
}

//===----------------------------------------------------------------------===//
// LibCallOp
//===----------------------------------------------------------------------===//

static ParseResult parseLibCallOp(OpAsmParser &parser, OperationState &result) {
    if(parser.parseOptionalAttrDict(result.attributes))
        return failure();

    Attribute calleeAttr;
    if(parser.parseAttribute(calleeAttr, parser.getBuilder().getNoneType(), "callee", result.attributes))
        return failure();

    SmallVector<OpAsmParser::OperandType, 4> operandsOperands;
    if(parser.parseOperandList(operandsOperands, OpAsmParser::Delimiter::Paren))
        return failure();

    FunctionType func;
    if(parser.parseColonType(func))
        return failure();

    ArrayRef<Type> operandsTypes = func.getInputs();
    ArrayRef<Type> allResultTypes = func.getResults();
    result.addTypes(allResultTypes);

    if(parser.resolveOperands(operandsOperands, operandsTypes, parser.getCurrentLocation(), result.operands))
        return failure();

    return success();
}

static void print(OpAsmPrinter &p, LibCallOp op) {
    p << op.getOperationName();
    p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"callee"});
    p << ' ';
    p.printAttributeWithoutType(op.calleeAttr());
    p << "(" << op.operands() << ")";
    p << " : ";
    p.printFunctionalType(op.operands().getTypes(), op.getOperation()->getResultTypes());
}

LogicalResult verify(LibCallOp op){
    return success();
}

//===----------------------------------------------------------------------===//
// AllocSymbolOp
//===----------------------------------------------------------------------===//

static ParseResult parseAllocSymbolOp(OpAsmParser &parser, OperationState &result) {
    StringAttr symAttr;
    if(parser.parseOptionalAttrDict(result.attributes))
        return failure();

    if(parser.parseLParen())
        return failure();
    if(parser.parseAttribute(symAttr, parser.getBuilder().getNoneType(), "sym", result.attributes))
        return failure();
    if(parser.parseRParen())
        return failure();

    return success();
}

static void print(OpAsmPrinter &p, AllocSymbolOp op) {
    p << op.getOperationName();
    p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"sym"});
    p << "(";
    p.printAttributeWithoutType(op.symAttr());
    p << ")";
}

LogicalResult verify(AllocSymbolOp op){
    if(op.sym().empty())
      return op.emitOpError("failed to verify that input string is not empty");
    
    if(!isalpha(op.sym().front()))
      return op.emitOpError("failed to verify that input string starts with an alphabetical character");

    for(auto c : op.sym()) 
      if(!isalnum(c))
        return op.emitOpError("failed to verify that input string only contains alphanumeric characters");

    return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "SDIR/SDIR_Ops.cpp.inc"
