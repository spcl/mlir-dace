#include "SDFG/Conversion/SDFGToGeneric/OpCreators.h"

using namespace mlir;
using namespace sdfg;

func::FuncOp conversion::createFunc(PatternRewriter &rewriter, Location loc,
                                    StringRef name, TypeRange inputTypes,
                                    TypeRange resultTypes,
                                    StringRef visibility) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, func::FuncOp::getOperationName());

  FunctionType func_type = builder.getFunctionType(inputTypes, resultTypes);
  StringAttr visAttr = builder.getStringAttr(visibility);

  func::FuncOp::build(builder, state, name, func_type, visAttr, {}, {});
  return cast<func::FuncOp>(rewriter.create(state));
}

func::ReturnOp conversion::createReturn(PatternRewriter &rewriter, Location loc,
                                        ValueRange operands) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, func::ReturnOp::getOperationName());

  func::ReturnOp::build(builder, state, operands);
  return cast<func::ReturnOp>(rewriter.create(state));
}

cf::BranchOp conversion::createBranch(PatternRewriter &rewriter, Location loc,
                                      ValueRange operands, Block *dest) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, cf::BranchOp::getOperationName());

  cf::BranchOp::build(builder, state, operands, dest);
  return cast<cf::BranchOp>(rewriter.create(state));
}

cf::CondBranchOp conversion::createCondBranch(PatternRewriter &rewriter,
                                              Location loc, Value condition,
                                              Block *trueDest,
                                              Block *falseDest) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, cf::CondBranchOp::getOperationName());

  cf::CondBranchOp::build(builder, state, condition, trueDest, falseDest);
  return cast<cf::CondBranchOp>(rewriter.create(state));
}

memref::AllocOp conversion::createAlloc(PatternRewriter &rewriter, Location loc,
                                        MemRefType memreftype) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, memref::AllocOp::getOperationName());

  memref::AllocOp::build(builder, state, memreftype);
  return cast<memref::AllocOp>(rewriter.create(state));
}

memref::LoadOp conversion::createLoad(PatternRewriter &rewriter, Location loc,
                                      Value memref, ValueRange indices) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, memref::LoadOp::getOperationName());

  memref::LoadOp::build(builder, state, memref, indices);
  return cast<memref::LoadOp>(rewriter.create(state));
}

memref::StoreOp conversion::createStore(PatternRewriter &rewriter, Location loc,
                                        Value value, Value memref,
                                        ValueRange indices) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, memref::StoreOp::getOperationName());

  memref::StoreOp::build(builder, state, value, memref, indices);
  return cast<memref::StoreOp>(rewriter.create(state));
}

// Allocates a symbol as a memref<i64> if it's not already allocated
void conversion::allocSymbol(PatternRewriter &rewriter, Location loc,
                             StringRef symName,
                             llvm::StringMap<memref::AllocOp> &symbolMap) {
  if (symbolMap.find(symName) != symbolMap.end())
    return;

  OpBuilder::InsertPoint insertionPoint = rewriter.saveInsertionPoint();

  // Set insertion point to the beginning of the first block (top of func)
  rewriter.setInsertionPointToStart(&rewriter.getBlock()->getParent()->front());

  IntegerType intType = IntegerType::get(loc->getContext(), 64);
  MemRefType memrefType = MemRefType::get({}, intType);
  memref::AllocOp allocOp = createAlloc(rewriter, loc, memrefType);

  // Update symbol map
  symbolMap[symName] = allocOp;

  rewriter.restoreInsertionPoint(insertionPoint);
}

arith::ConstantIntOp conversion::createConstantInt(PatternRewriter &rewriter,
                                                   Location loc, int val,
                                                   int width) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, arith::ConstantIntOp::getOperationName());

  arith::ConstantIntOp::build(builder, state, val, width);
  return cast<arith::ConstantIntOp>(rewriter.create(state));
}

arith::AddIOp conversion::createAddI(PatternRewriter &rewriter, Location loc,
                                     Value a, Value b) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, arith::AddIOp::getOperationName());

  arith::AddIOp::build(builder, state, a, b);
  return cast<arith::AddIOp>(rewriter.create(state));
}

arith::SubIOp conversion::createSubI(PatternRewriter &rewriter, Location loc,
                                     Value a, Value b) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, arith::SubIOp::getOperationName());

  arith::SubIOp::build(builder, state, a, b);
  return cast<arith::SubIOp>(rewriter.create(state));
}

arith::MulIOp conversion::createMulI(PatternRewriter &rewriter, Location loc,
                                     Value a, Value b) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, arith::MulIOp::getOperationName());

  arith::MulIOp::build(builder, state, a, b);
  return cast<arith::MulIOp>(rewriter.create(state));
}

arith::DivSIOp conversion::createDivSI(PatternRewriter &rewriter, Location loc,
                                       Value a, Value b) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, arith::DivSIOp::getOperationName());

  arith::DivSIOp::build(builder, state, a, b);
  return cast<arith::DivSIOp>(rewriter.create(state));
}

arith::FloorDivSIOp conversion::createFloorDivSI(PatternRewriter &rewriter,
                                                 Location loc, Value a,
                                                 Value b) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, arith::FloorDivSIOp::getOperationName());

  arith::FloorDivSIOp::build(builder, state, a, b);
  return cast<arith::FloorDivSIOp>(rewriter.create(state));
}

arith::RemSIOp conversion::createRemSI(PatternRewriter &rewriter, Location loc,
                                       Value a, Value b) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, arith::RemSIOp::getOperationName());

  arith::RemSIOp::build(builder, state, a, b);
  return cast<arith::RemSIOp>(rewriter.create(state));
}

arith::OrIOp conversion::createOrI(PatternRewriter &rewriter, Location loc,
                                   Value a, Value b) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, arith::OrIOp::getOperationName());

  arith::OrIOp::build(builder, state, a, b);
  return cast<arith::OrIOp>(rewriter.create(state));
}

arith::AndIOp conversion::createAndI(PatternRewriter &rewriter, Location loc,
                                     Value a, Value b) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, arith::AndIOp::getOperationName());

  arith::AndIOp::build(builder, state, a, b);
  return cast<arith::AndIOp>(rewriter.create(state));
}

arith::XOrIOp conversion::createXOrI(PatternRewriter &rewriter, Location loc,
                                     Value a, Value b) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, arith::XOrIOp::getOperationName());

  arith::XOrIOp::build(builder, state, a, b);
  return cast<arith::XOrIOp>(rewriter.create(state));
}

arith::ShLIOp conversion::createShLI(PatternRewriter &rewriter, Location loc,
                                     Value a, Value b) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, arith::ShLIOp::getOperationName());

  arith::ShLIOp::build(builder, state, a, b);
  return cast<arith::ShLIOp>(rewriter.create(state));
}

arith::ShRSIOp conversion::createShRSI(PatternRewriter &rewriter, Location loc,
                                       Value a, Value b) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, arith::ShRSIOp::getOperationName());

  arith::ShRSIOp::build(builder, state, a, b);
  return cast<arith::ShRSIOp>(rewriter.create(state));
}

arith::CmpIOp conversion::createCmpI(PatternRewriter &rewriter, Location loc,
                                     arith::CmpIPredicate predicate, Value lhs,
                                     Value rhs) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, arith::CmpIOp::getOperationName());

  arith::CmpIOp::build(builder, state, predicate, lhs, rhs);
  return cast<arith::CmpIOp>(rewriter.create(state));
}

arith::ExtSIOp conversion::createExtSI(PatternRewriter &rewriter, Location loc,
                                       Type out, Value in) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, arith::ExtSIOp::getOperationName());

  arith::ExtSIOp::build(builder, state, out, in);
  return cast<arith::ExtSIOp>(rewriter.create(state));
}

arith::IndexCastOp conversion::createIndexCast(PatternRewriter &rewriter,
                                               Location loc, Type out,
                                               Value in) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, arith::IndexCastOp::getOperationName());

  arith::IndexCastOp::build(builder, state, out, in);
  return cast<arith::IndexCastOp>(rewriter.create(state));
}