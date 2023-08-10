// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

/// This file contains convenience functions that build, create and insert
/// various operations.

#include "SDFG/Conversion/SDFGToGeneric/OpCreators.h"

using namespace mlir;
using namespace sdfg;

/// Builds, creates and inserts a func::FuncOp.
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

/// Builds, creates and inserts a func::CallOp.
func::CallOp conversion::createCall(PatternRewriter &rewriter, Location loc,
                                    TypeRange resultTypes, StringRef callee,
                                    ValueRange operands) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, func::CallOp::getOperationName());

  func::CallOp::build(builder, state, resultTypes, callee, operands);
  return cast<func::CallOp>(rewriter.create(state));
}

/// Builds, creates and inserts a func::ReturnOp.
func::ReturnOp conversion::createReturn(PatternRewriter &rewriter, Location loc,
                                        ValueRange operands) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, func::ReturnOp::getOperationName());

  func::ReturnOp::build(builder, state, operands);
  return cast<func::ReturnOp>(rewriter.create(state));
}

/// Builds, creates and inserts a cf::BranchOp.
cf::BranchOp conversion::createBranch(PatternRewriter &rewriter, Location loc,
                                      ValueRange operands, Block *dest) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, cf::BranchOp::getOperationName());

  cf::BranchOp::build(builder, state, operands, dest);
  return cast<cf::BranchOp>(rewriter.create(state));
}

/// Builds, creates and inserts a cf::CondBranchOp.
cf::CondBranchOp conversion::createCondBranch(PatternRewriter &rewriter,
                                              Location loc, Value condition,
                                              Block *trueDest,
                                              Block *falseDest) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, cf::CondBranchOp::getOperationName());

  cf::CondBranchOp::build(builder, state, condition, trueDest, falseDest);
  return cast<cf::CondBranchOp>(rewriter.create(state));
}

/// Builds, creates and inserts a memref::AllocOp.
memref::AllocOp conversion::createAlloc(PatternRewriter &rewriter, Location loc,
                                        MemRefType memrefType,
                                        ValueRange dynamicSizes) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, memref::AllocOp::getOperationName());

  memref::AllocaOp::build(builder, state, memrefType, dynamicSizes);
  return cast<memref::AllocOp>(rewriter.create(state));
}

/// Builds, creates and inserts a memref::LoadOp.
memref::LoadOp conversion::createLoad(PatternRewriter &rewriter, Location loc,
                                      Value memref, ValueRange indices) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, memref::LoadOp::getOperationName());

  memref::LoadOp::build(builder, state, memref, indices);
  return cast<memref::LoadOp>(rewriter.create(state));
}

/// Builds, creates and inserts a memref::StoreOp.
memref::StoreOp conversion::createStore(PatternRewriter &rewriter, Location loc,
                                        Value value, Value memref,
                                        ValueRange indices) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, memref::StoreOp::getOperationName());

  memref::StoreOp::build(builder, state, value, memref, indices);
  return cast<memref::StoreOp>(rewriter.create(state));
}

/// Builds, creates and inserts a memref::CopyOp.
memref::CopyOp conversion::createCopy(PatternRewriter &rewriter, Location loc,
                                      Value source, Value target) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, memref::CopyOp::getOperationName());

  memref::CopyOp::build(builder, state, source, target);
  return cast<memref::CopyOp>(rewriter.create(state));
}

/// Allocates a symbol as a memref<i64> if it's not already allocated and
/// populates the symbol map.
void conversion::allocSymbol(PatternRewriter &rewriter, Location loc,
                             StringRef symName,
                             llvm::StringMap<Value> &symbolMap) {
  if (symbolMap.find(symName) != symbolMap.end())
    return;

  OpBuilder::InsertPoint insertionPoint = rewriter.saveInsertionPoint();

  // Set insertion point to the beginning of the first block (top of func)
  rewriter.setInsertionPointToStart(&rewriter.getBlock()->getParent()->front());

  IntegerType intType = IntegerType::get(loc->getContext(), 64);
  MemRefType memrefType = MemRefType::get({}, intType);
  memref::AllocOp allocOp = createAlloc(rewriter, loc, memrefType, {});

  // Update symbol map
  symbolMap[symName] = allocOp;
  rewriter.restoreInsertionPoint(insertionPoint);
}

/// Builds, creates and inserts an arith::ConstantIntOp.
arith::ConstantIntOp conversion::createConstantInt(PatternRewriter &rewriter,
                                                   Location loc, int val,
                                                   int width) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, arith::ConstantIntOp::getOperationName());

  arith::ConstantIntOp::build(builder, state, val, width);
  return cast<arith::ConstantIntOp>(rewriter.create(state));
}

/// Builds, creates and inserts an arith::AddIOp.
arith::AddIOp conversion::createAddI(PatternRewriter &rewriter, Location loc,
                                     Value a, Value b) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, arith::AddIOp::getOperationName());

  arith::AddIOp::build(builder, state, a, b);
  return cast<arith::AddIOp>(rewriter.create(state));
}

/// Builds, creates and inserts an arith::SubIOp.
arith::SubIOp conversion::createSubI(PatternRewriter &rewriter, Location loc,
                                     Value a, Value b) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, arith::SubIOp::getOperationName());

  arith::SubIOp::build(builder, state, a, b);
  return cast<arith::SubIOp>(rewriter.create(state));
}

/// Builds, creates and inserts an arith::MulIOp.
arith::MulIOp conversion::createMulI(PatternRewriter &rewriter, Location loc,
                                     Value a, Value b) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, arith::MulIOp::getOperationName());

  arith::MulIOp::build(builder, state, a, b);
  return cast<arith::MulIOp>(rewriter.create(state));
}

/// Builds, creates and inserts an arith::DivSIOp.
arith::DivSIOp conversion::createDivSI(PatternRewriter &rewriter, Location loc,
                                       Value a, Value b) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, arith::DivSIOp::getOperationName());

  arith::DivSIOp::build(builder, state, a, b);
  return cast<arith::DivSIOp>(rewriter.create(state));
}

/// Builds, creates and inserts an arith::FloorDivSIOp.
arith::FloorDivSIOp conversion::createFloorDivSI(PatternRewriter &rewriter,
                                                 Location loc, Value a,
                                                 Value b) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, arith::FloorDivSIOp::getOperationName());

  arith::FloorDivSIOp::build(builder, state, a, b);
  return cast<arith::FloorDivSIOp>(rewriter.create(state));
}

/// Builds, creates and inserts an arith::RemSIOp.
arith::RemSIOp conversion::createRemSI(PatternRewriter &rewriter, Location loc,
                                       Value a, Value b) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, arith::RemSIOp::getOperationName());

  arith::RemSIOp::build(builder, state, a, b);
  return cast<arith::RemSIOp>(rewriter.create(state));
}

/// Builds, creates and inserts an arith::OrIOp.
arith::OrIOp conversion::createOrI(PatternRewriter &rewriter, Location loc,
                                   Value a, Value b) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, arith::OrIOp::getOperationName());

  arith::OrIOp::build(builder, state, a, b);
  return cast<arith::OrIOp>(rewriter.create(state));
}

/// Builds, creates and inserts an arith::AndIOp.
arith::AndIOp conversion::createAndI(PatternRewriter &rewriter, Location loc,
                                     Value a, Value b) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, arith::AndIOp::getOperationName());

  arith::AndIOp::build(builder, state, a, b);
  return cast<arith::AndIOp>(rewriter.create(state));
}

/// Builds, creates and inserts an arith::XOrIOp.
arith::XOrIOp conversion::createXOrI(PatternRewriter &rewriter, Location loc,
                                     Value a, Value b) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, arith::XOrIOp::getOperationName());

  arith::XOrIOp::build(builder, state, a, b);
  return cast<arith::XOrIOp>(rewriter.create(state));
}

/// Builds, creates and inserts an arith::ShLIOp.
arith::ShLIOp conversion::createShLI(PatternRewriter &rewriter, Location loc,
                                     Value a, Value b) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, arith::ShLIOp::getOperationName());

  arith::ShLIOp::build(builder, state, a, b);
  return cast<arith::ShLIOp>(rewriter.create(state));
}

/// Builds, creates and inserts an arith::ShRSIOp.
arith::ShRSIOp conversion::createShRSI(PatternRewriter &rewriter, Location loc,
                                       Value a, Value b) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, arith::ShRSIOp::getOperationName());

  arith::ShRSIOp::build(builder, state, a, b);
  return cast<arith::ShRSIOp>(rewriter.create(state));
}

/// Builds, creates and inserts an arith::CmpIOp.
arith::CmpIOp conversion::createCmpI(PatternRewriter &rewriter, Location loc,
                                     arith::CmpIPredicate predicate, Value lhs,
                                     Value rhs) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, arith::CmpIOp::getOperationName());

  arith::CmpIOp::build(builder, state, predicate, lhs, rhs);
  return cast<arith::CmpIOp>(rewriter.create(state));
}

/// Builds, creates and inserts an arith::ExtSIOp.
arith::ExtSIOp conversion::createExtSI(PatternRewriter &rewriter, Location loc,
                                       Type out, Value in) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, arith::ExtSIOp::getOperationName());

  arith::ExtSIOp::build(builder, state, out, in);
  return cast<arith::ExtSIOp>(rewriter.create(state));
}

/// Builds, creates and inserts an arith::TruncIOp.
arith::TruncIOp conversion::createTruncI(PatternRewriter &rewriter,
                                         Location loc, Type out, Value in) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, arith::TruncIOp::getOperationName());

  arith::TruncIOp::build(builder, state, out, in);
  return cast<arith::TruncIOp>(rewriter.create(state));
}

/// Builds, creates and inserts an arith::IndexCastOp.
arith::IndexCastOp conversion::createIndexCast(PatternRewriter &rewriter,
                                               Location loc, Type out,
                                               Value in) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, arith::IndexCastOp::getOperationName());

  arith::IndexCastOp::build(builder, state, out, in);
  return cast<arith::IndexCastOp>(rewriter.create(state));
}

/// Builds, creates and inserts a scf::ParallelOp.
scf::ParallelOp conversion::createParallel(PatternRewriter &rewriter,
                                           Location loc, ValueRange lowerBounds,
                                           ValueRange upperBounds,
                                           ValueRange steps) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, scf::ParallelOp::getOperationName());

  scf::ParallelOp::build(builder, state, lowerBounds, upperBounds, steps);
  return cast<scf::ParallelOp>(rewriter.create(state));
}

/// Builds, creates and inserts a scf::YieldOp.
scf::YieldOp conversion::createYield(PatternRewriter &rewriter, Location loc) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, scf::YieldOp::getOperationName());

  scf::YieldOp::build(builder, state);
  return cast<scf::YieldOp>(rewriter.create(state));
}
