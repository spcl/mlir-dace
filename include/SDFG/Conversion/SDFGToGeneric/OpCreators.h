// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

/// Header for convenience functions, creating various operations.

#ifndef SDFG_Conversion_SDFGToGeneric_Op_Creators_H
#define SDFG_Conversion_SDFGToGeneric_Op_Creators_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace mlir::sdfg::conversion {

/// Builds, creates and inserts a func::FuncOp.
func::FuncOp createFunc(PatternRewriter &rewriter, Location loc, StringRef name,
                        TypeRange inputTypes, TypeRange resultTypes,
                        StringRef visibility);

/// Builds, creates and inserts a func::CallOp.
func::CallOp createCall(PatternRewriter &rewriter, Location loc,
                        TypeRange resultTypes, StringRef callee,
                        ValueRange operands);

/// Builds, creates and inserts a func::ReturnOp.
func::ReturnOp createReturn(PatternRewriter &rewriter, Location loc,
                            ValueRange operands);

/// Builds, creates and inserts a cf::BranchOp.
cf::BranchOp createBranch(PatternRewriter &rewriter, Location loc,
                          ValueRange operands, Block *dest);

/// Builds, creates and inserts a cf::CondBranchOp.
cf::CondBranchOp createCondBranch(PatternRewriter &rewriter, Location loc,
                                  Value condition, Block *trueDest,
                                  Block *falseDest);

/// Builds, creates and inserts a memref::AllocOp.
memref::AllocOp createAlloc(PatternRewriter &rewriter, Location loc,
                            MemRefType memrefType, ValueRange dynamicSizes);

/// Builds, creates and inserts a memref::LoadOp.
memref::LoadOp createLoad(PatternRewriter &rewriter, Location loc, Value memref,
                          ValueRange indices);

/// Builds, creates and inserts a memref::StoreOp.
memref::StoreOp createStore(PatternRewriter &rewriter, Location loc,
                            Value value, Value memref, ValueRange indices);

/// Builds, creates and inserts a memref::CopyOp.
memref::CopyOp createCopy(PatternRewriter &rewriter, Location loc, Value source,
                          Value target);

/// Allocates a symbol as a memref<i64> if it's not already allocated and
/// populates the symbol map.
void allocSymbol(PatternRewriter &rewriter, Location loc, StringRef symName,
                 llvm::StringMap<Value> &symbolMap);

/// Builds, creates and inserts an arith::ConstantIntOp.
arith::ConstantIntOp createConstantInt(PatternRewriter &rewriter, Location loc,
                                       int val, int width);

/// Builds, creates and inserts an arith::AddIOp.
arith::AddIOp createAddI(PatternRewriter &rewriter, Location loc, Value a,
                         Value b);

/// Builds, creates and inserts an arith::SubIOp.
arith::SubIOp createSubI(PatternRewriter &rewriter, Location loc, Value a,
                         Value b);

/// Builds, creates and inserts an arith::MulIOp.
arith::MulIOp createMulI(PatternRewriter &rewriter, Location loc, Value a,
                         Value b);

/// Builds, creates and inserts an arith::DivSIOp.
arith::DivSIOp createDivSI(PatternRewriter &rewriter, Location loc, Value a,
                           Value b);

/// Builds, creates and inserts an arith::FloorDivSIOp.
arith::FloorDivSIOp createFloorDivSI(PatternRewriter &rewriter, Location loc,
                                     Value a, Value b);

/// Builds, creates and inserts an arith::RemSIOp.
arith::RemSIOp createRemSI(PatternRewriter &rewriter, Location loc, Value a,
                           Value b);

/// Builds, creates and inserts an arith::OrIOp.
arith::OrIOp createOrI(PatternRewriter &rewriter, Location loc, Value a,
                       Value b);

/// Builds, creates and inserts an arith::AndIOp.
arith::AndIOp createAndI(PatternRewriter &rewriter, Location loc, Value a,
                         Value b);

/// Builds, creates and inserts an arith::XOrIOp.
arith::XOrIOp createXOrI(PatternRewriter &rewriter, Location loc, Value a,
                         Value b);

/// Builds, creates and inserts an arith::ShLIOp.
arith::ShLIOp createShLI(PatternRewriter &rewriter, Location loc, Value a,
                         Value b);

/// Builds, creates and inserts an arith::ShRSIOp.
arith::ShRSIOp createShRSI(PatternRewriter &rewriter, Location loc, Value a,
                           Value b);

/// Builds, creates and inserts an arith::CmpIOp.
arith::CmpIOp createCmpI(PatternRewriter &rewriter, Location loc,
                         arith::CmpIPredicate predicate, Value lhs, Value rhs);

/// Builds, creates and inserts an arith::ExtSIOp.
arith::ExtSIOp createExtSI(PatternRewriter &rewriter, Location loc, Type out,
                           Value in);

/// Builds, creates and inserts an arith::TruncIOp.
arith::TruncIOp createTruncI(PatternRewriter &rewriter, Location loc, Type out,
                             Value in);

/// Builds, creates and inserts an arith::IndexCastOp.
arith::IndexCastOp createIndexCast(PatternRewriter &rewriter, Location loc,
                                   Type out, Value in);

/// Builds, creates and inserts a scf::ParallelOp.
scf::ParallelOp createParallel(PatternRewriter &rewriter, Location loc,
                               ValueRange lowerBounds, ValueRange upperBounds,
                               ValueRange steps);

/// Builds, creates and inserts a scf::YieldOp.
scf::YieldOp createYield(PatternRewriter &rewriter, Location loc);

} // namespace mlir::sdfg::conversion

#endif // SDFG_Conversion_SDFGToGeneric_Op_Creators_H
