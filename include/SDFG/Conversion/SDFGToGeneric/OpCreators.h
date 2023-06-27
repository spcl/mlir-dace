#ifndef SDFG_Conversion_SDFGToGeneric_Op_Creators_H
#define SDFG_Conversion_SDFGToGeneric_Op_Creators_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace mlir::sdfg::conversion {

func::FuncOp createFunc(PatternRewriter &rewriter, Location loc, StringRef name,
                        TypeRange inputTypes, TypeRange resultTypes,
                        StringRef visibility);

func::CallOp createCall(PatternRewriter &rewriter, Location loc,
                        TypeRange resultTypes, StringRef callee,
                        ValueRange operands);

func::ReturnOp createReturn(PatternRewriter &rewriter, Location loc,
                            ValueRange operands);

cf::BranchOp createBranch(PatternRewriter &rewriter, Location loc,
                          ValueRange operands, Block *dest);

cf::CondBranchOp createCondBranch(PatternRewriter &rewriter, Location loc,
                                  Value condition, Block *trueDest,
                                  Block *falseDest);

memref::AllocOp createAlloc(PatternRewriter &rewriter, Location loc,
                            MemRefType memreftype);

memref::LoadOp createLoad(PatternRewriter &rewriter, Location loc, Value memref,
                          ValueRange indices);

memref::StoreOp createStore(PatternRewriter &rewriter, Location loc,
                            Value value, Value memref, ValueRange indices);

void allocSymbol(PatternRewriter &rewriter, Location loc, StringRef symName,
                 llvm::StringMap<memref::AllocOp> &symbolMap);

arith::ConstantIntOp createConstantInt(PatternRewriter &rewriter, Location loc,
                                       int val, int width);

arith::AddIOp createAddI(PatternRewriter &rewriter, Location loc, Value a,
                         Value b);

arith::SubIOp createSubI(PatternRewriter &rewriter, Location loc, Value a,
                         Value b);

arith::MulIOp createMulI(PatternRewriter &rewriter, Location loc, Value a,
                         Value b);

arith::DivSIOp createDivSI(PatternRewriter &rewriter, Location loc, Value a,
                           Value b);

arith::FloorDivSIOp createFloorDivSI(PatternRewriter &rewriter, Location loc,
                                     Value a, Value b);

arith::RemSIOp createRemSI(PatternRewriter &rewriter, Location loc, Value a,
                           Value b);

arith::OrIOp createOrI(PatternRewriter &rewriter, Location loc, Value a,
                       Value b);

arith::AndIOp createAndI(PatternRewriter &rewriter, Location loc, Value a,
                         Value b);

arith::XOrIOp createXOrI(PatternRewriter &rewriter, Location loc, Value a,
                         Value b);

arith::ShLIOp createShLI(PatternRewriter &rewriter, Location loc, Value a,
                         Value b);

arith::ShRSIOp createShRSI(PatternRewriter &rewriter, Location loc, Value a,
                           Value b);

arith::CmpIOp createCmpI(PatternRewriter &rewriter, Location loc,
                         arith::CmpIPredicate predicate, Value lhs, Value rhs);

arith::ExtSIOp createExtSI(PatternRewriter &rewriter, Location loc, Type out,
                           Value in);

arith::IndexCastOp createIndexCast(PatternRewriter &rewriter, Location loc,
                                   Type out, Value in);

} // namespace mlir::sdfg::conversion

#endif // SDFG_Conversion_SDFGToGeneric_Op_Creators_H
