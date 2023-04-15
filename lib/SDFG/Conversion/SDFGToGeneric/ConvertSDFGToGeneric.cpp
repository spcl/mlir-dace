#include "SDFG/Conversion/SDFGToGeneric/PassDetail.h"
#include "SDFG/Conversion/SDFGToGeneric/Passes.h"
#include "SDFG/Conversion/SDFGToGeneric/SymbolicParser.h"
#include "SDFG/Dialect/Dialect.h"
#include "SDFG/Utils/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace sdfg;
using namespace conversion;

// Maps state name to their generated block
llvm::StringMap<Block *> blockMap;
// Maps symbols to the generated allocation operation
llvm::StringMap<memref::AllocOp *> symbolMap;

//
// SDFG -> func.func
// States -> block
// Edges ->
//       Default: cf.br
//       Condition: Insert block (false branch) => compute condition =>
//       cf.cond_br Assignment: Add assignments in the target block => cf.br
//
// Alloc -> memref.alloc
// Load -> memref.load
// Store -> memref.store
// Copy -> memref.copy
//
// Alloc Symbol -> memref.alloc (int64)
// Sym ->
//      If single symbol: memref.load
//      If expression: not supported for now
//
// Return -> func.return
// Tasklet -> func.func + func.call
//
// Map -> affine.parallel, affine.for, scf.forall, scf.for
//

//===----------------------------------------------------------------------===//
// Target & Type Converter
//===----------------------------------------------------------------------===//

struct GenericTarget : public ConversionTarget {
  GenericTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    // Every Op in the SDFG Dialect is illegal
    addIllegalDialect<SDFGDialect>();
    // Implicit top level module operation is legal
    addLegalOp<ModuleOp>();
    // Operations generated by this pass are legal
    addLegalOp<func::FuncOp>();
    addLegalOp<func::ReturnOp>();
    addLegalOp<cf::BranchOp>();
    addLegalOp<cf::CondBranchOp>();
    addLegalOp<memref::AllocOp>();
    addLegalOp<memref::LoadOp>();
    addLegalOp<memref::StoreOp>();
    addLegalOp<memref::CopyOp>();
    // All other operations are illegal
    markUnknownOpDynamicallyLegal([](Operation *op) { return true; });
  }
};

class ToMemrefConverter : public TypeConverter {
public:
  ToMemrefConverter() {
    addConversion([](Type type) { return type; });
    addConversion(convertMemrefTypes);
  }

  static Optional<Type> convertMemrefTypes(Type type) {
    if (MemRefType mem = type.dyn_cast<MemRefType>()) {
      SmallVector<int64_t> ints;
      SmallVector<StringAttr> symbols;
      SmallVector<bool> shape;
      for (int64_t dim : mem.getShape()) {
        if (dim <= 0) {
          StringAttr sym =
              StringAttr::get(mem.getContext(), sdfg::utils::generateName("s"));
          symbols.push_back(sym);
          shape.push_back(false);
        } else {
          ints.push_back(dim);
          shape.push_back(true);
        }
      }
      SizedType sized = SizedType::get(mem.getContext(), mem.getElementType(),
                                       symbols, ints, shape);

      return ArrayType::get(mem.getContext(), sized);
    }
    return std::nullopt;
  }
};

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

func::FuncOp createFunc(PatternRewriter &rewriter, Location loc, StringRef name,
                        TypeRange inputTypes, TypeRange resultTypes,
                        StringRef visibility) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, func::FuncOp::getOperationName());

  FunctionType func_type = builder.getFunctionType(inputTypes, resultTypes);
  StringAttr visAttr = builder.getStringAttr(visibility);

  func::FuncOp::build(builder, state, name, func_type, visAttr, {}, {});
  return cast<func::FuncOp>(rewriter.create(state));
}

func::ReturnOp createReturn(PatternRewriter &rewriter, Location loc,
                            ValueRange operands) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, func::ReturnOp::getOperationName());

  func::ReturnOp::build(builder, state, operands);
  return cast<func::ReturnOp>(rewriter.create(state));
}

cf::BranchOp createBranch(PatternRewriter &rewriter, Location loc,
                          ValueRange operands, Block *dest) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, cf::BranchOp::getOperationName());

  cf::BranchOp::build(builder, state, operands, dest);
  return cast<cf::BranchOp>(rewriter.create(state));
}

cf::CondBranchOp createCondBranch(PatternRewriter &rewriter, Location loc,
                                  Value condition, Block *trueDest,
                                  Block *falseDest) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, cf::CondBranchOp::getOperationName());

  cf::CondBranchOp::build(builder, state, condition, trueDest, falseDest);
  return cast<cf::CondBranchOp>(rewriter.create(state));
}

memref::AllocOp createAlloc(PatternRewriter &rewriter, Location loc,
                            MemRefType memreftype) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, memref::AllocOp ::getOperationName());

  memref::AllocOp ::build(builder, state, memreftype);
  return cast<memref::AllocOp>(rewriter.create(state));
}

// Allocates a symbol as a memref<i64> if it's not already allocated
void allocSymbol(PatternRewriter &rewriter, Location loc, StringRef symName) {
  if (symbolMap.find(symName) != symbolMap.end())
    return;

  OpBuilder::InsertPoint insertionPoint = rewriter.saveInsertionPoint();

  // Set insertion point to the beginning of the first block (top of func)
  rewriter.setInsertionPointToStart(&rewriter.getBlock()->getParent()->front());

  IntegerType intType = IntegerType::get(loc->getContext(), 64);
  MemRefType memrefType = MemRefType::get({}, intType);
  memref::AllocOp allocOp = createAlloc(rewriter, loc, memrefType);

  // Update symbol map
  symbolMap[symName] = &allocOp;

  rewriter.restoreInsertionPoint(insertionPoint);
}

// Creates operations that perform the symbolic expression
void symbolicExpressionToMLIR(PatternRewriter &rewriter, Location loc,
                              StringRef symExpr) {
  std::unique_ptr<Node> ast = SymbolicParser::parse(symExpr);
  if (!ast)
    emitError(loc, "failed to parse symbolic expression");

  // SmallVector<std::string> symbols;
  // ast->collect_variables(symbols);

  // for (std::string symbol : symbols)
  //   allocSymbol(rewriter, loc, symbol);

  // Value result = ast->codegen(rewriter, loc);
}

//===----------------------------------------------------------------------===//
// SDFG, State & Edge Patterns
//===----------------------------------------------------------------------===//

class SDFGToFunc : public OpConversionPattern<SDFGNode> {
public:
  using OpConversionPattern<SDFGNode>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SDFGNode op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Mark the entry state
    op.getEntryState()->setAttr("entry", rewriter.getBoolAttr(true));

    // Create a function and clone the sdfg body
    func::FuncOp funcOp =
        createFunc(rewriter, op.getLoc(), "sdfg_func",
                   op.getBody().getArgumentTypes(), {}, "public");
    funcOp.getBody().takeBody(op.getBody());

    rewriter.eraseOp(op);
    return success();
  }
};

class StateToBlock : public OpConversionPattern<StateNode> {
public:
  using OpConversionPattern<StateNode>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StateNode op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Split the current basic block at the current position
    Block *newBlock =
        rewriter.splitBlock(rewriter.getBlock(), rewriter.getInsertionPoint());

    // Add the mapping from the sdfg.state's name to the new basic block
    blockMap[op.getName()] = newBlock;

    // Connect to init block if it's an entry state
    if (op->hasAttrOfType<BoolAttr>("entry") &&
        op->getAttrOfType<BoolAttr>("entry").getValue()) {
      rewriter.setInsertionPointToEnd(&newBlock->getParent()->front());
      createBranch(rewriter, op.getLoc(), {}, newBlock);
    }

    // Clone the operations from the sdfg.state's body into the new basic block
    rewriter.setInsertionPointToStart(newBlock);

    for (Operation &operation : op.getBody().getOps()) {
      rewriter.clone(operation);
    }

    // If there is an outward edge, do not add a return op
    for (EdgeOp edge : op->getParentRegion()->getOps<EdgeOp>()) {
      if (edge.getSrc().equals(op.getSymName())) {
        rewriter.eraseOp(op);
        return success();
      }
    }

    createReturn(rewriter, op.getLoc(), {});
    rewriter.eraseOp(op);
    return success();
  }
};

class EdgeToBranch : public OpConversionPattern<EdgeOp> {
public:
  using OpConversionPattern<EdgeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(EdgeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Block *srcBlock = blockMap[adaptor.getSrc()];
    Block *destBlock = blockMap[adaptor.getDest()];

    if (!adaptor.getAssign().empty()) {
      rewriter.setInsertionPointToEnd(destBlock);
      for (Attribute assignment : adaptor.getAssign())
        symbolicExpressionToMLIR(rewriter, op.getLoc(),
                                 cast<StringAttr>(assignment));
    }

    rewriter.setInsertionPointToEnd(srcBlock);

    // If we don't have a condition (always true), add a simple branch
    if (adaptor.getCondition().equals("1")) {
      createBranch(rewriter, op.getLoc(), {}, destBlock);
      rewriter.eraseOp(op);
      return success();
    }

    // If we have a condition, create a new block (not taken path)
    Block *newBlock =
        rewriter.splitBlock(rewriter.getBlock(), rewriter.getInsertionPoint());

    rewriter.setInsertionPointToEnd(srcBlock);

    // Compute condition
    // symbolicExpressionToMLIR(rewriter, op.getLoc(), adaptor.getCondition());

    // Add conditional branch
    // createCondBranch(rewriter, op.getLoc(), condition, destBlock, newBlock);

    // Update blockMap
    blockMap[adaptor.getSrc()] = newBlock;

    // If there is another edge op for the source state, don't add return
    // statement to the new block
    rewriter.eraseOp(op);

    for (EdgeOp edge : op->getParentRegion()->getOps<EdgeOp>()) {
      if (edge.getSrc().equals(adaptor.getSrc())) {
        return success();
      }
    }

    rewriter.setInsertionPointToEnd(newBlock);
    createReturn(rewriter, op.getLoc(), {});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SDFG Patterns
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

void populateSDFGToGenericConversionPatterns(RewritePatternSet &patterns,
                                             TypeConverter &converter) {
  MLIRContext *ctxt = patterns.getContext();

  patterns.add<SDFGToFunc>(converter, ctxt);
  patterns.add<StateToBlock>(converter, ctxt);
  patterns.add<EdgeToBranch>(converter, ctxt);
}

namespace {
struct SDFGToGenericPass
    : public sdfg::conversion::SDFGToGenericPassBase<SDFGToGenericPass> {
  void runOnOperation() override;
};
} // namespace

void SDFGToGenericPass::runOnOperation() {
  ModuleOp module = getOperation();

  GenericTarget target(getContext());
  ToMemrefConverter converter;

  RewritePatternSet patterns(&getContext());
  populateSDFGToGenericConversionPatterns(patterns, converter);

  if (applyPartialConversion(module, target, std::move(patterns)).failed())
    signalPassFailure();

  module.dump();
}

std::unique_ptr<Pass> conversion::createSDFGToGenericPass() {
  return std::make_unique<SDFGToGenericPass>();
}
