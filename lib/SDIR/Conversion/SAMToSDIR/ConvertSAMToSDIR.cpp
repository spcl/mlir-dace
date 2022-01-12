#include "SDIR/Conversion/SAMToSDIR/PassDetail.h"
#include "SDIR/Conversion/SAMToSDIR/Passes.h"
#include "SDIR/Dialect/Dialect.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"

using namespace mlir;
using namespace sdir;
using namespace conversion;

struct SDIRTarget : public ConversionTarget {
  SDIRTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    // Every Op in the SDIR Dialect is legal
    addLegalDialect<SDIRDialect>();
    // Implicit top level module operation is legal
    // if it only contains a single SDFGNode or is empty
    // TODO: Add checks
    addDynamicallyLegalOp<ModuleOp>([](ModuleOp op) {
      return true;
      //(op.getOps().empty() || !op.getOps<SDFGNode>().empty());
    });
    // All other operations are illegal
    // NOTE: Disabled for debugging
    // markUnknownOpDynamicallyLegal([](Operation *op) { return false; });
  }
};

class MemrefTypeConverter : public TypeConverter {
public:
  MemrefTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion(convertMemrefTypes);
  }

  static Optional<Type> convertMemrefTypes(Type type) {
    if (MemRefType mem = type.dyn_cast<MemRefType>()) {
      SmallVector<int64_t> ints;
      SmallVector<bool> shape;
      for (int64_t dim : mem.getShape()) {
        ints.push_back(dim);
        shape.push_back(true);
      }
      return MemletType::get(mem.getContext(), mem.getElementType(), {}, ints,
                             shape);
    }
    return llvm::None;
  }
};

// Helper function. Recursively converting region types
void convertRec(Region &region, TypeConverter &converter,
                ConversionPatternRewriter &rewriter) {
  FailureOr<Block *> res = rewriter.convertRegionTypes(&region, converter);

  // hasValue() is inaccessible
  if (res.getPointer() == nullptr)
    return;

  Block *b = res.getValue();
  for (Operation &op : b->getOperations()) {
    MutableArrayRef<Region> regions = op.getRegions();
    for (size_t i = 0; i < regions.size(); ++i) {
      convertRec(regions[i], converter, rewriter);
    }
  }
}

class FuncToSDFG : public OpConversionPattern<FuncOp> {
public:
  using OpConversionPattern<FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    SmallVector<Type> inputResults;
    if (getTypeConverter()
            ->convertTypes(op.getType().getInputs(), inputResults)
            .failed())
      return failure();

    SmallVector<Type> outputResults;
    if (getTypeConverter()
            ->convertTypes(op.getType().getResults(), outputResults)
            .failed())
      return failure();

    FunctionType ft = rewriter.getFunctionType(inputResults, outputResults);

    SDFGNode sdfg = SDFGNode::create(rewriter, op.getLoc(), ft);
    StateNode state = StateNode::create(rewriter, op.getLoc());

    rewriter.updateRootInPlace(sdfg, [&] {
      sdfg.entryAttr(
          SymbolRefAttr::get(op.getLoc().getContext(), state.sym_name()));
    });

    rewriter.updateRootInPlace(state,
                               [&] { state.body().takeBody(op.body()); });

    convertRec(sdfg.body(), *getTypeConverter(), rewriter);

    rewriter.eraseOp(op);
    return success();
  }
};

class OpToTasklet : public RewritePattern {
public:
  OpToTasklet(PatternBenefit benefit, MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (mlir::ReturnOp rop = dyn_cast<mlir::ReturnOp>(op)) {
      rewriter.eraseOp(op);
      return success();
    }

    // TODO: Check if there is a proper way of doing this
    if (op->getDialect()->getNamespace() == "arith" ||
        op->getDialect()->getNamespace() == "math") {
      if (TaskletNode task = dyn_cast<TaskletNode>(op->getParentOp())) {
        // Operation already in a tasklet
        return failure();
      }

      FunctionType ft =
          rewriter.getFunctionType(op->getOperandTypes(), op->getResultTypes());
      TaskletNode task = TaskletNode::create(rewriter, op->getLoc(), ft);

      BlockAndValueMapping mapping;
      mapping.map(op->getOperands(), task.getArguments());

      // NOTE: Consider using move() or rewriter.clone()
      Operation *opClone = op->clone(mapping);
      rewriter.updateRootInPlace(
          task, [&] { task.body().getBlocks().front().push_front(opClone); });

      sdir::ReturnOp::create(rewriter, opClone->getLoc(),
                             opClone->getResults());

      rewriter.eraseOp(op);
      return success();
    }

    return failure();
  }
};

class MemrefLoadToSDIR : public OpConversionPattern<memref::LoadOp> {
public:
  using OpConversionPattern<memref::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LoadOp::create(rewriter, op.getLoc(),
                   getTypeConverter()->convertType(op.getType()),
                   adaptor.memref(), adaptor.indices());
    rewriter.eraseOp(op);
    return success();
  }
};

void populateSAMToSDIRConversionPatterns(RewritePatternSet &patterns,
                                         TypeConverter &converter) {
  MLIRContext *ctxt = patterns.getContext();
  patterns.add<FuncToSDFG>(converter, ctxt);
  patterns.add<OpToTasklet>(1, ctxt);
  patterns.add<MemrefLoadToSDIR>(converter, ctxt);
}

namespace {
struct SAMToSDIRPass : public SAMToSDIRPassBase<SAMToSDIRPass> {
  void runOnOperation() override;
};
} // namespace

void SAMToSDIRPass::runOnOperation() {
  // NOTE: Maybe change to a pass working on funcs?
  ModuleOp module = getOperation();

  RewritePatternSet patterns(&getContext());
  MemrefTypeConverter converter;
  populateSAMToSDIRConversionPatterns(patterns, converter);

  SDIRTarget target(getContext());
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> conversion::createSAMToSDIRPass() {
  return std::make_unique<SAMToSDIRPass>();
}
