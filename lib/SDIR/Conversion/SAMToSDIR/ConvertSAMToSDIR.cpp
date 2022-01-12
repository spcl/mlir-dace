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

    rewriter.cloneRegionBefore(op.body(), state.body(), state.body().begin());

    // hasValue() is inaccessable
    if (rewriter.convertRegionTypes(&state.body(), *getTypeConverter())
            .getPointer() == nullptr) {
      return failure();
    }

    for (unsigned i = 0; i < state.body().getBlocks().front().getNumArguments();
         ++i) {
      rewriter.replaceUsesOfBlockArgument(
          state.body().getBlocks().front().getArgument(i), sdfg.getArgument(i));
    }

    // NOTE: Consider using rewriter
    state.body().getBlocks().front().eraseArguments(
        [](BlockArgument ba) { return true; });

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

      // NOTE: For debugging only
      if (scf::ForOp task = dyn_cast<scf::ForOp>(op->getParentOp())) {
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

      rewriter.setInsertionPointAfter(task);
      sdir::CallOp call =
          sdir::CallOp::create(rewriter, op->getLoc(), task, op->getOperands());

      rewriter.replaceOp(op, call.getResults());

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
    LoadOp load = LoadOp::create(rewriter, op.getLoc(),
                                 getTypeConverter()->convertType(op.getType()),
                                 adaptor.memref(), adaptor.indices());

    rewriter.replaceOp(op, load.getResult());
    return success();
  }
};

class MemrefStoreToSDIR : public OpConversionPattern<memref::StoreOp> {
public:
  using OpConversionPattern<memref::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StoreOp::create(rewriter, op.getLoc(), adaptor.value(), adaptor.memref(),
                    adaptor.indices());
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
  patterns.add<MemrefStoreToSDIR>(converter, ctxt);
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
