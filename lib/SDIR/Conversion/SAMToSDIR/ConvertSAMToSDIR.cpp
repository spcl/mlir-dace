#include "SDIR/Conversion/SAMToSDIR/PassDetail.h"
#include "SDIR/Conversion/SAMToSDIR/Passes.h"
#include "SDIR/Dialect/Dialect.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/SCF/SCF.h"

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
      Region::OpIterator b = op.getOps().begin();
      Region::OpIterator e = op.getOps().end();
      return true;
      //(op.getOps().empty() || !op.getOps<SDFGNode>().empty());
    });
    // All other operations are illegal
    // markUnknownOpDynamicallyLegal([](Operation *op) { return false; });
  }
};

class SCFForConversion : public OpRewritePattern<scf::ForOp> {
public:
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    return failure();
  }
};

class FuncToSDFG : public OpRewritePattern<FuncOp> {
public:
  using OpRewritePattern<FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FuncOp op,
                                PatternRewriter &rewriter) const override {
    SDFGNode sdfg = SDFGNode::create(op.getLoc(), op.getType());
    StateNode state = StateNode::create(op.getLoc());
    sdfg.addState(state, /*isEntry=*/true);

    /*TaskletNode::create(op.getLoc(), "tsklt", op.getType(),
                        ArrayRef<NamedAttribute>());*/

    rewriter.insert(sdfg);
    rewriter.eraseOp(op);

    return success();
  }
};

void populateSAMToSDIRConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<FuncToSDFG, SCFForConversion>(patterns.getContext());
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
  populateSAMToSDIRConversionPatterns(patterns);

  SDIRTarget target(getContext());
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();

  // Build SDFG
  /*SDFGNode sdfg = SDFGNode::create(module.getLoc());
  StateNode statenode = StateNode::create(module.getLoc());
  sdfg.body().getBlocks().front().push_front(statenode);

  // Replace block content with SDFG
  Block &moduleBlock = module.body().getBlocks().front();
  moduleBlock.clear();
  moduleBlock.push_front(sdfg);*/
}

std::unique_ptr<Pass> conversion::createSAMToSDIRPass() {
  return std::make_unique<SAMToSDIRPass>();
}
