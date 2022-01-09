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
      return true;
      //(op.getOps().empty() || !op.getOps<SDFGNode>().empty());
    });
    // All other operations are illegal
    // markUnknownOpDynamicallyLegal([](Operation *op) { return false; });
  }
};

class FuncToSDFG : public OpRewritePattern<FuncOp> {
public:
  using OpRewritePattern<FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FuncOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    SmallVector<AllocOp> allocs;
    SmallVector<GetAccessOp> access;

    for (BlockArgument arg : op.getArguments()) {
      Type t = arg.getType();
      SmallVector<int64_t> ints;
      SmallVector<bool> shape;

      /*if (MemRefType mrt = t.dyn_cast<MemRefType>()) {
        t = mrt.getElementType();

        for (int64_t dim : mrt.getShape()) {
          ints.push_back(dim);
          shape.push_back(true);
        }
      }*/

      ArrayType art = ArrayType::get(loc->getContext(), t, {}, ints, shape);
      AllocOp alloc = AllocOp::create(loc, art);
      allocs.push_back(alloc);

      GetAccessOp acc = GetAccessOp::create(loc, alloc);
      access.push_back(acc);
    }

    SDFGNode sdfg = SDFGNode::create(loc, allocs);
    StateNode state = StateNode::create(loc);
    sdfg.addState(state, /*isEntry=*/true);

    TaskletNode task = TaskletNode::create(loc, op.getType());
    task.body().takeBody(op.body());
    state.addOp(*task);

    for (GetAccessOp acc : access) {
      state.addOp(*acc);
    }

    SmallVector<Value> accessV;
    for (GetAccessOp acc : access) {
      accessV.push_back(acc);
    }
    ValueRange vr = ValueRange(accessV);
    // TODO: Add loads before
    // sdir::CallOp call = sdir::CallOp::create(loc, task, vr);
    //  state.addOp(*call);

    rewriter.insert(sdfg);
    rewriter.eraseOp(op);
    return success();
  }
};

class TaskletTerminator : public OpRewritePattern<mlir::ReturnOp> {
public:
  using OpRewritePattern<mlir::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ReturnOp op,
                                PatternRewriter &rewriter) const override {
    if (TaskletNode tn = dyn_cast<TaskletNode>(op->getParentOp())) {
      sdir::ReturnOp ret =
          sdir::ReturnOp::create(op.getLoc(), op.getOperands());
      rewriter.insert(ret);
      rewriter.eraseOp(op);
      return success();
    }

    return failure();
  }
};

void populateSAMToSDIRConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<FuncToSDFG, TaskletTerminator>(patterns.getContext());
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
}

std::unique_ptr<Pass> conversion::createSAMToSDIRPass() {
  return std::make_unique<SAMToSDIRPass>();
}
