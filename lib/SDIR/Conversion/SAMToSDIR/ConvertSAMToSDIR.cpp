#include "SDIR/Conversion/SAMToSDIR/PassDetail.h"
#include "SDIR/Conversion/SAMToSDIR/Passes.h"
#include "SDIR/Dialect/Dialect.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

    SDFGNode sdfg = SDFGNode::create(rewriter, loc, op.getType());
    StateNode state = StateNode::create(rewriter, loc);
    sdfg.addEntryState(state);

    TaskletNode task = TaskletNode::create(rewriter, loc, op.getType());

    // TODO: Use rewriter
    // rewriter.cloneRegionBefore(&op.body(), &task.body(), task.body());
    task.body().takeBody(op.body());

    // sdir::CallOp::create(rewriter, loc, task, sdfg.getArguments());

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
      sdir::ReturnOp::create(rewriter, op.getLoc(), op.getOperands());
      rewriter.eraseOp(op);
      return success();
    }

    return failure();
  }
};

class ConstantPromotion : public OpRewritePattern<arith::ConstantOp> {
public:
  using OpRewritePattern<arith::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ConstantOp op,
                                PatternRewriter &rewriter) const override {

    if (TaskletNode task = dyn_cast<TaskletNode>(op->getParentOp())) {
      StateNode state = cast<StateNode>(task->getParentOp());
      TaskletNode taskC = TaskletNode::create(rewriter, op.getLoc(), op);
      sdir::CallOp callC =
          sdir::CallOp::create(rewriter, op.getLoc(), taskC, {});
      // state.addOp(*callC, /*toFront=*/true);
      // state.addOp(*taskC, /*toFront=*/true);

      unsigned numArgs = task.getNumArguments();
      task.insertArgument(numArgs, op.getType(), {});

      for (sdir::CallOp callOp : state.getOps<sdir::CallOp>()) {
        if (callOp.callee() == task.sym_name()) {
          SmallVector<Value> operands;
          for (Value v : callOp.getOperands())
            operands.push_back(v);
          operands.push_back(callC.getResult(0));
          ValueRange vr = ValueRange(operands);
          callOp->setOperands(vr);
        }
      }

      rewriter.replaceOp(op, {task.getArgument(numArgs)});
      return success();
    }

    return failure();
  }
};

// TODO: This Pattern doesn't get called
class TaskletReordering : public OpRewritePattern<TaskletNode> {
public:
  using OpRewritePattern<TaskletNode>::OpRewritePattern;

  LogicalResult matchAndRewrite(TaskletNode op,
                                PatternRewriter &rewriter) const override {
    StateNode state = cast<StateNode>(op->getParentOp());
    Operation *prev =
        state.body().getBlocks().front().findAncestorOpInBlock(*op);
    if (TaskletNode task = dyn_cast<TaskletNode>(prev))
      return failure();

    rewriter.eraseOp(op);
    rewriter.eraseOp(prev);

    rewriter.insert(prev);
    rewriter.insert(op);

    return success();
  }
};

class MemrefLoadToSDIR : public OpRewritePattern<memref::LoadOp> {
public:
  using OpRewritePattern<memref::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::LoadOp op,
                                PatternRewriter &rewriter) const override {
    if (TaskletNode task = dyn_cast<TaskletNode>(op->getParentOp())) {
      Location loc = op.getLoc();
      StateNode state = cast<StateNode>(task->getParentOp());
      Type elT = op.getMemRefType().getElementType();

      SmallVector<int64_t> ints;
      SmallVector<bool> shape;

      for (int64_t dim : op.getMemRefType().getShape()) {
        ints.push_back(dim);
        shape.push_back(true);
      }

      MemletType mem = MemletType::get(loc.getContext(), elT, {}, ints, shape);

      // Replace tasklet type
      unsigned argIdx;
      for (BlockArgument a : task.getArguments()) {
        for (Operation *b : a.getUsers()) {
          if (b == op.getOperation()) {
            argIdx = a.getArgNumber();
            break;
          }
        }
      }

      rewriter.eraseOp(op);
      unsigned numArgs = task.getNumArguments();
      task.insertArgument(numArgs, mem, {});
      // task.eraseArgument(argIdx);
      LoadOp load = LoadOp::create(rewriter, loc, elT,
                                   task.getArgument(numArgs), op.indices());
      // state.addOp(*load);

      // Replace all memref.load/store referencing this memref with sdir

      // rewriter.replaceOp(op, {task.getArgument(numArgs)});
      return success();
    }
    return failure();
  }
};

void populateSAMToSDIRConversionPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctxt = patterns.getContext();

  patterns.add<FuncToSDFG>(ctxt);
  patterns.add<TaskletTerminator>(ctxt);
  // patterns.add<ConstantPromotion>(ctxt);
  //  patterns.add<TaskletReordering>(ctxt);
  //  patterns.add<MemrefLoadToSDIR>(ctxt);
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
