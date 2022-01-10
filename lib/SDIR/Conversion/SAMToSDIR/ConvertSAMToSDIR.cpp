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

    /*SmallVector<AllocOp> allocs;
    SmallVector<GetAccessOp> access;
    SmallVector<LoadOp> loads;
    TaskletNode zeroTask = TaskletNode::create(loc, 0);
    sdir::CallOp zeroCall = sdir::CallOp::create(loc, zeroTask, {});

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
      }

      ArrayType art = ArrayType::get(loc->getContext(), t, {}, ints, shape);
      AllocOp alloc = AllocOp::create(loc, art);
      allocs.push_back(alloc);

      GetAccessOp acc = GetAccessOp::create(loc, alloc);
      access.push_back(acc);

      SmallVector<Value> idxV;
      for (size_t i = 0; i < shape.size(); ++i) {
        idxV.push_back(zeroCall.getResult(0));
      }
      ValueRange vr = ValueRange(idxV);
      LoadOp load = LoadOp::create(loc, acc, vr);
      loads.push_back(load);
    }*/

    SDFGNode sdfg = SDFGNode::create(loc, op.getType());
    StateNode state = StateNode::create(loc);
    sdfg.addState(state, /*isEntry=*/true);

    TaskletNode task = TaskletNode::create(loc, op.getType());
    task.body().takeBody(op.body());
    state.addOp(*task);

    /*for (GetAccessOp acc : access) {
      state.addOp(*acc);
    }

    state.addOp(*zeroTask);
    state.addOp(*zeroCall);

    for (LoadOp load : loads) {
      state.addOp(*load);
    }

    SmallVector<Value> loadV;
    for (LoadOp load : loads) {
      loadV.push_back(load);
    }
    ValueRange vr = ValueRange(loadV);*/
    sdir::CallOp call = sdir::CallOp::create(loc, task, sdfg.getArguments());
    state.addOp(*call);

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

class ConstantPromotion : public OpRewritePattern<arith::ConstantOp> {
public:
  using OpRewritePattern<arith::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ConstantOp op,
                                PatternRewriter &rewriter) const override {

    if (TaskletNode task = dyn_cast<TaskletNode>(op->getParentOp())) {
      StateNode state = cast<StateNode>(task->getParentOp());
      TaskletNode taskC = TaskletNode::create(op.getLoc(), op);
      sdir::CallOp callC = sdir::CallOp::create(op.getLoc(), taskC, {});
      state.addOp(*callC, /*toFront=*/true);
      state.addOp(*taskC, /*toFront=*/true);

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
    op.emitError("here");
    if (TaskletNode task = dyn_cast<TaskletNode>(prev))
      return failure();

    rewriter.eraseOp(op);
    rewriter.eraseOp(prev);

    rewriter.insert(prev);
    rewriter.insert(op);

    return success();
  }
};

void populateSAMToSDIRConversionPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
    FuncToSDFG, 
    TaskletTerminator,
    ConstantPromotion, 
    TaskletReordering
  >(patterns.getContext());
  // clang-format on
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
