// BUG: Remapping of values faulty
#include "SDIR/Conversion/SAMToSDIR/PassDetail.h"
#include "SDIR/Conversion/SAMToSDIR/Passes.h"
#include "SDIR/Dialect/Dialect.h"
#include "SDIR/Utils/Utils.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BlockAndValueMapping.h"

using namespace mlir;
using namespace sdir;
using namespace conversion;

struct SDIRTarget : public ConversionTarget {
  SDIRTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    // Every Op in the SDIR Dialect is legal
    addLegalDialect<SDIRDialect>();
    // Implicit top level module operation is legal
    // if it is empty or only contains a single SDFGNode
    addDynamicallyLegalOp<ModuleOp>([](ModuleOp op) {
      size_t numOps = op.body().getBlocks().front().getOperations().size();
      return numOps == 0 || numOps == 1;
    });
    // All other operations are illegal
    markUnknownOpDynamicallyLegal([](Operation *op) { return false; });
  }
};

class MemrefToMemletConverter : public TypeConverter {
public:
  MemrefToMemletConverter() {
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
              StringAttr::get(mem.getContext(), utils::generateName("s"));
          symbols.push_back(sym);
          shape.push_back(false);
        } else {
          ints.push_back(dim);
          shape.push_back(true);
        }
      }
      return MemletType::get(mem.getContext(), mem.getElementType(), symbols,
                             ints, shape);
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
    for (unsigned i = 0; i < op.getNumArguments(); i++) {
      // NOTE: Hotfix, check if a better solution exists
      MemrefToMemletConverter memo;
      Type nt = memo.convertType(op.getType().getInput(i));
      inputResults.push_back(nt);
    }

    SmallVector<Type> outputResults;
    for (unsigned i = 0; i < op.getNumResults(); i++) {
      // NOTE: Hotfix, check if a better solution exists
      MemrefToMemletConverter memo;
      Type nt = memo.convertType(op.getType().getResult(i));
      outputResults.push_back(nt);
    }

    FunctionType ft = rewriter.getFunctionType(inputResults, outputResults);
    SDFGNode sdfg = SDFGNode::create(rewriter, op.getLoc(), ft, op.sym_name());
    StateNode state = StateNode::create(rewriter, op.getLoc());
    rewriter.createBlock(&state.body());

    for (unsigned i = 0; i < op.getNumArguments(); ++i) {
      op.getArgument(i).replaceAllUsesWith(sdfg.getArgument(i));
    }

    rewriter.updateRootInPlace(sdfg, [&] {
      sdfg.entryAttr(
          SymbolRefAttr::get(op.getLoc().getContext(), state.sym_name()));
    });

    rewriter.inlineRegionBefore(op.body(), state.body(), state.body().end());
    rewriter.eraseOp(op);
    rewriter.mergeBlocks(&state.body().getBlocks().back(),
                         &state.body().getBlocks().front(),
                         sdfg.getArguments());

    // hasValue() is inaccessable
    if (rewriter.convertRegionTypes(&state.body(), *getTypeConverter())
            .getPointer() == nullptr)
      return failure();

    for (unsigned i = 0; i < state.body().getBlocks().front().getNumArguments();
         ++i) {
      rewriter.replaceUsesOfBlockArgument(
          state.body().getBlocks().front().getArgument(i), sdfg.getArgument(i));
    }

    return success();
  }
};

class OpToTasklet : public ConversionPattern {
public:
  OpToTasklet(TypeConverter &converter, MLIRContext *context)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: Check if there is a proper way of doing this
    if (op->getDialect()->getNamespace() == "arith" ||
        op->getDialect()->getNamespace() == "math") {

      if (isa<TaskletNode>(op->getParentOp()))
        return failure(); // Operation already in a tasklet

      FunctionType ft =
          rewriter.getFunctionType(op->getOperandTypes(), op->getResultTypes());
      TaskletNode task = TaskletNode::create(rewriter, op->getLoc(), ft);

      BlockAndValueMapping mapping;
      mapping.map(op->getOperands(), task.getArguments());

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

class EraseTerminators : public RewritePattern {
public:
  EraseTerminators(PatternBenefit benefit, MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {

    if (isa<scf::YieldOp>(op) && !isa<scf::ForOp>(op->getParentOp())) {
      rewriter.eraseOp(op);
      return success();
    }

    if (isa<mlir::ReturnOp>(op) && !isa<FuncOp>(op->getParentOp())) {
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

class SCFForToSDIR : public OpConversionPattern<scf::ForOp> {
public:
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  void getExternalValues(scf::ForOp &root, scf::ForOp &curr,
                         SmallVector<Value> &vals,
                         DenseSet<Value> &valSet) const {
    for (Operation &nested : curr.getRegion().getOps()) {
      for (Value v : nested.getOperands()) {
        if (root.isDefinedOutsideOfLoop(v) && !valSet.contains(v)) {
          vals.push_back(v);
          valSet.insert(v);
        }
      }
      if (scf::ForOp fop = dyn_cast<scf::ForOp>(nested)) {
        getExternalValues(root, fop, vals, valSet);
      }
    }
  }

  bool isNested(scf::ForOp &root, Operation &op) const {
    for (Operation &nested : root.getRegion().getOps()) {
      if (&nested == &op)
        return true;

      if (scf::ForOp fop = dyn_cast<scf::ForOp>(nested)) {
        return isNested(fop, op);
      }
    }
    return false;
  }

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> valsBounds = {
        adaptor.getLowerBound(), adaptor.getUpperBound(), adaptor.getStep()};

    DenseSet<Value> valSet;
    for (Value v : valsBounds)
      valSet.insert(v);

    SmallVector<Value> vals;
    getExternalValues(op, op, vals, valSet);

    unsigned lowerBoundIdx = 0;
    unsigned upperBoundIdx = 1;
    unsigned stepIdx = 2;
    unsigned boundsLen = 3;

    // SDFG
    SmallVector<Type> inputs = {valsBounds[0].getType()};
    if (valsBounds[1].getDefiningOp() == valsBounds[0].getDefiningOp()) {
      upperBoundIdx = lowerBoundIdx;
      boundsLen--;
    } else {
      inputs.push_back(valsBounds[1].getType());
    }

    if (valsBounds[2].getDefiningOp() == valsBounds[1].getDefiningOp()) {
      stepIdx = upperBoundIdx;
      boundsLen--;
    } else if (valsBounds[2].getDefiningOp() == valsBounds[0].getDefiningOp()) {
      stepIdx = lowerBoundIdx;
      boundsLen--;
    } else {
      inputs.push_back(valsBounds[2].getType());
    }

    for (Value v : vals)
      inputs.push_back(v.getType());

    SmallVector<Type> inputResults;
    for (unsigned i = 0; i < inputs.size(); i++) {
      // NOTE: Hotfix, check if a better solution exists
      MemrefToMemletConverter memo;
      Type nt = memo.convertType(inputs[i]);
      inputResults.push_back(nt);
    }

    FunctionType ft = rewriter.getFunctionType(inputResults, {});
    SDFGNode sdfg = SDFGNode::create(rewriter, op.getLoc(), ft);
    AllocSymbolOp::create(rewriter, op.getLoc(), "idx");
    OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();

    // States
    StateNode init = StateNode::create(rewriter, op.getLoc(), "init");
    rewriter.createBlock(&init.body());

    rewriter.updateRootInPlace(sdfg, [&] {
      sdfg.entryAttr(
          SymbolRefAttr::get(op.getLoc().getContext(), init.sym_name()));
    });

    rewriter.restoreInsertionPoint(ip);
    StateNode guard = StateNode::create(rewriter, op.getLoc(), "guard");
    rewriter.createBlock(&guard.body());

    rewriter.restoreInsertionPoint(ip);
    StateNode body = StateNode::create(rewriter, op.getLoc(), "body");

    for (unsigned i = 0; i < vals.size(); ++i) {
      vals[i].replaceUsesWithIf(
          sdfg.getArgument(i + boundsLen),
          [&](OpOperand &opop) { return isNested(op, *opop.getOwner()); });
    }

    valsBounds[0].replaceUsesWithIf(
        sdfg.getArgument(lowerBoundIdx),
        [&](OpOperand &opop) { return isNested(op, *opop.getOwner()); });

    valsBounds[1].replaceUsesWithIf(
        sdfg.getArgument(upperBoundIdx),
        [&](OpOperand &opop) { return isNested(op, *opop.getOwner()); });

    valsBounds[2].replaceUsesWithIf(
        sdfg.getArgument(stepIdx),
        [&](OpOperand &opop) { return isNested(op, *opop.getOwner()); });

    rewriter.inlineRegionBefore(op.getLoopBody(), body.body(),
                                body.body().begin());
    rewriter.eraseOp(op);

    rewriter.setInsertionPointToStart(&body.body().getBlocks().front());

    SymOp symop =
        SymOp::create(rewriter, op.getLoc(), rewriter.getIndexType(), "idx");

    if (rewriter.convertRegionTypes(&body.body(), *getTypeConverter())
            .getPointer() == nullptr)
      return failure();

    rewriter.replaceUsesOfBlockArgument(
        body.body().getBlocks().front().getArgument(0), symop);

    // TODO: Get rid of block arguments
    // body.body().getBlocks().front().eraseArgument(0);
    //   NOTE: Infinite loop
    //   rewriter.createBlock(&body.body());
    //   rewriter.mergeBlocks(&body.body().getBlocks().front(),
    //                       &body.body().getBlocks().back(), {symop});

    rewriter.restoreInsertionPoint(ip);
    StateNode exit = StateNode::create(rewriter, op.getLoc(), "exit");
    rewriter.createBlock(&exit.body());

    // Edges
    rewriter.restoreInsertionPoint(ip);

    ArrayAttr emptyArr;
    StringAttr emptyStr;

    ArrayAttr initArr = rewriter.getStrArrayAttr({"idx: ref"});
    EdgeOp::create(rewriter, op.getLoc(), init, guard, initArr, emptyStr,
                   sdfg.getArgument(lowerBoundIdx));

    StringAttr guardStr = rewriter.getStringAttr("idx < ref");
    EdgeOp::create(rewriter, op.getLoc(), guard, body, emptyArr, guardStr,
                   sdfg.getArgument(upperBoundIdx));

    ArrayAttr bodyArr = rewriter.getStrArrayAttr({"idx: idx + ref"});
    EdgeOp::create(rewriter, op.getLoc(), body, guard, bodyArr, emptyStr,
                   sdfg.getArgument(stepIdx));

    StringAttr exitStr = rewriter.getStringAttr("not(idx < ref)");
    EdgeOp::create(rewriter, op.getLoc(), guard, exit, emptyArr, exitStr,
                   sdfg.getArgument(upperBoundIdx));

    rewriter.setInsertionPointAfter(sdfg);
    SmallVector<Value> callVals = adaptor.getOperands();
    SmallVector<Value> callValsReduced = {callVals[lowerBoundIdx]};
    if (upperBoundIdx != lowerBoundIdx)
      callValsReduced.push_back(callVals[upperBoundIdx]);

    if (stepIdx != upperBoundIdx && stepIdx != lowerBoundIdx)
      callValsReduced.push_back(callVals[stepIdx]);

    callValsReduced.append(vals);
    sdir::CallOp::create(rewriter, op.getLoc(), sdfg, callValsReduced);

    return success();
  }
};

void populateSAMToSDIRConversionPatterns(RewritePatternSet &patterns,
                                         TypeConverter &converter) {
  MLIRContext *ctxt = patterns.getContext();
  patterns.add<FuncToSDFG>(converter, ctxt);
  patterns.add<OpToTasklet>(converter, ctxt);
  patterns.add<EraseTerminators>(1, ctxt);
  patterns.add<MemrefLoadToSDIR>(converter, ctxt);
  patterns.add<MemrefStoreToSDIR>(converter, ctxt);
  patterns.add<SCFForToSDIR>(converter, ctxt);
}

namespace {
struct SAMToSDIRPass : public SAMToSDIRPassBase<SAMToSDIRPass> {
  void runOnOperation() override;
};
} // namespace

void SAMToSDIRPass::runOnOperation() {
  ModuleOp module = getOperation();

  RewritePatternSet patterns(&getContext());
  MemrefToMemletConverter converter;
  populateSAMToSDIRConversionPatterns(patterns, converter);

  SDIRTarget target(getContext());
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> conversion::createSAMToSDIRPass() {
  return std::make_unique<SAMToSDIRPass>();
}
