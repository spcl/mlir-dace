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
                         SetVector<Value> &valSet) const {
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
    // if (!op.getLoopBody().getOps<scf::ForOp>().empty())
    //   return failure();

    SmallVector<Value> vals;
    SetVector<Value> valSet;

    getExternalValues(op, op, vals, valSet);

    // SDFG
    SmallVector<Type> inputs = {
        rewriter.getIndexType(), // lower bound
        rewriter.getIndexType(), // upper bound
        rewriter.getIndexType()  // step bound
    };
    for (Value v : vals)
      inputs.push_back(v.getType());

    SmallVector<Type> inputResults;
    if (getTypeConverter()->convertTypes(inputs, inputResults).failed())
      return failure();

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
      vals[i].replaceUsesWithIf(sdfg.getArgument(i + 3), [&](OpOperand &opop) {
        return isNested(op, *opop.getOwner());
      });
    }

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

    // body.body().getBlocks().front().eraseArgument(0);
    //  NOTE: Infinite loop
    //  rewriter.createBlock(&body.body());
    //  rewriter.mergeBlocks(&body.body().getBlocks().front(),
    //                      &body.body().getBlocks().back(), {symop});

    rewriter.restoreInsertionPoint(ip);
    StateNode exit = StateNode::create(rewriter, op.getLoc(), "exit");
    rewriter.createBlock(&exit.body());

    // Edges
    rewriter.restoreInsertionPoint(ip);

    ArrayAttr emptyArr;
    StringAttr emptyStr;

    ArrayAttr initArr = rewriter.getStrArrayAttr({"idx: ref"});
    EdgeOp::create(rewriter, op.getLoc(), init, guard, initArr, emptyStr,
                   sdfg.getArgument(0));

    StringAttr guardStr = rewriter.getStringAttr("idx < ref");
    EdgeOp::create(rewriter, op.getLoc(), guard, body, emptyArr, guardStr,
                   sdfg.getArgument(1));

    ArrayAttr bodyArr = rewriter.getStrArrayAttr({"idx: idx + ref"});
    EdgeOp::create(rewriter, op.getLoc(), body, guard, bodyArr, emptyStr,
                   sdfg.getArgument(2));

    StringAttr exitStr = rewriter.getStringAttr("not(idx < ref)");
    EdgeOp::create(rewriter, op.getLoc(), guard, exit, emptyArr, exitStr,
                   sdfg.getArgument(1));

    rewriter.setInsertionPointAfter(sdfg);
    SmallVector<Value> callVals = adaptor.getOperands();
    callVals.append(vals);
    sdir::CallOp::create(rewriter, op.getLoc(), sdfg, callVals);

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
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> conversion::createSAMToSDIRPass() {
  return std::make_unique<SAMToSDIRPass>();
}
