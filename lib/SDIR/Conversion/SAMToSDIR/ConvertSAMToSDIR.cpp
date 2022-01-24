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

//===----------------------------------------------------------------------===//
// Target & Type Converter
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

SmallVector<Value> createLoads(PatternRewriter &rewriter, Location loc,
                               ArrayRef<Value> vals) {
  SmallVector<Value> loadedOps;
  for (Value operand : vals) {
    if (operand.getDefiningOp() != nullptr &&
        isa<LoadOp>(operand.getDefiningOp())) {
      LoadOp load = cast<LoadOp>(operand.getDefiningOp());
      GetAccessOp gao = cast<GetAccessOp>(load.arr().getDefiningOp());
      AllocTransientOp alloc =
          cast<AllocTransientOp>(gao.arr().getDefiningOp());

      GetAccessOp acc =
          GetAccessOp::create(rewriter, loc, alloc.getType().toMemlet(), alloc);
      LoadOp loadNew = LoadOp::create(rewriter, loc, acc, ValueRange());

      loadedOps.push_back(loadNew);
    } else if (operand.getDefiningOp() != nullptr &&
               isa<SymOp>(operand.getDefiningOp())) {
      SymOp sym = cast<SymOp>(operand.getDefiningOp());
      SymOp symNew = SymOp::create(rewriter, loc, sym.getType(), sym.expr());
      loadedOps.push_back(symNew);
    } else {
      loadedOps.push_back(operand);
    }
  }
  return loadedOps;
}

Value createLoad(PatternRewriter &rewriter, Location loc, Value val) {
  SmallVector<Value> loadedOps = {val};
  return createLoads(rewriter, loc, loadedOps)[0];
}

SDFGNode getTopSDFG(Operation *op) {
  Operation *parent = op->getParentOp();

  if (isa<SDFGNode>(parent))
    return cast<SDFGNode>(parent);

  return getTopSDFG(parent);
}

void linkToLastState(PatternRewriter &rewriter, Location loc,
                     StateNode &state) {
  SDFGNode sdfg = getTopSDFG(state);
  OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToEnd(&sdfg.body().getBlocks().front());

  StateNode prev;

  for (StateNode sn : sdfg.body().getOps<StateNode>()) {
    if (sn == state) {
      EdgeOp::create(rewriter, loc, prev, state);
      break;
    }
    prev = sn;
  }
  rewriter.restoreInsertionPoint(ip);
}

void linkToNextState(PatternRewriter &rewriter, Location loc,
                     StateNode &state) {
  SDFGNode sdfg = getTopSDFG(state);
  OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToEnd(&sdfg.body().getBlocks().front());

  bool visitedState = false;
  for (StateNode sn : sdfg.body().getOps<StateNode>()) {
    if (visitedState) {
      EdgeOp::create(rewriter, loc, state, sn);
      break;
    }

    if (sn == state)
      visitedState = true;
  }
  rewriter.restoreInsertionPoint(ip);
}

bool markedToLink(Operation &op) {
  return op.hasAttr("linkToNext") &&
         op.getAttr("linkToNext").cast<BoolAttr>().getValue();
}
//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

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
    StateNode state = StateNode::create(rewriter, op.getLoc(), "init");

    rewriter.updateRootInPlace(sdfg, [&] {
      sdfg.entryAttr(
          SymbolRefAttr::get(op.getLoc().getContext(), state.sym_name()));
    });

    for (unsigned i = 0; i < op.getNumArguments(); ++i) {
      op.getArgument(i).replaceAllUsesWith(sdfg.getArgument(i));
    }

    rewriter.inlineRegionBefore(op.body(), sdfg.body(), sdfg.body().end());
    rewriter.eraseOp(op);
    rewriter.mergeBlocks(&sdfg.body().getBlocks().back(),
                         &sdfg.body().getBlocks().front(), sdfg.getArguments());

    // hasValue() is inaccessable
    if (rewriter.convertRegionTypes(&sdfg.body(), *getTypeConverter())
            .getPointer() == nullptr)
      return failure();

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

      StateNode state = StateNode::create(rewriter, op->getLoc());

      // NOTE: Hotfix, check if a better solution exists
      MemrefToMemletConverter memo;
      Type nt = memo.convertType(op->getResultTypes()[0]);
      if (MemletType mem = nt.dyn_cast<MemletType>())
        nt = mem.toArray();
      else
        nt = ArrayType::get(op->getLoc().getContext(), nt, {}, {}, {});

      SDFGNode sdfg = getTopSDFG(state);
      OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointToStart(&sdfg.body().getBlocks().front());
      AllocTransientOp alloc =
          AllocTransientOp::create(rewriter, op->getLoc(), nt, "_tmp");

      rewriter.restoreInsertionPoint(ip);

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

      SmallVector<Value> loadedOps =
          createLoads(rewriter, op->getLoc(), operands);

      sdir::CallOp call =
          sdir::CallOp::create(rewriter, op->getLoc(), task, loadedOps);

      GetAccessOp access = GetAccessOp::create(
          rewriter, op->getLoc(), nt.cast<ArrayType>().toMemlet(), alloc);
      StoreOp::create(rewriter, op->getLoc(), call.getResult(0), access,
                      ValueRange());

      LoadOp load =
          LoadOp::create(rewriter, op->getLoc(), access, ValueRange());

      rewriter.replaceOp(op, {load});

      linkToLastState(rewriter, op->getLoc(), state);
      if (markedToLink(*op))
        linkToNextState(rewriter, op->getLoc(), state);

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
      StateNode state = StateNode::create(rewriter, op->getLoc(), "yield");
      linkToLastState(rewriter, op->getLoc(), state);
      if (markedToLink(*op))
        linkToNextState(rewriter, op->getLoc(), state);
      rewriter.eraseOp(op);
      return success();
    }

    if (isa<mlir::ReturnOp>(op) && !isa<FuncOp>(op->getParentOp())) {
      StateNode state = StateNode::create(rewriter, op->getLoc(), "return");
      linkToLastState(rewriter, op->getLoc(), state);
      if (markedToLink(*op))
        linkToNextState(rewriter, op->getLoc(), state);
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
    Type type = getTypeConverter()->convertType(op.getType());
    Type arrT = type;

    if (MemletType mem = arrT.dyn_cast<MemletType>())
      arrT = mem.toArray();
    else
      arrT = ArrayType::get(op->getLoc().getContext(), arrT, {}, {}, {});

    SDFGNode sdfg = getTopSDFG(op);
    OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(&sdfg.body().getBlocks().front());
    AllocTransientOp alloc =
        AllocTransientOp::create(rewriter, op->getLoc(), arrT, "_tmp");

    rewriter.restoreInsertionPoint(ip);

    StateNode state = StateNode::create(rewriter, op->getLoc());

    Value memref = createLoad(rewriter, op.getLoc(), adaptor.memref());

    SmallVector<Value> indices = adaptor.indices();
    indices = createLoads(rewriter, op.getLoc(), indices);

    LoadOp load = LoadOp::create(rewriter, op.getLoc(), type, memref, indices);

    GetAccessOp access = GetAccessOp::create(
        rewriter, op->getLoc(), arrT.cast<ArrayType>().toMemlet(), alloc);
    StoreOp::create(rewriter, op->getLoc(), load, access, ValueRange());

    LoadOp newLoad =
        LoadOp::create(rewriter, op->getLoc(), access, ValueRange());

    linkToLastState(rewriter, op->getLoc(), state);
    if (markedToLink(*op))
      linkToNextState(rewriter, op->getLoc(), state);

    rewriter.replaceOp(op, {newLoad});
    return success();
  }
};

class MemrefStoreToSDIR : public OpConversionPattern<memref::StoreOp> {
public:
  using OpConversionPattern<memref::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    StateNode state = StateNode::create(rewriter, op->getLoc());

    Value val = createLoad(rewriter, op.getLoc(), adaptor.value());
    Value memref = createLoad(rewriter, op.getLoc(), adaptor.memref());

    SmallVector<Value> indices = adaptor.indices();
    indices = createLoads(rewriter, op.getLoc(), indices);

    StoreOp::create(rewriter, op.getLoc(), val, memref, indices);

    linkToLastState(rewriter, op->getLoc(), state);
    if (markedToLink(*op))
      linkToNextState(rewriter, op->getLoc(), state);

    rewriter.eraseOp(op);
    return success();
  }
};

class SCFForToSDIR : public OpConversionPattern<scf::ForOp> {
public:
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  Value mappedValue(Value val) const {
    if (val.getDefiningOp() != nullptr && isa<LoadOp>(val.getDefiningOp())) {
      LoadOp load = cast<LoadOp>(val.getDefiningOp());
      GetAccessOp gao = cast<GetAccessOp>(load.arr().getDefiningOp());
      return gao.arr();
    }

    return val;
  }

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::string idxName = utils::generateName("loopIdx");

    SDFGNode sdfg = getTopSDFG(op);
    OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(&sdfg.body().getBlocks().front());
    AllocSymbolOp::create(rewriter, op.getLoc(), idxName);

    rewriter.restoreInsertionPoint(ip);

    StateNode init = StateNode::create(rewriter, op.getLoc(), "init");
    SymOp idxSym = SymOp::create(rewriter, op.getLoc(),
                                 op.getInductionVar().getType(), idxName);
    op.getInductionVar().replaceAllUsesWith(idxSym);

    linkToLastState(rewriter, op.getLoc(), init);

    rewriter.setInsertionPointAfter(init);
    StateNode guard = StateNode::create(rewriter, op.getLoc(), "guard");

    rewriter.setInsertionPointAfter(guard);
    StateNode body = StateNode::create(rewriter, op.getLoc(), "body");

    rewriter.setInsertionPointAfter(body);

    SmallVector<Operation *> copies;
    for (Operation &nop : op.getLoopBody().getOps())
      copies.push_back(&nop);

    copies.back()->setAttr("linkToNext", rewriter.getBoolAttr(true));
    if (op.moveOutOfLoop(copies).failed())
      return failure();

    StateNode returnState =
        StateNode::create(rewriter, op.getLoc(), "loopReturn");

    rewriter.setInsertionPointAfter(op);
    StateNode exitState = StateNode::create(rewriter, op.getLoc(), "exit");

    rewriter.setInsertionPointAfter(exitState);
    ArrayAttr emptyArr;
    StringAttr emptyStr;
    ArrayAttr initArr = rewriter.getStrArrayAttr({idxName + ": ref"});
    EdgeOp::create(rewriter, op.getLoc(), init, guard, initArr, emptyStr,
                   mappedValue(adaptor.getLowerBound()));

    StringAttr guardStr = rewriter.getStringAttr(idxName + " < ref");
    EdgeOp::create(rewriter, op.getLoc(), guard, body, emptyArr, guardStr,
                   mappedValue(adaptor.getUpperBound()));

    ArrayAttr returnArr =
        rewriter.getStrArrayAttr({idxName + ": " + idxName + " + ref"});
    EdgeOp::create(rewriter, op.getLoc(), returnState, guard, returnArr,
                   emptyStr, mappedValue(adaptor.getStep()));

    StringAttr exitStr = rewriter.getStringAttr("not(" + idxName + " < ref)");
    EdgeOp::create(rewriter, op.getLoc(), guard, exitState, emptyArr, exitStr,
                   mappedValue(adaptor.getUpperBound()));

    if (markedToLink(*op))
      linkToNextState(rewriter, op->getLoc(), exitState);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

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

  SDIRTarget target(getContext());
  MemrefToMemletConverter converter;

  RewritePatternSet patterns(&getContext());
  populateSAMToSDIRConversionPatterns(patterns, converter);

  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> conversion::createSAMToSDIRPass() {
  return std::make_unique<SAMToSDIRPass>();
}
