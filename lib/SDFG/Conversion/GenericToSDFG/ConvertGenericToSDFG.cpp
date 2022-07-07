#include "SDFG/Conversion/GenericToSDFG/PassDetail.h"
#include "SDFG/Conversion/GenericToSDFG/Passes.h"
#include "SDFG/Dialect/Dialect.h"
#include "SDFG/Utils/Utils.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BlockAndValueMapping.h"

using namespace mlir;
using namespace sdfg;
using namespace conversion;

//===----------------------------------------------------------------------===//
// Target & Type Converter
//===----------------------------------------------------------------------===//

struct SDFGTarget : public ConversionTarget {
  SDFGTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    // Every Op in the SDFG Dialect is legal
    addLegalDialect<SDFGDialect>();
    // Implicit top level module operation is legal
    // if it is empty or only contains a single SDFGNode
    addLegalOp<ModuleOp>();
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
      SizedType sized = SizedType::get(mem.getContext(), mem.getElementType(),
                                       symbols, ints, shape);

      return ArrayType::get(mem.getContext(), sized);
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
      AllocOp alloc = cast<AllocOp>(load.arr().getDefiningOp());
      LoadOp loadNew = LoadOp::create(rewriter, loc, alloc, ValueRange());

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

Value getTransientValue(Value val) {
  if (val.getDefiningOp() != nullptr && isa<LoadOp>(val.getDefiningOp())) {
    LoadOp load = cast<LoadOp>(val.getDefiningOp());
    AllocOp alloc = cast<AllocOp>(load.arr().getDefiningOp());
    return alloc;
  }

  return val;
}

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

class FuncToSDFG : public OpConversionPattern<func::FuncOp> {
public:
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (op.isDeclaration()) {
      rewriter.eraseOp(op);
      return success();
    }

    SmallVector<Type> args;
    for (unsigned i = 0; i < op.getNumArguments(); i++) {
      MemrefToMemletConverter memo;
      Type nt = memo.convertType(op.getArgumentTypes()[i]);
      args.push_back(nt);
    }

    for (unsigned i = 0; i < op.getNumResults(); i++) {
      MemrefToMemletConverter memo;
      Type nt = memo.convertType(op.getResultTypes()[i]);

      if (!nt.isa<ArrayType>()) {
        SizedType sized = SizedType::get(nt.getContext(), nt, {}, {}, {});
        nt = ArrayType::get(nt.getContext(), sized);
      }

      args.push_back(nt);
    }

    SDFGNode sdfg =
        SDFGNode::create(rewriter, op.getLoc(), op.getNumArguments(), args);
    StateNode state = StateNode::create(rewriter, op.getLoc(), "init");

    rewriter.updateRootInPlace(sdfg, [&] {
      sdfg.entryAttr(
          SymbolRefAttr::get(op.getLoc().getContext(), state.sym_name()));
    });

    for (unsigned i = 0; i < op.getNumArguments(); ++i) {
      op.getArgument(i).replaceAllUsesWith(sdfg.body().getArgument(i));
    }

    rewriter.inlineRegionBefore(op.getBody(), sdfg.body(), sdfg.body().end());
    rewriter.eraseOp(op);

    rewriter.mergeBlocks(
        &sdfg.body().getBlocks().back(), &sdfg.body().getBlocks().front(),
        sdfg.body().getArguments().take_front(sdfg.num_args()));

    // hasValue() is inaccessable
    if (rewriter.convertRegionTypes(&sdfg.body(), *getTypeConverter())
            .getPointer() == nullptr)
      return failure();

    return success();
  }
};

class CallToSDFG : public OpConversionPattern<func::CallOp> {
public:
  using OpConversionPattern<func::CallOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: Should be nested SDFGs or C++ tasklets
    std::string callee = op.getCallee().str();

    SDFGNode sdfg = getTopSDFG(op);
    OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(&sdfg.body().getBlocks().front());

    std::vector<AllocOp> allocs = {};

    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      MemrefToMemletConverter memo;
      Type newType = memo.convertType(op.getResultTypes()[i]);
      SizedType sized =
          SizedType::get(op->getLoc().getContext(), newType, {}, {}, {});
      newType = ArrayType::get(op->getLoc().getContext(), sized);

      AllocOp alloc = AllocOp::create(rewriter, op->getLoc(), newType,
                                      "_" + callee + std::to_string(i) + "_tmp",
                                      /*transient=*/true);
      allocs.push_back(alloc);
    }

    rewriter.restoreInsertionPoint(ip);

    StateNode state = StateNode::create(rewriter, op.getLoc(), "_" + callee);

    std::vector<Value> args = {};
    for (Value v : adaptor.getOperands())
      args.push_back(v);

    SmallVector<Value> loadedArgs = createLoads(rewriter, op.getLoc(), args);

    LibCallOp libcall = LibCallOp::create(
        rewriter, op.getLoc(), op.getResultTypes(), callee, loadedArgs);

    std::vector<Value> loads = {};
    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      StoreOp::create(rewriter, op->getLoc(), libcall.getResult(i), allocs[i],
                      ValueRange());

      LoadOp load =
          LoadOp::create(rewriter, op->getLoc(), allocs[i], ValueRange());
      loads.push_back(load.getResult());
    }

    rewriter.replaceOp(op, loads);

    linkToLastState(rewriter, op->getLoc(), state);
    if (markedToLink(*op))
      linkToNextState(rewriter, op->getLoc(), state);

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

      std::string name = utils::operationToString(*op);
      StateNode state = StateNode::create(rewriter, op->getLoc(), name);

      MemrefToMemletConverter memo;
      Type nt = memo.convertType(op->getResultTypes()[0]);
      SizedType sized =
          SizedType::get(op->getLoc().getContext(), nt, {}, {}, {});
      nt = ArrayType::get(op->getLoc().getContext(), sized);

      SDFGNode sdfg = getTopSDFG(state);
      OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointToStart(&sdfg.body().getBlocks().front());
      AllocOp alloc =
          AllocOp::create(rewriter, op->getLoc(), nt, "_" + name + "_tmp",
                          /*transient=*/true);

      rewriter.restoreInsertionPoint(ip);

      SmallVector<Value> loadedOps =
          createLoads(rewriter, op->getLoc(), operands);

      TaskletNode task = TaskletNode::create(rewriter, op->getLoc(), loadedOps,
                                             op->getResultTypes());

      BlockAndValueMapping mapping;
      mapping.map(op->getOperands(), task.body().getArguments());

      Operation *opClone = op->clone(mapping);
      rewriter.updateRootInPlace(
          task, [&] { task.body().getBlocks().front().push_front(opClone); });

      sdfg::ReturnOp::create(rewriter, opClone->getLoc(),
                             opClone->getResults());

      rewriter.setInsertionPointAfter(task);

      StoreOp::create(rewriter, op->getLoc(), task.getResult(0), alloc,
                      ValueRange());

      LoadOp load = LoadOp::create(rewriter, op->getLoc(), alloc, ValueRange());

      rewriter.replaceOp(op, {load});

      linkToLastState(rewriter, op->getLoc(), state);
      if (markedToLink(*op))
        linkToNextState(rewriter, op->getLoc(), state);

      return success();
    }

    return failure();
  }
};

class EraseTerminators : public ConversionPattern {
public:
  EraseTerminators(TypeConverter &converter, MLIRContext *context)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    if (isa<scf::YieldOp>(op) && !isa<scf::ForOp>(op->getParentOp())) {
      StateNode state = StateNode::create(rewriter, op->getLoc(), "yield");
      linkToLastState(rewriter, op->getLoc(), state);
      if (markedToLink(*op))
        linkToNextState(rewriter, op->getLoc(), state);
      rewriter.eraseOp(op);
      return success();
    }

    if (isa<func::ReturnOp>(op) && !isa<func::FuncOp>(op->getParentOp())) {
      StateNode state = StateNode::create(rewriter, op->getLoc(), "return");
      SDFGNode sdfg = getTopSDFG(state);

      SmallVector<Value> loadedOps =
          createLoads(rewriter, op->getLoc(), operands);

      for (unsigned i = 0; i < loadedOps.size(); ++i) {
        StoreOp::create(rewriter, op->getLoc(), loadedOps[i],
                        sdfg.body().getArgument(sdfg.num_args() + i),
                        ValueRange());
      }

      linkToLastState(rewriter, op->getLoc(), state);
      if (markedToLink(*op))
        linkToNextState(rewriter, op->getLoc(), state);
      rewriter.eraseOp(op);
      return success();
    }

    return failure();
  }
};

class MemrefLoadToSDFG : public OpConversionPattern<memref::LoadOp> {
public:
  using OpConversionPattern<memref::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type type = getTypeConverter()->convertType(op.getType());

    SizedType sized =
        SizedType::get(op->getLoc().getContext(), type, {}, {}, {});
    Type arrT = ArrayType::get(op->getLoc().getContext(), sized);

    SDFGNode sdfg = getTopSDFG(op);
    OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(&sdfg.body().getBlocks().front());
    AllocOp alloc = AllocOp::create(rewriter, op->getLoc(), arrT, "_load_tmp",
                                    /*transient=*/true);

    rewriter.restoreInsertionPoint(ip);
    StateNode state = StateNode::create(rewriter, op->getLoc(), "load");
    Value memref = createLoad(rewriter, op.getLoc(), adaptor.memref());

    SmallVector<Value> indices = adaptor.indices();
    indices = createLoads(rewriter, op.getLoc(), indices);

    LoadOp load = LoadOp::create(rewriter, op.getLoc(), type, memref, indices);
    StoreOp::create(rewriter, op->getLoc(), load, alloc, ValueRange());

    LoadOp newLoad =
        LoadOp::create(rewriter, op->getLoc(), alloc, ValueRange());

    linkToLastState(rewriter, op->getLoc(), state);
    if (markedToLink(*op))
      linkToNextState(rewriter, op->getLoc(), state);

    rewriter.replaceOp(op, {newLoad});
    return success();
  }
};

class MemrefStoreToSDFG : public OpConversionPattern<memref::StoreOp> {
public:
  using OpConversionPattern<memref::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StateNode state = StateNode::create(rewriter, op->getLoc(), "store");

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

class MemrefGlobalToSDFG : public OpConversionPattern<memref::GlobalOp> {
public:
  using OpConversionPattern<memref::GlobalOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::GlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

class MemrefGetGlobalToSDFG : public OpConversionPattern<memref::GetGlobalOp> {
public:
  using OpConversionPattern<memref::GetGlobalOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::GetGlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SDFGNode sdfg = getTopSDFG(op);

    OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(&sdfg.body().getBlocks().front());

    Type type = getTypeConverter()->convertType(op.getType());
    AllocOp alloc = AllocOp::create(rewriter, op->getLoc(), type,
                                    "_" + adaptor.name().str(),
                                    /*transient=*/false);

    // TODO: Replace all memref.get_global using the same global array

    rewriter.restoreInsertionPoint(ip);
    rewriter.replaceOp(op, {alloc});
    return success();
  }
};

class MemrefAllocToSDFG : public OpConversionPattern<memref::AllocOp> {
public:
  using OpConversionPattern<memref::AllocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SDFGNode sdfg = getTopSDFG(op);

    OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(&sdfg.body().getBlocks().front());

    Type type = getTypeConverter()->convertType(op.getType());
    AllocOp alloc = AllocOp::create(rewriter, op->getLoc(), type, "_alloc_tmp",
                                    /*transient=*/true);

    rewriter.restoreInsertionPoint(ip);
    StateNode init = StateNode::create(rewriter, op.getLoc(), "alloc_init");
    linkToLastState(rewriter, op.getLoc(), init);

    StateNode lastState = init;
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      rewriter.setInsertionPointAfter(lastState);
      StateNode alloc_param =
          StateNode::create(rewriter, op.getLoc(), "alloc_param");

      rewriter.setInsertionPointToEnd(&sdfg.body().getBlocks().front());

      std::string sym = utils::getSizedType(type).getSymbols()[i].str();
      ArrayAttr initArr = rewriter.getStrArrayAttr({sym + ": ref"});
      StringAttr trueCondition = rewriter.getStringAttr("1");

      EdgeOp::create(rewriter, op.getLoc(), lastState, alloc_param, initArr,
                     trueCondition,
                     getTransientValue(adaptor.getOperands()[i]));

      lastState = alloc_param;
    }

    if (markedToLink(*op))
      linkToNextState(rewriter, op->getLoc(), lastState);

    rewriter.replaceOp(op, {alloc});
    return success();
  }
};

class SCFForToSDFG : public OpConversionPattern<scf::ForOp> {
public:
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::string idxName = utils::generateName("for_idx");

    SDFGNode sdfg = getTopSDFG(op);
    OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(&sdfg.body().getBlocks().front());
    AllocSymbolOp::create(rewriter, op.getLoc(), idxName);

    rewriter.restoreInsertionPoint(ip);

    StateNode init = StateNode::create(rewriter, op.getLoc(), "for_init");
    SymOp idxSym = SymOp::create(rewriter, op.getLoc(),
                                 op.getInductionVar().getType(), idxName);
    op.getInductionVar().replaceAllUsesWith(idxSym);

    linkToLastState(rewriter, op.getLoc(), init);

    rewriter.setInsertionPointAfter(init);
    StateNode guard = StateNode::create(rewriter, op.getLoc(), "for_guard");

    rewriter.setInsertionPointAfter(guard);
    StateNode body = StateNode::create(rewriter, op.getLoc(), "for_body");

    rewriter.setInsertionPointAfter(body);

    SmallVector<Operation *> copies;
    for (Operation &nop : op.getLoopBody().getOps())
      copies.push_back(&nop);

    copies.back()->setAttr("linkToNext", rewriter.getBoolAttr(true));

    for (Operation *oper : copies)
      op.moveOutOfLoop(oper);

    StateNode returnState =
        StateNode::create(rewriter, op.getLoc(), "for_return");

    rewriter.setInsertionPointAfter(op);
    StateNode exitState = StateNode::create(rewriter, op.getLoc(), "for_exit");

    rewriter.setInsertionPointToEnd(&sdfg.body().getBlocks().front());
    ArrayAttr emptyArr = rewriter.getStrArrayAttr({});
    StringAttr emptyStr = rewriter.getStringAttr("1");
    ArrayAttr initArr = rewriter.getStrArrayAttr({idxName + ": ref"});

    EdgeOp::create(rewriter, op.getLoc(), init, guard, initArr, emptyStr,
                   getTransientValue(adaptor.getLowerBound()));

    StringAttr guardStr = rewriter.getStringAttr(idxName + " < ref");
    EdgeOp::create(rewriter, op.getLoc(), guard, body, emptyArr, guardStr,
                   getTransientValue(adaptor.getUpperBound()));

    ArrayAttr returnArr =
        rewriter.getStrArrayAttr({idxName + ": " + idxName + " + ref"});
    EdgeOp::create(rewriter, op.getLoc(), returnState, guard, returnArr,
                   emptyStr, getTransientValue(adaptor.getStep()));

    StringAttr exitStr = rewriter.getStringAttr("not(" + idxName + " < ref)");
    EdgeOp::create(rewriter, op.getLoc(), guard, exitState, emptyArr, exitStr,
                   getTransientValue(adaptor.getUpperBound()));

    if (markedToLink(*op))
      linkToNextState(rewriter, op->getLoc(), exitState);

    rewriter.eraseOp(op);
    return success();
  }
};

class SCFIfToSDFG : public OpConversionPattern<scf::IfOp> {
public:
  using OpConversionPattern<scf::IfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::string condName = utils::generateName("if_cond");

    SDFGNode sdfg = getTopSDFG(op);
    OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(&sdfg.body().getBlocks().front());
    AllocSymbolOp::create(rewriter, op.getLoc(), condName);

    rewriter.restoreInsertionPoint(ip);
    StateNode init = StateNode::create(rewriter, op.getLoc(), "if_init");
    linkToLastState(rewriter, op.getLoc(), init);

    rewriter.setInsertionPointAfter(init);
    StateNode guard = StateNode::create(rewriter, op.getLoc(), "if_guard");

    rewriter.setInsertionPointAfter(guard);
    StateNode ifBranch = StateNode::create(rewriter, op.getLoc(), "if_then");

    rewriter.setInsertionPointAfter(ifBranch);

    bool hasThenBlock = op.thenBlock();

    if (hasThenBlock) {
      op.thenBlock()->back().setAttr("linkToNext", rewriter.getBoolAttr(true));

      Block *cont = rewriter.splitBlock(rewriter.getBlock(),
                                        rewriter.getInsertionPoint());

      rewriter.mergeBlocks(op.thenBlock(), rewriter.getBlock(), {});
      rewriter.mergeBlocks(cont, rewriter.getBlock(), {});
    }

    StateNode jump = StateNode::create(rewriter, op.getLoc(), "if_jump");

    if (!hasThenBlock) {
      linkToLastState(rewriter, op.getLoc(), jump);
    }

    rewriter.setInsertionPointAfter(jump);
    StateNode elseBranch = StateNode::create(rewriter, op.getLoc(), "if_else");

    rewriter.setInsertionPointAfter(elseBranch);

    bool hasElseBlock = op.elseBlock();

    if (hasElseBlock) {
      op.elseBlock()->back().setAttr("linkToNext", rewriter.getBoolAttr(true));

      Block *cont = rewriter.splitBlock(rewriter.getBlock(),
                                        rewriter.getInsertionPoint());

      rewriter.mergeBlocks(op.elseBlock(), rewriter.getBlock(), {});
      rewriter.mergeBlocks(cont, rewriter.getBlock(), {});
    }

    StateNode merge = StateNode::create(rewriter, op.getLoc(), "if_merge");

    if (!hasElseBlock) {
      linkToLastState(rewriter, op.getLoc(), merge);
    }

    rewriter.setInsertionPointToEnd(&sdfg.body().getBlocks().front());

    ArrayAttr initArr = rewriter.getStrArrayAttr({condName + ": ref"});
    StringAttr trueCondition = rewriter.getStringAttr("1");
    EdgeOp::create(rewriter, op.getLoc(), init, guard, initArr, trueCondition,
                   getTransientValue(adaptor.getCondition()));

    ArrayAttr emptyArray = rewriter.getStrArrayAttr({});
    StringAttr condStr = rewriter.getStringAttr(condName);
    EdgeOp::create(rewriter, op.getLoc(), guard, ifBranch, emptyArray, condStr,
                   nullptr);

    StringAttr notCondStr = rewriter.getStringAttr("not (" + condName + ")");
    EdgeOp::create(rewriter, op.getLoc(), guard, elseBranch, emptyArray,
                   notCondStr, nullptr);

    EdgeOp::create(rewriter, op.getLoc(), jump, merge);

    if (markedToLink(*op))
      linkToNextState(rewriter, op->getLoc(), merge);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

void populateGenericToSDFGConversionPatterns(RewritePatternSet &patterns,
                                             TypeConverter &converter) {
  MLIRContext *ctxt = patterns.getContext();

  patterns.add<FuncToSDFG>(converter, ctxt);
  patterns.add<CallToSDFG>(converter, ctxt);
  patterns.add<OpToTasklet>(converter, ctxt);
  patterns.add<EraseTerminators>(converter, ctxt);

  patterns.add<MemrefLoadToSDFG>(converter, ctxt);
  patterns.add<MemrefStoreToSDFG>(converter, ctxt);
  patterns.add<MemrefGlobalToSDFG>(converter, ctxt);
  patterns.add<MemrefGetGlobalToSDFG>(converter, ctxt);
  patterns.add<MemrefAllocToSDFG>(converter, ctxt);

  patterns.add<SCFForToSDFG>(converter, ctxt);
  patterns.add<SCFIfToSDFG>(converter, ctxt);
}

namespace {
struct GenericToSDFGPass : public GenericToSDFGPassBase<GenericToSDFGPass> {
  void runOnOperation() override;
};
} // namespace

void GenericToSDFGPass::runOnOperation() {
  ModuleOp module = getOperation();

  // Clear all attributes
  for (NamedAttribute a : module->getAttrs())
    module->removeAttr(a.getName());

  SDFGTarget target(getContext());
  MemrefToMemletConverter converter;

  RewritePatternSet patterns(&getContext());
  populateGenericToSDFGConversionPatterns(patterns, converter);

  if (applyFullConversion(module, target, std::move(patterns)).failed())
    signalPassFailure();
}

std::unique_ptr<Pass> conversion::createGenericToSDFGPass() {
  return std::make_unique<GenericToSDFGPass>();
}
