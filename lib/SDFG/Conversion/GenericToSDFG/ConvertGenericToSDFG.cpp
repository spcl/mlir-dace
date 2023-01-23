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
    addConversion(convertLLVMPtrTypes);
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

  static Optional<Type> convertLLVMPtrTypes(Type type) {
    if (mlir::LLVM::LLVMPointerType ptrType =
            type.dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      SmallVector<int64_t> ints;
      SmallVector<StringAttr> symbols;
      SmallVector<bool> shape;

      Type elem = ptrType;

      while (mlir::LLVM::LLVMPointerType elemPtr =
                 elem.dyn_cast<mlir::LLVM::LLVMPointerType>()) {
        StringAttr sym =
            StringAttr::get(ptrType.getContext(), utils::generateName("s"));
        symbols.push_back(sym);
        shape.push_back(false);
        elem = elemPtr.getElementType();
      }

      SizedType sized =
          SizedType::get(ptrType.getContext(), elem, symbols, ints, shape);

      return ArrayType::get(ptrType.getContext(), sized);
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

Operation *getParentSDFG(Operation *op) {
  Operation *parent = op->getParentOp();

  if (isa<SDFGNode>(parent))
    return parent;

  if (isa<NestedSDFGNode>(parent))
    return parent;

  return getParentSDFG(parent);
}

uint32_t getSDFGNumArgs(Operation *op) {
  if (SDFGNode sdfg = dyn_cast<SDFGNode>(op)) {
    return sdfg.num_args();
  }

  if (NestedSDFGNode sdfg = dyn_cast<NestedSDFGNode>(op)) {
    return sdfg.num_args();
  }

  return -1;
}

StateNode getFirstState(Operation *op) {
  if (SDFGNode sdfg = dyn_cast<SDFGNode>(op)) {
    return sdfg.getFirstState();
  }

  if (NestedSDFGNode sdfg = dyn_cast<NestedSDFGNode>(op)) {
    return sdfg.getFirstState();
  }

  return nullptr;
}

ModuleOp getTopModuleOp(Operation *op) {
  Operation *parent = op->getParentOp();

  if (isa<ModuleOp>(parent))
    return cast<ModuleOp>(parent);

  return getTopModuleOp(parent);
}

void linkToLastState(PatternRewriter &rewriter, Location loc,
                     StateNode &state) {
  Operation *sdfg = getParentSDFG(state);
  OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToEnd(&sdfg->getRegion(0).getBlocks().front());

  StateNode prev;

  for (StateNode sn : sdfg->getRegion(0).getOps<StateNode>()) {
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
  Operation *sdfg = getParentSDFG(state);
  OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToEnd(&sdfg->getRegion(0).getBlocks().front());

  bool visitedState = false;
  for (StateNode sn : sdfg->getRegion(0).getOps<StateNode>()) {
    if (visitedState) {
      EdgeOp::create(rewriter, loc, state, sn);
      break;
    }

    if (sn == state)
      visitedState = true;
  }
  rewriter.restoreInsertionPoint(ip);
}

void markToLink(Operation &op) {
  BoolAttr boolAttr = BoolAttr::get(op.getContext(), true);
  op.setAttr("linkToNext", boolAttr);
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
// Func Patterns
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

    // TODO: Should be passed by a subflag
    if (!op.getName().equals("main")) {
      // COMMENT: The nested SDFG is created at the call operation conversion
      rewriter.eraseOp(op);
      return success();
    }

    // HACK: Replaces the print_array call with returning arrays (PolybenchC)
    if (op.getName().equals("main")) {
      for (int i = op.getNumArguments() - 1; i >= 0; --i)
        if (op.getArgument(i).getUses().empty())
          op.eraseArgument(i);

      SmallVector<Value> arrays = {};
      SmallVector<Type> arrayTypes = {};
      bool hasPrinter = false;

      for (func::CallOp callOp : op.getBody().getOps<func::CallOp>()) {
        if (callOp.getCallee().equals("print_array")) {
          hasPrinter = true;

          for (Value operand : callOp.getOperands()) {
            if (operand.getType().isa<MemRefType>()) {
              arrays.push_back(operand);
              arrayTypes.push_back(operand.getType());
            }
          }
        }
      }

      if (hasPrinter) {
        op.setType(rewriter.getFunctionType({}, arrayTypes));

        for (func::ReturnOp returnOp : op.getBody().getOps<func::ReturnOp>()) {
          returnOp->setOperands(arrays);
        }
      }
    }

    SmallVector<Type> args = {};
    for (unsigned i = 0; i < op.getNumArguments(); ++i) {
      MemrefToMemletConverter memo;
      Type nt = memo.convertType(op.getArgumentTypes()[i]);
      args.push_back(nt);
    }

    for (unsigned i = 0; i < op.getNumResults(); ++i) {
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

    if (failed(rewriter.convertRegionTypes(&sdfg.body(), *getTypeConverter())))
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

    // TODO: Support external calls
    // TODO: Support return values
    std::string callee = op.getCallee().str();
    ModuleOp mod = getTopModuleOp(op);
    func::FuncOp funcOp = dyn_cast<func::FuncOp>(mod.lookupSymbol(callee));

    // HACK: The function call got replaced at `FuncToSDFG` (PolybenchC)
    if (callee == "print_array") {
      rewriter.eraseOp(op);
      rewriter.eraseOp(funcOp);
      return success();
    }

    // HACK: Removes special function calls (cbrt, exit) and creates tasklet
    //  with annotation (LULESH)
    if (callee == "cbrt" || callee == "exit") {
      StateNode state = StateNode::create(rewriter, op->getLoc(), callee);

      SmallVector<Value> operands = adaptor.operands();
      SmallVector<Value> loadedOps =
          createLoads(rewriter, op->getLoc(), operands);

      TaskletNode task = TaskletNode::create(rewriter, op->getLoc(), loadedOps,
                                             op->getResultTypes());
      task->setAttr("insert_code", rewriter.getStringAttr(callee));

      if (task.getNumResults() == 1)
        sdfg::ReturnOp::create(rewriter, op.getLoc(),
                               task.body().getArguments());
      else
        sdfg::ReturnOp::create(rewriter, op.getLoc(), {});

      rewriter.setInsertionPointAfter(task);

      if (task.getNumResults() == 1) {
        MemrefToMemletConverter memo;
        Type nt = memo.convertType(op->getResultTypes()[0]);
        SizedType sized =
            SizedType::get(op->getLoc().getContext(), nt, {}, {}, {});
        nt = ArrayType::get(op->getLoc().getContext(), sized);

        Operation *sdfg = getParentSDFG(state);
        OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
        rewriter.setInsertionPointToStart(
            &sdfg->getRegion(0).getBlocks().front());
        AllocOp alloc =
            AllocOp::create(rewriter, op->getLoc(), nt, "_" + callee + "_tmp",
                            /*transient=*/true);

        rewriter.restoreInsertionPoint(ip);

        StoreOp::create(rewriter, op->getLoc(), task.getResult(0), alloc,
                        ValueRange());

        LoadOp load =
            LoadOp::create(rewriter, op->getLoc(), alloc, ValueRange());

        rewriter.replaceOp(op, {load});
      } else {
        rewriter.eraseOp(op);
      }

      linkToLastState(rewriter, op->getLoc(), state);
      if (markedToLink(*op))
        linkToNextState(rewriter, op->getLoc(), state);

      return success();
    }

    StateNode callState = StateNode::create(rewriter, op.getLoc(), callee);

    SmallVector<Value> operands = adaptor.getOperands();
    SmallVector<Value> loadedOps =
        createLoads(rewriter, op->getLoc(), operands);

    // FIXME: Write a proper reordering. Primitives then Memrefs
    unsigned firstMemref = 0;
    for (unsigned i = 0; i < loadedOps.size(); ++i) {
      if (loadedOps[i].getType().isa<MemRefType>() ||
          loadedOps[i].getType().isa<ArrayType>() ||
          loadedOps[i].getType().isa<StreamType>()) {
        firstMemref = i;
        break;
      }
    }

    NestedSDFGNode nestedSDFG = NestedSDFGNode::create(
        rewriter, op.getLoc(), firstMemref, ValueRange(loadedOps));
    StateNode initState =
        StateNode::create(rewriter, op.getLoc(), callee + "_init");

    rewriter.updateRootInPlace(nestedSDFG, [&] {
      nestedSDFG.entryAttr(
          SymbolRefAttr::get(op.getLoc().getContext(), initState.sym_name()));
    });

    for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
      funcOp.getArgument(i).replaceAllUsesWith(
          nestedSDFG.body().getArgument(i));
    }

    rewriter.inlineRegionBefore(funcOp.getBody(), nestedSDFG.body(),
                                nestedSDFG.body().end());

    rewriter.mergeBlocks(&nestedSDFG.body().getBlocks().back(),
                         &nestedSDFG.body().getBlocks().front(),
                         nestedSDFG.body().getArguments());

    if (failed(rewriter.convertRegionTypes(&nestedSDFG.body(),
                                           *getTypeConverter())))
      return failure();

    rewriter.eraseOp(funcOp);

    linkToLastState(rewriter, op.getLoc(), callState);
    if (markedToLink(*op))
      linkToNextState(rewriter, op->getLoc(), callState);

    rewriter.eraseOp(op);
    return success();
  }
};

class ReturnToSDFG : public OpConversionPattern<func::ReturnOp> {
public:
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StateNode state = StateNode::create(rewriter, op->getLoc(), "return");
    Operation *sdfg = getParentSDFG(state);

    std::vector<Value> operands = {};
    for (Value operand : adaptor.operands())
      operands.push_back(operand);

    SmallVector<Value> loadedOps =
        createLoads(rewriter, op->getLoc(), operands);

    for (unsigned i = 0; i < loadedOps.size(); ++i) {
      if (loadedOps[i].getType().isa<ArrayType>()) {
        CopyOp::create(
            rewriter, op->getLoc(), loadedOps[i],
            sdfg->getRegion(0).getArgument(getSDFGNumArgs(sdfg) + i));
      } else {
        StoreOp::create(
            rewriter, op->getLoc(), loadedOps[i],
            sdfg->getRegion(0).getArgument(getSDFGNumArgs(sdfg) + i),
            ValueRange());
      }
    }

    linkToLastState(rewriter, op->getLoc(), state);
    if (markedToLink(*op))
      linkToNextState(rewriter, op->getLoc(), state);
    rewriter.eraseOp(op);
    return success();

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Arith & Math Patterns
//===----------------------------------------------------------------------===//

class OpToTasklet : public ConversionPattern {
public:
  OpToTasklet(TypeConverter &converter, MLIRContext *context)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getDialect()->getNamespace() ==
            arith::ArithmeticDialect::getDialectNamespace() ||
        op->getDialect()->getNamespace() ==
            math::MathDialect::getDialectNamespace()) {

      if (isa<TaskletNode>(op->getParentOp()))
        return failure(); // Operation already in a
                          // tasklet

      std::string name = utils::operationToString(*op);
      StateNode state = StateNode::create(rewriter, op->getLoc(), name);

      Operation *sdfg = getParentSDFG(state);
      OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointToStart(
          &sdfg->getRegion(0).getBlocks().front());

      SmallVector<AllocOp> allocs;

      for (Type opType : op->getResultTypes()) {
        MemrefToMemletConverter memo;
        Type newType = memo.convertType(opType);
        SizedType sizedType =
            SizedType::get(op->getLoc().getContext(), newType, {}, {}, {});
        newType = ArrayType::get(op->getLoc().getContext(), sizedType);
        AllocOp alloc = AllocOp::create(rewriter, op->getLoc(), newType,
                                        "_" + name + "_tmp",
                                        /*transient=*/true);
        allocs.push_back(alloc);
      }

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

      SmallVector<Value> loads;

      for (AllocOp alloc : allocs) {
        StoreOp::create(rewriter, op->getLoc(), task.getResult(0), alloc,
                        ValueRange());

        LoadOp load =
            LoadOp::create(rewriter, op->getLoc(), alloc, ValueRange());
        loads.push_back(load);
      }

      rewriter.replaceOp(op, loads);

      linkToLastState(rewriter, op->getLoc(), state);
      if (markedToLink(*op))
        linkToNextState(rewriter, op->getLoc(), state);

      return success();
    }

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Memref Patterns
//===----------------------------------------------------------------------===//

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

    Operation *sdfg = getParentSDFG(op);
    OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(&sdfg->getRegion(0).getBlocks().front());
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
    Operation *sdfg = getParentSDFG(op);

    OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(&sdfg->getRegion(0).getBlocks().front());

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
    Operation *sdfg = getParentSDFG(op);

    OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(&sdfg->getRegion(0).getBlocks().front());

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

      rewriter.setInsertionPointToEnd(&sdfg->getRegion(0).getBlocks().front());

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

class MemrefAllocaToSDFG : public OpConversionPattern<memref::AllocaOp> {
public:
  using OpConversionPattern<memref::AllocaOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *sdfg = getParentSDFG(op);

    OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(&sdfg->getRegion(0).getBlocks().front());

    Type type = getTypeConverter()->convertType(op.getType());
    AllocOp alloc = AllocOp::create(rewriter, op->getLoc(), type, "_alloca_tmp",
                                    /*transient=*/true);

    rewriter.restoreInsertionPoint(ip);
    StateNode init = StateNode::create(rewriter, op.getLoc(), "alloca_init");
    linkToLastState(rewriter, op.getLoc(), init);

    StateNode lastState = init;
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      rewriter.setInsertionPointAfter(lastState);
      StateNode alloc_param =
          StateNode::create(rewriter, op.getLoc(), "alloca_param");

      rewriter.setInsertionPointToEnd(&sdfg->getRegion(0).getBlocks().front());

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

class MemrefDeallocToSDFG : public OpConversionPattern<memref::DeallocOp> {
public:
  using OpConversionPattern<memref::DeallocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

class MemrefCastToSDFG : public OpConversionPattern<memref::CastOp> {
public:
  using OpConversionPattern<memref::CastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type type = getTypeConverter()->convertType(op.getType());

    SizedType sized =
        SizedType::get(op->getLoc().getContext(), type, {}, {}, {});
    Type arrT = ArrayType::get(op->getLoc().getContext(), sized);

    Operation *sdfg = getParentSDFG(op);
    OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(&sdfg->getRegion(0).getBlocks().front());
    AllocOp alloc = AllocOp::create(rewriter, op->getLoc(), arrT, "_cast_tmp",
                                    /*transient=*/true);

    rewriter.restoreInsertionPoint(ip);
    StateNode state = StateNode::create(rewriter, op->getLoc(), "cast");
    CopyOp::create(rewriter, op.getLoc(), adaptor.source(), alloc);

    rewriter.replaceOp(op, {alloc});

    return success();
  }
};

//===----------------------------------------------------------------------===//
// SCF Patterns
//===----------------------------------------------------------------------===//

class SCFForToSDFG : public OpConversionPattern<scf::ForOp> {
public:
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::string idxName = utils::generateName("for_idx");

    Operation *sdfg = getParentSDFG(op);
    OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(&sdfg->getRegion(0).getBlocks().front());
    AllocSymbolOp::create(rewriter, op.getLoc(), idxName);

    std::vector<AllocOp> iterAllocs = {};

    for (unsigned i = 0; i < op.getNumIterOperands(); ++i) {
      Value iterOp = op.getIterOperands()[i];

      MemrefToMemletConverter memo;
      Type nt = memo.convertType(iterOp.getType());
      SizedType sized =
          SizedType::get(op->getLoc().getContext(), nt, {}, {}, {});
      nt = ArrayType::get(op->getLoc().getContext(), sized);

      AllocOp alloc =
          AllocOp::create(rewriter, op->getLoc(), nt,
                          "_" + idxName + "_iter" + std::to_string(i),
                          /*transient=*/true);

      iterAllocs.push_back(alloc);
    }

    rewriter.restoreInsertionPoint(ip);

    StateNode init = StateNode::create(rewriter, op.getLoc(), "for_init");

    for (unsigned i = 0; i < op.getNumIterOperands(); ++i) {
      Value iterOp = op.getIterOperands()[i];
      Value initVal =
          createLoad(rewriter, op.getLoc(), adaptor.getInitArgs()[i]);

      StoreOp::create(rewriter, op->getLoc(), initVal, iterAllocs[i],
                      ValueRange());

      LoadOp newLoad =
          LoadOp::create(rewriter, op->getLoc(), iterAllocs[i], ValueRange());

      op.getRegionIterArgs()[i].replaceAllUsesWith(newLoad);

      // FIXME: Yield all
      for (scf::YieldOp yieldOp : op.getLoopBody().getOps<scf::YieldOp>()) {
        yieldOp->insertOperands(1, {iterAllocs[i]});
      }
    }

    linkToLastState(rewriter, op.getLoc(), init);

    rewriter.setInsertionPointAfter(init);
    StateNode guard = StateNode::create(rewriter, op.getLoc(), "for_guard");
    SymOp idxSym = SymOp::create(rewriter, op.getLoc(),
                                 op.getInductionVar().getType(), idxName);
    op.getInductionVar().replaceAllUsesWith(idxSym);

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

    rewriter.setInsertionPointToEnd(&sdfg->getRegion(0).getBlocks().front());
    ArrayAttr emptyArr = rewriter.getStrArrayAttr({});
    StringAttr emptyStr = rewriter.getStringAttr("1");

    if (adaptor.getLowerBound().getDefiningOp() != nullptr &&
        isa<SymOp>(adaptor.getLowerBound().getDefiningOp())) {
      SymOp symOp = cast<SymOp>(adaptor.getLowerBound().getDefiningOp());
      std::string assignment = idxName + ": " + symOp.expr().str();
      ArrayAttr initArr = rewriter.getStrArrayAttr({assignment});
      EdgeOp::create(rewriter, op.getLoc(), init, guard, initArr, emptyStr,
                     nullptr);
    } else {
      ArrayAttr initArr = rewriter.getStrArrayAttr({idxName + ": ref"});
      EdgeOp::create(rewriter, op.getLoc(), init, guard, initArr, emptyStr,
                     getTransientValue(adaptor.getLowerBound()));
    }

    if (adaptor.getUpperBound().getDefiningOp() != nullptr &&
        isa<SymOp>(adaptor.getUpperBound().getDefiningOp())) {
      SymOp symOp = cast<SymOp>(adaptor.getUpperBound().getDefiningOp());
      StringAttr guardStr =
          rewriter.getStringAttr(idxName + " < " + symOp.expr());
      EdgeOp::create(rewriter, op.getLoc(), guard, body, emptyArr, guardStr,
                     nullptr);
    } else {
      StringAttr guardStr = rewriter.getStringAttr(idxName + " < ref");
      EdgeOp::create(rewriter, op.getLoc(), guard, body, emptyArr, guardStr,
                     getTransientValue(adaptor.getUpperBound()));
    }

    if (adaptor.getStep().getDefiningOp() != nullptr &&
        isa<SymOp>(adaptor.getStep().getDefiningOp())) {
      SymOp symOp = cast<SymOp>(adaptor.getStep().getDefiningOp());
      std::string assignment =
          idxName + ": " + idxName + " + " + symOp.expr().str();
      ArrayAttr returnArr = rewriter.getStrArrayAttr({assignment});
      EdgeOp::create(rewriter, op.getLoc(), returnState, guard, returnArr,
                     emptyStr, nullptr);
    } else {
      ArrayAttr returnArr =
          rewriter.getStrArrayAttr({idxName + ": " + idxName + " + ref"});
      EdgeOp::create(rewriter, op.getLoc(), returnState, guard, returnArr,
                     emptyStr, getTransientValue(adaptor.getStep()));
    }

    if (adaptor.getUpperBound().getDefiningOp() != nullptr &&
        isa<SymOp>(adaptor.getUpperBound().getDefiningOp())) {
      SymOp symOp = cast<SymOp>(adaptor.getUpperBound().getDefiningOp());
      StringAttr exitStr =
          rewriter.getStringAttr("not(" + idxName + " < " + symOp.expr() + ")");
      EdgeOp::create(rewriter, op.getLoc(), guard, exitState, emptyArr, exitStr,
                     nullptr);
    } else {
      StringAttr exitStr = rewriter.getStringAttr("not(" + idxName + " < ref)");
      EdgeOp::create(rewriter, op.getLoc(), guard, exitState, emptyArr, exitStr,
                     getTransientValue(adaptor.getUpperBound()));
    }

    if (markedToLink(*op))
      linkToNextState(rewriter, op->getLoc(), exitState);

    rewriter.eraseOp(op);
    return success();
  }
};

class SCFWhileToSDFG : public OpConversionPattern<scf::WhileOp> {
public:
  using OpConversionPattern<scf::WhileOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::WhileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Values
    Location loc = op.getLoc();
    MLIRContext *context = loc->getContext();
    scf::ConditionOp conditionOp = op.getConditionOp();
    scf::YieldOp yieldOp = op.getYieldOp();

    // Allocs
    Operation *sdfg = getParentSDFG(op);
    OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(&sdfg->getRegion(0).getBlocks().front());

    std::vector<AllocOp> iterAllocs = {};
    std::vector<AllocOp> argAllocs = {};

    // Itervars
    for (BlockArgument arg : op.getBeforeArguments()) {
      std::string symName = utils::generateName(utils::valueToString(arg));

      MemrefToMemletConverter converter;
      Type newType = converter.convertType(arg.getType());
      SizedType sizedType = SizedType::get(context, newType, {}, {}, {});
      ArrayType arrayType = ArrayType::get(context, sizedType);

      AllocOp alloc = AllocOp::create(rewriter, loc, arrayType, symName,
                                      /*transient=*/true);
      yieldOp->insertOperands(yieldOp.getNumOperands(), {alloc});
      iterAllocs.push_back(alloc);
    }

    // Condition
    MemrefToMemletConverter converter;
    Type newType = converter.convertType(conditionOp.getCondition().getType());
    SizedType sizedType = SizedType::get(context, newType, {}, {}, {});
    ArrayType arrayType = ArrayType::get(context, sizedType);
    AllocOp conditionAlloc =
        AllocOp::create(rewriter, loc, arrayType, "while_condition",
                        /*transient=*/true);

    // Condition Arguments
    for (Value arg : conditionOp.getArgs()) {
      MemrefToMemletConverter converter;
      Type newType = converter.convertType(arg.getType());
      SizedType sizedType = SizedType::get(context, newType, {}, {}, {});
      ArrayType arrayType = ArrayType::get(context, sizedType);
      AllocOp alloc = AllocOp::create(rewriter, loc, arrayType, "while_arg",
                                      /*transient=*/true);
      argAllocs.push_back(alloc);
    }

    rewriter.restoreInsertionPoint(ip);

    // Init state
    StateNode initState = StateNode::create(rewriter, loc, "while_init");

    linkToLastState(rewriter, loc, initState);
    rewriter.setInsertionPointAfter(initState);

    // Guard begin state
    StateNode guardBeginState =
        StateNode::create(rewriter, loc, "while_guard_begin");
    // Add loads
    std::vector<LoadOp> argLoads = {};
    for (AllocOp alloc : argAllocs) {
      LoadOp loadOp = LoadOp::create(rewriter, loc, alloc, {});
      argLoads.push_back(loadOp);
    }
    std::vector<Value> argLoadsValue(argLoads.begin(), argLoads.end());

    linkToLastState(rewriter, loc, guardBeginState);
    rewriter.setInsertionPointAfter(guardBeginState);

    // Mark last op for linking
    markToLink(op.getBefore().front().back());

    // Copy ops for guard calculation
    Block *cont =
        rewriter.splitBlock(rewriter.getBlock(), rewriter.getInsertionPoint());

    rewriter.mergeBlocks(&op.getBefore().front(), rewriter.getBlock(),
                         argLoadsValue);
    rewriter.mergeBlocks(cont, rewriter.getBlock(), {});

    // Guard end state
    StateNode guardEndState =
        StateNode::create(rewriter, loc, "while_guard_end");

    rewriter.setInsertionPointAfter(guardEndState);

    // Body state
    StateNode bodyState = StateNode::create(rewriter, loc, "while_body");
    rewriter.setInsertionPointAfter(bodyState);

    // Mark last op for linking
    markToLink(op.getAfter().front().back());

    // Copy all body ops
    cont =
        rewriter.splitBlock(rewriter.getBlock(), rewriter.getInsertionPoint());

    rewriter.mergeBlocks(&op.getAfter().front(), rewriter.getBlock(),
                         argLoadsValue);
    rewriter.mergeBlocks(cont, rewriter.getBlock(), {});

    // Return state
    StateNode returnState = StateNode::create(rewriter, loc, "while_return");
    rewriter.setInsertionPointAfter(returnState);

    // Exit state
    StateNode exitState = StateNode::create(rewriter, loc, "while_exit");

    // Edges
    rewriter.setInsertionPointToEnd(&sdfg->getRegion(0).getBlocks().front());
    ArrayAttr emptyAssign = rewriter.getStrArrayAttr({});
    // Guard_end -> Body
    StringAttr condStr = rewriter.getStringAttr("ref");
    EdgeOp::create(rewriter, loc, guardEndState, bodyState, emptyAssign,
                   condStr, conditionAlloc);
    // Guard_end -> Exit
    StringAttr notCondStr = rewriter.getStringAttr("not (ref)");
    EdgeOp::create(rewriter, loc, guardEndState, exitState, emptyAssign,
                   notCondStr, conditionAlloc);

    // Return -> Guard_begin
    EdgeOp::create(rewriter, loc, returnState, guardBeginState);

    // Inject condition array
    conditionOp->insertOperands(conditionOp.getNumOperands(), {conditionAlloc});

    // Inject argument arrays
    std::vector<Value> argAllocsValue(argAllocs.begin(), argAllocs.end());
    conditionOp->insertOperands(conditionOp.getNumOperands(), argAllocsValue);

    if (markedToLink(*op))
      linkToNextState(rewriter, loc, exitState);

    rewriter.replaceOp(op, argLoadsValue);
    return success();
  }
};

class SCFConditionToSDFG : public OpConversionPattern<scf::ConditionOp> {
public:
  using OpConversionPattern<scf::ConditionOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ConditionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StateNode state = StateNode::create(rewriter, op->getLoc(), "condition");

    // SCFWhileToSDFG sets the arguments in the following order:
    // [Arg Values] condition_array [Arg Arrays]

    unsigned argLength = (adaptor.getArgs().size() - 1) / 2;

    // Store condition value
    Value val = createLoad(rewriter, op.getLoc(), adaptor.getCondition());
    StoreOp::create(rewriter, op.getLoc(), val, adaptor.getArgs()[argLength],
                    {});

    // Store argument values
    for (unsigned i = 0; i < argLength; ++i) {
      Value val = createLoad(rewriter, op.getLoc(), adaptor.getArgs()[i]);
      StoreOp::create(rewriter, op.getLoc(), val,
                      adaptor.getArgs()[argLength + 1 + i], {});
    }

    linkToLastState(rewriter, op->getLoc(), state);
    if (markedToLink(*op))
      linkToNextState(rewriter, op->getLoc(), state);

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

    Operation *sdfg = getParentSDFG(op);
    OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(&sdfg->getRegion(0).getBlocks().front());
    AllocSymbolOp::create(rewriter, op.getLoc(), condName);

    if (op.getNumResults() > 0) {
      // FIXME: Alloc all
      MemrefToMemletConverter memo;

      Type nt = memo.convertType(op.getResultTypes()[0]);
      SizedType sized =
          SizedType::get(op->getLoc().getContext(), nt, {}, {}, {});
      nt = ArrayType::get(op->getLoc().getContext(), sized);

      AllocOp alloc =
          AllocOp::create(rewriter, op->getLoc(), nt, "_" + condName + "_yield",
                          /*transient=*/true);

      op.thenYield()->insertOperands(1, {alloc});
      op.elseYield()->insertOperands(1, {alloc});

      op.getResult(0).replaceAllUsesWith(alloc);
    }

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

    rewriter.setInsertionPointToEnd(&sdfg->getRegion(0).getBlocks().front());

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

class SCFYieldToSDFG : public OpConversionPattern<scf::YieldOp> {
public:
  using OpConversionPattern<scf::YieldOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StateNode state = StateNode::create(rewriter, op->getLoc(), "yield");

    // The operands are set by the For/While/If patterns in the following shape:
    // [Array of Values] [Array of Memrefs]

    // IDEA: Maybe add a check for divisibility?
    unsigned numVals = op->getNumOperands() / 2;

    // IDEA: Maybe replace with updating symbols
    for (unsigned i = 0; i < numVals; ++i) {
      Value val = createLoad(rewriter, op->getLoc(), adaptor.getOperands()[i]);
      Value memref = createLoad(rewriter, op->getLoc(),
                                adaptor.getOperands()[i + numVals]);
      StoreOp::create(rewriter, op->getLoc(), val, memref, {});
    }

    linkToLastState(rewriter, op->getLoc(), state);
    if (markedToLink(*op))
      linkToNextState(rewriter, op->getLoc(), state);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// LLVM Patterns
//===----------------------------------------------------------------------===//

class LLVMAllocaToSDFG : public OpConversionPattern<mlir::LLVM::AllocaOp> {
public:
  using OpConversionPattern<mlir::LLVM::AllocaOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::LLVM::AllocaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *sdfg = getParentSDFG(op);

    OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(&sdfg->getRegion(0).getBlocks().front());

    Type type = getTypeConverter()->convertType(op.getType());
    AllocOp alloc = AllocOp::create(rewriter, op->getLoc(), type, "_alloc_tmp",
                                    /*transient=*/true);

    rewriter.restoreInsertionPoint(ip);
    StateNode init = StateNode::create(rewriter, op.getLoc(), "alloc_init");
    linkToLastState(rewriter, op.getLoc(), init);

    StateNode lastState = init;
    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      rewriter.setInsertionPointAfter(lastState);
      StateNode alloc_param =
          StateNode::create(rewriter, op.getLoc(), "alloc_param");

      rewriter.setInsertionPointToEnd(&sdfg->getRegion(0).getBlocks().front());

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

class LLVMBitcastToSDFG : public OpConversionPattern<mlir::LLVM::BitcastOp> {
public:
  using OpConversionPattern<mlir::LLVM::BitcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::LLVM::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *sdfg = getParentSDFG(op);
    rewriter.setInsertionPoint(getFirstState(sdfg));

    Type type = getTypeConverter()->convertType(op.getType());
    ViewCastOp viewCast =
        ViewCastOp::create(rewriter, op.getLoc(), adaptor.getArg(), type);

    rewriter.replaceOp(op, {viewCast});
    return success();
  }
};

class LLVMGEPToSDFG : public OpConversionPattern<mlir::LLVM::GEPOp> {
public:
  using OpConversionPattern<mlir::LLVM::GEPOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::LLVM::GEPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value base = op.getBase();
    OperandRange indices = op.getIndices();
    SmallVector<Value> castedIndices = {};

    for (Value idx : indices) {
      OpBuilder builder(op.getLoc()->getContext());
      OperationState state(op.getLoc(), arith::IndexCastOp::getOperationName());

      arith::IndexCastOp::build(builder, state, rewriter.getIndexType(), idx);
      castedIndices.push_back(
          cast<arith::IndexCastOp>(rewriter.create(state)).getResult());
    }

    for (Operation *user : op.getRes().getUsers()) {
      user->insertOperands(user->getNumOperands(), castedIndices);
    }

    op.getRes().replaceAllUsesWith(base);

    rewriter.eraseOp(op);
    return success();
  }
};

class LLVMLoadToSDFG : public OpConversionPattern<mlir::LLVM::LoadOp> {
public:
  using OpConversionPattern<mlir::LLVM::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::LLVM::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Array and indices have been set by LLVMGEPToSDFG

    Type elemT = utils::getSizedType(op.getAddr().getType().cast<ArrayType>())
                     .getElementType();
    Type type = getTypeConverter()->convertType(elemT);
    SizedType sized =
        SizedType::get(op->getLoc().getContext(), type, {}, {}, {});
    Type arrT = ArrayType::get(op->getLoc().getContext(), sized);

    Operation *sdfg = getParentSDFG(op);
    OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(&sdfg->getRegion(0).getBlocks().front());
    AllocOp alloc = AllocOp::create(rewriter, op->getLoc(), arrT, "_load_tmp",
                                    /*transient=*/true);

    rewriter.restoreInsertionPoint(ip);
    StateNode state = StateNode::create(rewriter, op->getLoc(), "load");

    Value array = createLoad(rewriter, op.getLoc(), adaptor.getAddr());

    SmallVector<Value> indices =
        adaptor.getOperands().slice(1, op->getNumOperands() - 1);
    indices = createLoads(rewriter, op.getLoc(), indices);

    LoadOp load = LoadOp::create(rewriter, op.getLoc(), type, array, indices);
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

class LLVMStoreToSDFG : public OpConversionPattern<mlir::LLVM::StoreOp> {
public:
  using OpConversionPattern<mlir::LLVM::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::LLVM::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Array and indices have been set by LLVMGEPToSDFG

    StateNode state = StateNode::create(rewriter, op->getLoc(), "store");

    Value val = createLoad(rewriter, op.getLoc(), adaptor.getValue());
    Value array = createLoad(rewriter, op.getLoc(), adaptor.getAddr());

    SmallVector<Value> indices =
        adaptor.getOperands().slice(2, op->getNumOperands() - 2);
    indices = createLoads(rewriter, op.getLoc(), indices);

    StoreOp::create(rewriter, op.getLoc(), val, array, indices);

    linkToLastState(rewriter, op->getLoc(), state);
    if (markedToLink(*op))
      linkToNextState(rewriter, op->getLoc(), state);

    rewriter.eraseOp(op);
    return success();
  }
};

class LLVMGlobalToSDFG : public OpConversionPattern<mlir::LLVM::GlobalOp> {
public:
  using OpConversionPattern<mlir::LLVM::GlobalOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::LLVM::GlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

class LLVMFuncToSDFG : public OpConversionPattern<mlir::LLVM::LLVMFuncOp> {
public:
  using OpConversionPattern<mlir::LLVM::LLVMFuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::LLVM::LLVMFuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

class LLVMUndefToSDFG : public OpConversionPattern<mlir::LLVM::UndefOp> {
public:
  using OpConversionPattern<mlir::LLVM::UndefOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::LLVM::UndefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::string name = utils::operationToString(*op);
    StateNode state = StateNode::create(rewriter, op->getLoc(), name);

    MemrefToMemletConverter memo;
    Type nt = memo.convertType(op->getResultTypes()[0]);
    SizedType sized = SizedType::get(op->getLoc().getContext(), nt, {}, {}, {});
    nt = ArrayType::get(op->getLoc().getContext(), sized);

    Operation *sdfg = getParentSDFG(state);
    OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(&sdfg->getRegion(0).getBlocks().front());

    SmallVector<AllocOp> allocs;

    for (Type opType : op->getResultTypes()) {
      MemrefToMemletConverter memo;
      Type newType = memo.convertType(opType);
      SizedType sizedType =
          SizedType::get(op->getLoc().getContext(), newType, {}, {}, {});
      newType = ArrayType::get(op->getLoc().getContext(), sizedType);
      AllocOp alloc =
          AllocOp::create(rewriter, op->getLoc(), newType, "_" + name + "_tmp",
                          /*transient=*/true);
      allocs.push_back(alloc);
    }

    rewriter.restoreInsertionPoint(ip);

    TaskletNode task =
        TaskletNode::create(rewriter, op->getLoc(), {}, op->getResultTypes());

    BlockAndValueMapping mapping;
    mapping.map(op->getOperands(), task.body().getArguments());

    Operation *opClone = op->clone(mapping);
    rewriter.updateRootInPlace(
        task, [&] { task.body().getBlocks().front().push_front(opClone); });

    sdfg::ReturnOp::create(rewriter, opClone->getLoc(), opClone->getResults());

    rewriter.setInsertionPointAfter(task);

    SmallVector<Value> loads;

    for (AllocOp alloc : allocs) {
      StoreOp::create(rewriter, op->getLoc(), task.getResult(0), alloc,
                      ValueRange());

      LoadOp load = LoadOp::create(rewriter, op->getLoc(), alloc, ValueRange());
      loads.push_back(load);
    }

    rewriter.replaceOp(op, loads);

    linkToLastState(rewriter, op->getLoc(), state);
    if (markedToLink(*op))
      linkToNextState(rewriter, op->getLoc(), state);

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
  patterns.add<ReturnToSDFG>(converter, ctxt);
  patterns.add<CallToSDFG>(converter, ctxt);
  patterns.add<OpToTasklet>(converter, ctxt);

  patterns.add<MemrefLoadToSDFG>(converter, ctxt);
  patterns.add<MemrefStoreToSDFG>(converter, ctxt);
  patterns.add<MemrefGlobalToSDFG>(converter, ctxt);
  patterns.add<MemrefGetGlobalToSDFG>(converter, ctxt);
  patterns.add<MemrefAllocToSDFG>(converter, ctxt);
  patterns.add<MemrefAllocaToSDFG>(converter, ctxt);
  patterns.add<MemrefDeallocToSDFG>(converter, ctxt);
  patterns.add<MemrefCastToSDFG>(converter, ctxt);

  patterns.add<SCFForToSDFG>(converter, ctxt);
  patterns.add<SCFWhileToSDFG>(converter, ctxt);
  patterns.add<SCFConditionToSDFG>(converter, ctxt);
  patterns.add<SCFIfToSDFG>(converter, ctxt);
  patterns.add<SCFYieldToSDFG>(converter, ctxt);

  patterns.add<LLVMAllocaToSDFG>(converter, ctxt);
  patterns.add<LLVMBitcastToSDFG>(converter, ctxt);
  patterns.add<LLVMGEPToSDFG>(converter, ctxt);
  patterns.add<LLVMLoadToSDFG>(converter, ctxt);
  patterns.add<LLVMStoreToSDFG>(converter, ctxt);
  patterns.add<LLVMGlobalToSDFG>(converter, ctxt);
  patterns.add<LLVMFuncToSDFG>(converter, ctxt);
  patterns.add<LLVMUndefToSDFG>(converter, ctxt);
}

namespace {
struct GenericToSDFGPass
    : public sdfg::conversion::GenericToSDFGPassBase<GenericToSDFGPass> {
  std::string mainFuncName;

  GenericToSDFGPass() = default;

  explicit GenericToSDFGPass(StringRef mainFuncName)
      : mainFuncName(mainFuncName.str()) {}

  void runOnOperation() override;
};
} // namespace

// Gets the name of the first function that isn't called by any other function
llvm::Optional<std::string> getMainFunctionName(ModuleOp moduleOp) {
  for (func::FuncOp mainFuncOp : moduleOp.getOps<func::FuncOp>()) {
    // No need to check function declarations
    if (mainFuncOp.isDeclaration())
      continue;

    bool foundCallInOtherFunc = false;

    // Check against every other function
    for (func::FuncOp funcOp : moduleOp.getOps<func::FuncOp>()) {
      // No need to check function against itself
      if (funcOp.getName() == mainFuncOp.getName())
        continue;
      // No need to check function declarations
      if (funcOp.isDeclaration())
        continue;

      // Check every callOp for a call to the main function
      for (func::CallOp callOp : funcOp.getOps<func::CallOp>()) {
        if (callOp.getCallee() == mainFuncOp.getName()) {
          foundCallInOtherFunc = true;
          break;
        }
      }

      if (foundCallInOtherFunc)
        break;
    }

    if (!foundCallInOtherFunc)
      return mainFuncOp.getName().str();
  }

  return llvm::None;
}

void GenericToSDFGPass::runOnOperation() {
  ModuleOp module = getOperation();

  // TODO: Find a way to get func name via CLI instead of inferring
  llvm::Optional<std::string> mainFuncNameOpt = getMainFunctionName(module);
  if (mainFuncNameOpt)
    mainFuncName = *mainFuncNameOpt;

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

std::unique_ptr<Pass>
conversion::createGenericToSDFGPass(StringRef getMainFuncName) {
  return std::make_unique<GenericToSDFGPass>(getMainFuncName);
}
