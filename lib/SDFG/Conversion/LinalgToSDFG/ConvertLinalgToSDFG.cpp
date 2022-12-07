#include "SDFG/Conversion/LinalgToSDFG/PassDetail.h"
#include "SDFG/Conversion/LinalgToSDFG/Passes.h"
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
    // Every operation is legal (best effort)
    markUnknownOpDynamicallyLegal([](Operation *op) { return true; });
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

void populateLinalgToSDFGConversionPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctxt = patterns.getContext();

}

namespace {
struct LinalgToSDFGPass
    : public sdfg::conversion::LinalgToSDFGPassBase<LinalgToSDFGPass> {
  LinalgToSDFGPass() = default;

  void runOnOperation() override;
};
} // namespace

void LinalgToSDFGPass::runOnOperation() {
  ModuleOp module = getOperation();

  SDFGTarget target(getContext());

  RewritePatternSet patterns(&getContext());
  populateLinalgToSDFGConversionPatterns(patterns);

  if (applyFullConversion(module, target, std::move(patterns)).failed())
    signalPassFailure();
}

std::unique_ptr<Pass>
conversion::createLinalgToSDFGPass() {
  return std::make_unique<LinalgToSDFGPass>();
}
