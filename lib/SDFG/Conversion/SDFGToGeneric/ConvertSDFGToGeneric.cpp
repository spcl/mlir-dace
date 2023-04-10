#include "SDFG/Conversion/SDFGToGeneric/PassDetail.h"
#include "SDFG/Conversion/SDFGToGeneric/Passes.h"
#include "SDFG/Dialect/Dialect.h"
#include "SDFG/Utils/Utils.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/IRMapping.h"

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

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Func Patterns
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

void populateSDFGToGenericConversionPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctxt = patterns.getContext();
}

namespace {
struct SDFGToGenericPass
    : public sdfg::conversion::SDFGToGenericPassBase<SDFGToGenericPass> {
  void runOnOperation() override;
};
} // namespace

void SDFGToGenericPass::runOnOperation() {
  ModuleOp module = getOperation();

  SDFGTarget target(getContext());

  RewritePatternSet patterns(&getContext());
  populateSDFGToGenericConversionPatterns(patterns);

  if (applyFullConversion(module, target, std::move(patterns)).failed())
    signalPassFailure();
}

std::unique_ptr<Pass> conversion::createSDFGToGenericPass() {
  return std::make_unique<SDFGToGenericPass>();
}
