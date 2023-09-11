// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

/// This file defines a converter from the linalg dialect to the SDFG dialect.

#include "SDFG/Conversion/LinalgToSDFG/PassDetail.h"
#include "SDFG/Conversion/LinalgToSDFG/Passes.h"
#include "SDFG/Dialect/Dialect.h"
#include "SDFG/Utils/Utils.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"

using namespace mlir;
using namespace sdfg;
using namespace conversion;

//===----------------------------------------------------------------------===//
// Target & Type Converter
//===----------------------------------------------------------------------===//

/// Defines the target to convert to.
struct SDFGTarget : public ConversionTarget {
  SDFGTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    // Every operation is legal (best effort)
    markUnknownOpDynamicallyLegal([](Operation *op) { return true; });
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

/// Registers all the patterns above in a RewritePatternSet.
void populateLinalgToSDFGConversionPatterns(RewritePatternSet &patterns) {}

namespace {
struct LinalgToSDFGPass
    : public sdfg::conversion::LinalgToSDFGPassBase<LinalgToSDFGPass> {
  LinalgToSDFGPass() = default;

  void runOnOperation() override;
};
} // namespace

/// Runs the pass on the top-level module operation.
void LinalgToSDFGPass::runOnOperation() {
  ModuleOp module = getOperation();

  SDFGTarget target(getContext());

  RewritePatternSet patterns(&getContext());
  populateLinalgToSDFGConversionPatterns(patterns);

  if (applyFullConversion(module, target, std::move(patterns)).failed())
    signalPassFailure();
}

/// Returns a unique pointer to this pass.
std::unique_ptr<Pass> conversion::createLinalgToSDFGPass() {
  return std::make_unique<LinalgToSDFGPass>();
}
