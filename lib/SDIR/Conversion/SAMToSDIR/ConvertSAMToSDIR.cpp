#include "SDIR/Conversion/SAMToSDIR/PassDetail.h"
#include "SDIR/Conversion/SAMToSDIR/Passes.h"

using namespace mlir;
using namespace sdir;
using namespace conversion;

namespace {
struct SAMToSDIRPass : public SAMToSDIRPassBase<SAMToSDIRPass> {
  void runOnOperation() override;
};
} // namespace

void SAMToSDIRPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (module.body().getBlocks().size() > 1) {
    return signalPassFailure();
  }

  // Build SDFG
  SDFGNode sdfg = SDFGNode::create(module.getLoc());
  StateNode statenode = StateNode::create(module.getLoc());
  sdfg.body().getBlocks().front().push_front(statenode);

  // Replace block content with SDFG
  Block &moduleBlock = module.body().getBlocks().front();
  moduleBlock.clear();
  moduleBlock.push_front(sdfg);
}

std::unique_ptr<Pass> conversion::createSAMToSDIRPass() {
  return std::make_unique<SAMToSDIRPass>();
}
