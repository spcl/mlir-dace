#include "SDFG/Utils/ValueToString.h"
#include "SDFG/Utils/Utils.h"
#include "mlir/IR/AsmState.h"

namespace mlir::sdfg::utils {

std::string valueToString(Value value) {
  if (value.getDefiningOp() != nullptr)
    return valueToString(value, *value.getDefiningOp());

  return valueToString(value, *value.getParentBlock()->getParentOp());
}

std::string valueToString(Value value, Operation &op) {
  SDFGNode sdfg;

  if (SDFGNode sdfgNode = dyn_cast<SDFGNode>(op))
    sdfg = sdfgNode;
  else
    sdfg = utils::getParentSDFG(op);

  AsmState state(sdfg);
  std::string name;
  llvm::raw_string_ostream nameStream(name);
  value.printAsOperand(nameStream, state);
  utils::sanitizeName(name);
  return name;
}

} // namespace mlir::sdfg::utils
