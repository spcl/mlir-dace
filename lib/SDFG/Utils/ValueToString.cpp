// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

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
  Operation *sdfg;

  if (isa<SDFGNode>(op))
    sdfg = &op;
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
