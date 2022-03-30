#include "SDFG/Utils/ValueToString.h"
#include "SDFG/Utils/Utils.h"
#include "mlir/IR/AsmState.h"

namespace mlir::sdfg::utils {

std::string valueToString(Value value) {
  return valueToString(value, *value.getDefiningOp());
}

std::string valueToString(Value value, Operation &op) {
  AsmState state(utils::getParentSDFG(op));
  std::string name;
  llvm::raw_string_ostream nameStream(name);
  value.printAsOperand(nameStream, state);
  utils::sanitizeName(name);
  return name;
}

} // namespace mlir::sdfg::utils
