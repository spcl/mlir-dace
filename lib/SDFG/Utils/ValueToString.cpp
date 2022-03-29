#include "SDFG/Utils/ValueToString.h"
#include "SDFG/Utils/Utils.h"
#include "mlir/IR/AsmState.h"

namespace mlir::sdfg::utils {

std::string valueToString(Value value, bool useSDFG) {
  return valueToString(value, *value.getDefiningOp(), useSDFG);
}

std::string valueToString(Value value, Operation &op, bool useSDFG) {
  AsmState state(useSDFG ? utils::getParentSDFG(op)
                         : utils::getParentState(op));
  std::string name;
  llvm::raw_string_ostream nameStream(name);
  value.printAsOperand(nameStream, state);
  utils::sanitizeName(name);
  return name;
}

} // namespace mlir::sdfg::utils
