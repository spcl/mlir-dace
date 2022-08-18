#include "SDFG/Utils/AttributeToString.h"
#include "SDFG/Utils/Utils.h"
#include "mlir/IR/AsmState.h"

namespace mlir::sdfg::utils {

std::string attributeToString(Attribute attribute, Operation &op) {
  Operation *sdfg;

  if (isa<SDFGNode>(op))
    sdfg = &op;
  else
    sdfg = utils::getParentSDFG(op);

  // NOTE: This AsmState seems to be unnecessary
  AsmState state(sdfg);
  std::string name;
  llvm::raw_string_ostream nameStream(name);

  if (IntegerAttr attr = attribute.dyn_cast<IntegerAttr>()) {
    return std::to_string(attr.getInt());
  }

  if (StringAttr attr = attribute.dyn_cast<StringAttr>()) {
    return attr.getValue().str();
  }

  attribute.print(nameStream);
  utils::sanitizeName(name);
  return name;
}

} // namespace mlir::sdfg::utils
