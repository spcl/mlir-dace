#ifndef SDFG_Utils_AttributeToString_H
#define SDFG_Utils_AttributeToString_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include <string>

namespace mlir::sdfg::utils {

std::string attributeToString(Attribute attribute, Operation &op);

} // namespace mlir::sdfg::utils

#endif // SDFG_Utils_AttributeToString_H
