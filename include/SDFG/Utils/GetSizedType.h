#ifndef SDFG_Utils_GetSizedType_H
#define SDFG_Utils_GetSizedType_H

#include "SDFG/Dialect/Dialect.h"

namespace mlir::sdfg::utils {

SizedType getSizedType(Type t);
bool isSizedType(Type t);

} // namespace mlir::sdfg::utils

#endif // SDFG_Utils_GetSizedType_H