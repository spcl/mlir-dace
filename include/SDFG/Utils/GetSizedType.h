// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

/// Header for the sized type utility functions.

#ifndef SDFG_Utils_GetSizedType_H
#define SDFG_Utils_GetSizedType_H

#include "SDFG/Dialect/Dialect.h"

namespace mlir::sdfg::utils {

/// Extracts the sized type from an array or stream type.
SizedType getSizedType(Type t);
/// Returns true if the provided type is a sized type.
bool isSizedType(Type t);

} // namespace mlir::sdfg::utils

#endif // SDFG_Utils_GetSizedType_H
