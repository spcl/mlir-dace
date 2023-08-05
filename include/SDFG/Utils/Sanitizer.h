// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

#ifndef SDFG_Utils_Sanitizer_H
#define SDFG_Utils_Sanitizer_H

#include <string>

namespace mlir::sdfg::utils {

/// Sanitizes the provided string to only include alphanumericals and
/// underscores.
void sanitizeName(std::string &name);

} // namespace mlir::sdfg::utils

#endif // SDFG_Utils_Sanitizer_H
