// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

/// Header for the name generator utility functions.

#ifndef SDFG_Utils_NameGenerator_H
#define SDFG_Utils_NameGenerator_H

#include <string>

namespace mlir::sdfg::utils {

/// Converts the provided string to a globally unique one.
std::string generateName(std::string base);

} // namespace mlir::sdfg::utils

#endif // SDFG_Utils_NameGenerator_H
