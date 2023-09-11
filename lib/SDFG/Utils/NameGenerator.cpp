// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

/// This file contains the name generator utility functions.

#include "SDFG/Utils/NameGenerator.h"

namespace mlir::sdfg::utils {
namespace {
int nameGeneratorID = 0;
}

/// Converts the provided string to a globally unique one.
std::string generateName(std::string base) {
  return base + "_" + std::to_string(nameGeneratorID++);
}

} // namespace mlir::sdfg::utils
