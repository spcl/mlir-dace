// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

#include "SDFG/Utils/NameGenerator.h"

namespace mlir::sdfg::utils {
namespace {
int nameGeneratorID = 0;
}

std::string generateName(std::string base) {
  return base + "_" + std::to_string(nameGeneratorID++);
}

} // namespace mlir::sdfg::utils
