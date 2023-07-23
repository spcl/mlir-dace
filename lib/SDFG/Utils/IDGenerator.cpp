// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

#include "SDFG/Utils/IDGenerator.h"

namespace mlir::sdfg::utils {
namespace {
unsigned idGeneratorID = 0;
}

unsigned generateID() { return idGeneratorID++; }

void resetIDGenerator() { idGeneratorID = 0; }

} // namespace mlir::sdfg::utils
