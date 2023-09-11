// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

/// This file contains the ID generator utility functions.

#include "SDFG/Utils/IDGenerator.h"

namespace mlir::sdfg::utils {
namespace {
unsigned idGeneratorID = 0;
}

/// Returns a globally unique ID.
unsigned generateID() { return idGeneratorID++; }

/// Resets the ID generator.
void resetIDGenerator() { idGeneratorID = 0; }

} // namespace mlir::sdfg::utils
