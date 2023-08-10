// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

/// Header for the ID generator utility functions.

#ifndef SDFG_Utils_IDGenerator_H
#define SDFG_Utils_IDGenerator_H

namespace mlir::sdfg::utils {

/// Returns a globally unique ID.
unsigned generateID();
/// Resets the ID generator.
void resetIDGenerator();

} // namespace mlir::sdfg::utils

#endif // SDFG_Utils_IDGenerator_H
