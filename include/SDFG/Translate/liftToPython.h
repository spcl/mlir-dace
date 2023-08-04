// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

#ifndef SDFG_Translation_LiftToPython_H
#define SDFG_Translation_LiftToPython_H

#include "SDFG/Dialect/Dialect.h"

namespace mlir::sdfg::translation {

/// Converts the operations in the first region of op to Python code.
Optional<std::string> liftToPython(Operation &op);
/// Provides a name for the tasklet.
std::string getTaskletName(Operation &op);

} // namespace mlir::sdfg::translation

#endif // SDFG_Translation_LiftToPython_H
