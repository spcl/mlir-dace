// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

/// Header for Generic to SDFG conversion passes.

#ifndef SDFG_Conversion_GenericToSDFG_H
#define SDFG_Conversion_GenericToSDFG_H

#include "mlir/Pass/Pass.h"

namespace mlir::sdfg::conversion {

/// Generate the code for declaring passes.
#define GEN_PASS_DECL
#include "SDFG/Conversion/GenericToSDFG/Passes.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "SDFG/Conversion/GenericToSDFG/Passes.h.inc"

} // namespace mlir::sdfg::conversion

#endif // SDFG_Conversion_GenericToSDFG_H
