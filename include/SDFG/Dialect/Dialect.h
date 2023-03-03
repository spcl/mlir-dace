#ifndef SDFG_DIALECT_DIALECT_H
#define SDFG_DIALECT_DIALECT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

#include "SDFG/Dialect/OpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "SDFG/Dialect/OpsTypes.h.inc"

#define GET_OP_CLASSES
#include "SDFG/Dialect/Ops.h.inc"

#endif // SDFG_DIALECT_DIALECT_H
