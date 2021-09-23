#ifndef SDIR_DIALECT_DIALECT_H
#define SDIR_DIALECT_DIALECT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

#include "SDIR/Dialect/OpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "SDIR/Dialect/OpsTypes.h.inc"

#define GET_OP_CLASSES
#include "SDIR/Dialect/Ops.h.inc"

#endif // SDIR_DIALECT_DIALECT_H
