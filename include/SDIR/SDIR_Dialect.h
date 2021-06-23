#ifndef SDIR_SDIR_DIALECT_H
#define SDIR_SDIR_DIALECT_H

#include "mlir/IR/Dialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"

#include "SDIR/SDIR_OpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "SDIR/SDIR_OpsTypes.h.inc"

#define GET_OP_CLASSES
#include "SDIR/SDIR_Ops.h.inc"

#endif // SDIR_SDIR_DIALECT_H