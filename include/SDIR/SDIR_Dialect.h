#ifndef SDIR_SDIR_DIALECT_H
#define SDIR_SDIR_DIALECT_H

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"

#include "mlir/IR/Types.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/TypeSwitch.h"

#include "SDIR/SDIR_OpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "SDIR/SDIR_OpsTypes.h.inc"

#define GET_OP_CLASSES
#include "SDIR/SDIR_Ops.h.inc"

#endif // SDIR_SDIR_DIALECT_H