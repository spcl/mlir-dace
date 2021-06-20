#ifndef SDIR_SDIR_OPS_H
#define SDIR_SDIR_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "SDIR/SDIR_Ops.h.inc"

#endif // SDIR_SDIR_OPS_H