#include "SDIR/SDIR_Dialect.h"
#include "SDIR/SDIR_Ops.h"

using namespace mlir;
using namespace mlir::sdir;

//===----------------------------------------------------------------------===//
// SDIR dialect
//===----------------------------------------------------------------------===//

void SDIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "SDIR/SDIR_Ops.cpp.inc"
      >();
}