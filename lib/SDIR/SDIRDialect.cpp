#include "SDIR/SDIRDialect.h"
#include "SDIR/SDIROps.h"

using namespace mlir;
using namespace mlir::sdir;

//===----------------------------------------------------------------------===//
// SDIR dialect
//===----------------------------------------------------------------------===//

void SDIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "SDIR/SDIROps.cpp.inc"
      >();
}