#include "SDIR/SDIR_Dialect.h"

using namespace mlir;
using namespace mlir::sdir;

#include "SDIR/SDIR_OpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// SDIR Dialect
//===----------------------------------------------------------------------===//

void SDIRDialect::initialize() {
  addOperations<
    #define GET_OP_LIST
    #include "SDIR/SDIR_Ops.cpp.inc"
  >();

  addTypes<
    #define GET_TYPEDEF_LIST
    #include "SDIR/SDIR_OpsTypes.cpp.inc"
  >();
}

//===----------------------------------------------------------------------===//
// SDIR Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "SDIR/SDIR_OpsTypes.cpp.inc"

::mlir::Type SDIRDialect::parseType(::mlir::DialectAsmParser &parser) const{

}

void SDIRDialect::printType(::mlir::Type type, ::mlir::DialectAsmPrinter &os) const{

}