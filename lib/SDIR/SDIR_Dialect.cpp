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
  ::mlir::Type value;
  ::mlir::OptionalParseResult opr;

  opr = generatedTypeParser(getContext(), parser, "stream", value);
  if(opr.hasValue() && opr.getValue().succeeded()) return value;

  opr = generatedTypeParser(getContext(), parser, "memlet", value);
  if(opr.hasValue() && opr.getValue().succeeded()) return value;

  return ::mlir::Type();
}

void SDIRDialect::printType(::mlir::Type type, ::mlir::DialectAsmPrinter &os) const{
  ::mlir::LogicalResult logRes = generatedTypePrinter(type, os);
  if(logRes.failed())
    ::mlir::emitError(nullptr, "Failed to print dialect type");
}