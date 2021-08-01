#ifndef SDIR_Dialect
#define SDIR_Dialect

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"

//===----------------------------------------------------------------------===//
// SDIR Dialect
//===----------------------------------------------------------------------===//

def SDIR_Dialect : Dialect {
  let name = "sdir";
  let summary = "A high-level dialect for representing SDFGs.";
  let description = [{A high-level dialect for representing SDFGs.}];
  let cppNamespace = "::mlir::sdir";
}

//===----------------------------------------------------------------------===//
// SDIR Types
//===----------------------------------------------------------------------===//

class SDIR_Type<string name, list<Trait> traits = []> : TypeDef<SDIR_Dialect, name, traits>{}

def SDIR_IntegerType : SDIR_Type<"TestInteger"> {
  let mnemonic = "int";

  let summary = "An integer type with special semantics";

  let description = [{
    An alternate integer type. This type differentiates itself from the
    standard integer type by not having a SignednessSemantics parameter, just
    a width.
  }];

  let parameters = (ins "unsigned":$width);

  // We define the printer inline.
  let printer = [{
    $_printer << "int<" << getImpl()->width << ">";
  }];

  // The parser is defined here also.
  let parser = [{
    if ($_parser.parseLess())
      return Type();
    int width;
    if ($_parser.parseInteger(width))
      return Type();
    if ($_parser.parseGreater())
      return Type();
    return get($_ctxt, width);
  }];
}

#endif // SDIR_Dialect