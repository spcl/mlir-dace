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

def SDIR_MemletType : SDIR_Type<"Memlet"> {
  let mnemonic = "memlet";
  let summary = "A memlet type";
  let description = [{
    A type representing a memlet.
  }];

  let parameters = (ins "Type":$elementType);

  let printer = [{
    $_printer << "memlet<" << getImpl()->elementType << ">";
  }];

  let parser = [{
    if ($_parser.parseLess())
      return Type();
    Type elementType;
    if ($_parser.parseType(elementType))
      return Type();
    if ($_parser.parseGreater())
      return Type();
    return get($_ctxt, elementType);
  }];
}

def SDIR_StreamType : SDIR_Type<"Stream"> {
  let mnemonic = "stream";
  let summary = "A stream type";
  let description = [{
    A type representing a stream.
  }];

  let parameters = (ins "Type":$elementType);

  let printer = [{
    $_printer << "stream<" << getImpl()->elementType << ">";
  }];

  let parser = [{
    if ($_parser.parseLess())
      return Type();
    Type elementType;
    if ($_parser.parseType(elementType))
      return Type();
    if ($_parser.parseGreater())
      return Type();
    return get($_ctxt, elementType);
  }];
}

#endif // SDIR_Dialect