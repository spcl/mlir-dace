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

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

def SDIR_ArrayType : SDIR_Type<"Array"> {
  let mnemonic = "array";
  let summary = "A array type";
  let description = [{
    A type representing a array.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape, 
    "Type":$elementType
  );

  let printer = [{
    $_printer << "array<";
    for(int i = 0; i < getImpl()->shape.size(); i++)
      $_printer << getImpl()->shape[i] << "x" ;
    $_printer << getImpl()->elementType << ">";
  }];

  let parser = [{
    if ($_parser.parseLess())
      return Type();
    ::llvm::SmallVector<int64_t> shape;
    if($_parser.parseDimensionList(shape))
      return Type();
    Type elementType;
    if ($_parser.parseType(elementType))
      return Type();
    if ($_parser.parseGreater())
      return Type();
    return get($_ctxt, shape, elementType);
  }];
}

//===----------------------------------------------------------------------===//
// MemletType
//===----------------------------------------------------------------------===//

def SDIR_MemletType : SDIR_Type<"Memlet"> {
  let mnemonic = "memlet";
  let summary = "A memlet type";
  let description = [{
    A type representing a memlet.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape, 
    "Type":$elementType
  );

  let printer = [{
    $_printer << "memlet<";
    for(int i = 0; i < getImpl()->shape.size(); i++)
      $_printer << getImpl()->shape[i] << "x" ;
    $_printer << getImpl()->elementType << ">";
  }];

  let parser = [{
    if ($_parser.parseLess())
      return Type();
    ::llvm::SmallVector<int64_t> shape;
    if($_parser.parseDimensionList(shape))
      return Type();
    Type elementType;
    if ($_parser.parseType(elementType))
      return Type();
    if ($_parser.parseGreater())
      return Type();
    return get($_ctxt, shape, elementType);
  }];
}

//===----------------------------------------------------------------------===//
// StreamArrayType
//===----------------------------------------------------------------------===//

def SDIR_StreamArrayType : SDIR_Type<"StreamArray"> {
  let mnemonic = "stream_array";
  let summary = "A stream array type";
  let description = [{
    A type representing a stream array.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape, 
    "Type":$elementType
  );

  let printer = [{
    $_printer << "stream_array<";
    for(int i = 0; i < getImpl()->shape.size(); i++)
      $_printer << getImpl()->shape[i] << "x" ;
    $_printer << getImpl()->elementType << ">";
  }];

  let parser = [{
    if ($_parser.parseLess())
      return Type();
    ::llvm::SmallVector<int64_t> shape;
    if($_parser.parseDimensionList(shape))
      return Type();
    Type elementType;
    if ($_parser.parseType(elementType))
      return Type();
    if ($_parser.parseGreater())
      return Type();
    return get($_ctxt, shape, elementType);
  }];
}

//===----------------------------------------------------------------------===//
// StreamType
//===----------------------------------------------------------------------===//

def SDIR_StreamType : SDIR_Type<"Stream"> {
  let mnemonic = "stream";
  let summary = "A stream type";
  let description = [{
    A type representing a stream.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape, 
    "Type":$elementType
  );

  let printer = [{
    $_printer << "stream<";
    for(int i = 0; i < getImpl()->shape.size(); i++)
      $_printer << getImpl()->shape[i] << "x" ;
    $_printer << getImpl()->elementType << ">";
  }];

  let parser = [{
    if ($_parser.parseLess())
      return Type();
    ::llvm::SmallVector<int64_t> shape;
    if($_parser.parseDimensionList(shape))
      return Type();
    Type elementType;
    if ($_parser.parseType(elementType))
      return Type();
    if ($_parser.parseGreater())
      return Type();
    return get($_ctxt, shape, elementType);
  }];
}

#endif // SDIR_Dialect
