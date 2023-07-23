// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

#ifndef SDFG_Dialect
#define SDFG_Dialect

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"

//===----------------------------------------------------------------------===//
// SDFG Dialect
//===----------------------------------------------------------------------===//

def SDFG_Dialect : Dialect{
    let name = "sdfg";
    let summary = "A high-level dialect for representing SDFGs.";
    let description = [{A high-level dialect for representing SDFGs.}];
    let cppNamespace = "::mlir::sdfg";
    let useDefaultTypePrinterParser = 1;
    let useFoldAPI = kEmitFoldAdaptorFolder;
}

//===----------------------------------------------------------------------===//
// SDFG Types
//===----------------------------------------------------------------------===//

class SDFG_Type<string name, list<Trait> traits = []> : 
                                        TypeDef<SDFG_Dialect, name, traits>{}

//===----------------------------------------------------------------------===//
// SizedType
//===----------------------------------------------------------------------===//

def SDFG_SizedType : SDFG_Type<"Sized">{
    let parameters = (ins
        "Type":$elementType,
        ArrayRefParameter<"StringAttr">:$symbols,
        ArrayRefParameter<"int64_t">:$integers, 
        ArrayRefParameter<"bool">:$shape 
    );

    let extraClassDeclaration = [{
        size_t getRank(){ return getShape().size(); }

        size_t getUndefRank(){ 
            size_t undefSize = 0;
            for(int64_t dim : getIntegers()) if(dim < 0) undefSize++;
            return undefSize;
        }

        bool hasZeros(){
            for(int64_t dim : getIntegers()) if(dim == 0) return true;
            return false;
        }
    }];
}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

def SDFG_ArrayType : SDFG_Type<"Array">{
    let mnemonic = "array";
    let summary = "A array type";
    let description = [{
        A type representing a array.
    }];

    let parameters = (ins SDFG_SizedType:$dimensions);
    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        Type getElementType();
        ArrayRef<StringAttr> getSymbols();
        ArrayRef<int64_t> getIntegers();
        ArrayRef<bool> getShape();
    }];
}

//===----------------------------------------------------------------------===//
// StreamType
//===----------------------------------------------------------------------===//

def SDFG_StreamType : SDFG_Type<"Stream">{
    let mnemonic = "stream";
    let summary = "A stream type";
    let description = [{
        A type representing a stream.
    }];

    let parameters = (ins SDFG_SizedType:$dimensions);
    let hasCustomAssemblyFormat = 1;
}

#endif // SDFG_Dialect
