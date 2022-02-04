#ifndef SDIR_Dialect
#define SDIR_Dialect

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"

//===----------------------------------------------------------------------===//
// SDIR Dialect
//===----------------------------------------------------------------------===//

def SDIR_Dialect : Dialect{
    let name = "sdir";
    let summary = "A high-level dialect for representing SDFGs.";
    let description = [{A high-level dialect for representing SDFGs.}];
    let cppNamespace = "::mlir::sdir";
}

//===----------------------------------------------------------------------===//
// SDIR Types
//===----------------------------------------------------------------------===//

class SDIR_Type<string name, list<Trait> traits = []> : 
                                        TypeDef<SDIR_Dialect, name, traits>{}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

def SDIR_ArrayType : SDIR_Type<"Array">{
    let mnemonic = "array";
    let summary = "A array type";
    let description = [{
        A type representing a array.
    }];

    let parameters = (ins
        "Type":$elementType,
        ArrayRefParameter<"StringAttr">:$symbols,
        ArrayRefParameter<"int64_t">:$integers, 
        ArrayRefParameter<"bool">:$shape 
    );

    let printer = [{
        printDimensionList($_printer,getImpl()->elementType, getImpl()->symbols,
                        getImpl()->integers, getImpl()->shape);
    }];

    let parser = [{
        Type elementType;
        SmallVector<StringAttr> symbols;
        SmallVector<int64_t> integers;
        SmallVector<bool> shape;
        if(parseDimensionList($_parser, elementType, symbols, integers, shape))
            return Type();
        
        return get($_ctxt, elementType, symbols, integers, shape);
    }];

    let extraClassDeclaration = [{
        size_t getRank(){ return getShape().size(); }

        size_t getUndefRank(){ 
            size_t undefSize = 0;
            for(int64_t dim : getIntegers()) if(dim == -1) undefSize++;
            return undefSize;
        }

        bool hasZeros(){
            for(int64_t dim : getIntegers()) if(dim == 0) return true;
            return false;
        }
    }];
}

//===----------------------------------------------------------------------===//
// StreamType
//===----------------------------------------------------------------------===//

def SDIR_StreamType : SDIR_Type<"Stream">{
    let mnemonic = "stream";
    let summary = "A stream type";
    let description = [{
        A type representing a stream.
    }];

    let parameters = (ins
        "Type":$elementType,
        ArrayRefParameter<"StringAttr">:$symbols,
        ArrayRefParameter<"int64_t">:$integers, 
        ArrayRefParameter<"bool">:$shape 
    );

    let printer = [{
        printDimensionList($_printer,getImpl()->elementType, getImpl()->symbols,
                        getImpl()->integers, getImpl()->shape);
    }];

    let parser = [{
        Type elementType;
        SmallVector<StringAttr> symbols;
        SmallVector<int64_t> integers;
        SmallVector<bool> shape;
        if(parseDimensionList($_parser, elementType, symbols, integers, shape))
            return Type();
        
        return get($_ctxt, elementType, symbols, integers, shape);
    }];
}

#endif // SDIR_Dialect
