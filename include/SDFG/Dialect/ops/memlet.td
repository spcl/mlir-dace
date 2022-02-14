#ifndef SDFG_MemletOps
#define SDFG_MemletOps

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

def SDFG_AllocOp : SDFG_Op<"alloc", [
    ParentOneOf<["SDFGNode", "StateNode"]>
]> {
    let summary = "Array allocation operation";
    let description = [{
        Alloc operation to create arrays and reserve the specified space.
        For example:

        ```mlir
            %A = sdfg.alloc() : !sdfg.array<i32>
        ```
    }];

    let arguments = (ins 
        Variadic<Index>:$params,
        OptionalAttr<StrAttr>:$name,
        UnitAttr:$transient
    );
    let results = (outs AnyTypeOf<[SDFG_ArrayType, SDFG_StreamType]>:$res);

    let extraClassDeclaration = [{
        static AllocOp create(PatternRewriter &rewriter, Location loc, Type res, bool transient);
        static AllocOp create(PatternRewriter &rewriter, Location loc, Type res, StringRef name, bool transient);
        static AllocOp create(Location loc, Type res, StringRef name, bool transient);
        SDFGNode getParentSDFG();
        Type getElementType();
        bool isScalar();
        bool isInState();
        std::string getName();
    }];
}


//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

def SDFG_LoadOp : SDFG_Op<"load", [
    ParentOneOf<["StateNode", "MapNode", "ConsumeNode"]>,
    TypesMatchWith<"result type matches element type of 'memlet'", "arr", "res",
                    "$_self.cast<ArrayType>().getDimensions().getElementType()">
]> {
    let summary = "Memlet load operation";
    let description = [{
        Allows a state to load a value from a memlet.
        For example:

        ```mlir
        sdfg.state @state_0 {
            ...
            %a = sdfg.get_access %A : !sdfg.memlet<i32>
            %a_1 = sdfg.load %a[0] : !sdfg.memlet<i32> -> i32
            ...
        }
        ```
    }];

    let arguments = (ins Variadic<Index>:$indices, SDFG_ArrayType:$arr);
    let results = (outs AnyType:$res);

    let extraClassDeclaration = [{
        static LoadOp create(PatternRewriter &rewriter, Location loc, AllocOp alloc, ValueRange indices);
        static LoadOp create(Location loc, AllocOp alloc, ValueRange indices);
        static LoadOp create(PatternRewriter &rewriter, Location loc, Type t, Value mem, ValueRange indices);
        static LoadOp create(Location loc, Type t, Value mem, ValueRange indices);
        bool isIndirect();
    }];
}


//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

def SDFG_StoreOp : SDFG_Op<"store", [
    ParentOneOf<["StateNode", "MapNode", "ConsumeNode"]>,
    TypesMatchWith<"value type matches element type of 'memlet'", "arr", "val",
                    "$_self.cast<ArrayType>().getDimensions().getElementType()">
]> {
    let summary = "Memlet store operation";
    let description = [{
        Allows a state to store a value in a memlet.
        For example:

        ```mlir
        sdfg.state @state_0 {
            ...
            %1 = arith.constant 1 : i32
            %a = sdfg.get_access %A : !sdfg.memlet<i32>
            sdfg.store %1, %a[0] : !sdfg.memlet<i32>
            ...
        }
        ```
    }];

    let arguments = (ins 
        Variadic<Index>:$indices, 
        AnyType:$val, 
        SDFG_ArrayType:$arr
    );

    let extraClassDeclaration = [{
        static StoreOp create(PatternRewriter &rewriter, Location loc, Value val, Value mem, ValueRange indices);
        static StoreOp create(Location loc, Value val, Value mem, ValueRange indices);
        static StoreOp create(Location loc, Value val, Value mem, ArrayRef<StringRef> indices);
        static StoreOp create(Location loc, Value val, Value mem);
        bool isIndirect();
    }];
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

def SDFG_CopyOp : SDFG_Op<"copy", [
    ParentOneOf<["StateNode", "MapNode", "ConsumeNode"]>,
    SameTypeOperands
]> {
    let summary = "Memlet copy operation";
    let description = [{
        Allows a state to copy the contents from one memlet to another.
        For example:

        ```mlir
        sdfg.state @state_0 {
            ...
            sdfg.copy %a -> %c : !sdfg.memlet<i32>
            ...
        }
        ```
    }];

    let arguments = (ins SDFG_ArrayType:$src, SDFG_ArrayType:$dest);
}

//===----------------------------------------------------------------------===//
// MemletCastOp
//===----------------------------------------------------------------------===//

def SDFG_MemletCastOp : SDFG_Op<"memlet_cast", [
    ParentOneOf<["StateNode", "MapNode", "ConsumeNode"]>
]> {
    let summary = "memlet casting operation";
    let description = [{
        Casts a memlet from one view to another.
        For example:

        ```mlir
        sdfg.state {
            ...
            %B = sdfg.memlet_cast %A : !sdfg.memlet<2x16xi32> -> !sdfg.memlet<32xi32>
            ...
        }
        ```
    }];

    let arguments = (ins SDFG_ArrayType:$src);
    let results = (outs SDFG_ArrayType:$res);
}

//===----------------------------------------------------------------------===//
// ViewCastOp
//===----------------------------------------------------------------------===//

def SDFG_ViewCastOp : SDFG_Op<"view_cast", [
    ParentOneOf<["StateNode", "MapNode", "ConsumeNode"]>
]> {
    let summary = "view node operation";
    let description = [{
        Represents a view node to cast a memlet from one view to another.
        For example:

        ```mlir
        sdfg.state {
            ...
            %B = sdfg.view_cast %A : !sdfg.memlet<2x16xi32> -> !sdfg.memlet<32xi32>
            ...
        }
        ```
    }];

    let arguments = (ins SDFG_ArrayType:$src);
    let results = (outs SDFG_ArrayType:$res);
}

//===----------------------------------------------------------------------===//
// SubviewOp
//===----------------------------------------------------------------------===//

def SDFG_SubviewOp : SDFG_Op<"subview", [
    ParentOneOf<["StateNode", "MapNode", "ConsumeNode"]>
]> {
    let summary = "Subview operation";
    let description = [{
        Return a subview of a memlet with the provided offsets, sizes and 
        strides.
        For example:

        ```mlir
        sdfg.state {
            ...
            %B = sdfg.subview %A[3, 4, 2][1, 6, 3][1, 1, 1] : !sdfg.memlet<8x16x4xi32> -> !sdfg.memlet<6x3xi32>
            ...
        }
        ```
    }];

    let arguments = (ins 
        Variadic<Index>:$ranges,
        SDFG_ArrayType:$src,
        ArrayAttr:$offsets,
        ArrayAttr:$sizes,
        ArrayAttr:$strides
    );
    let results = (outs SDFG_ArrayType:$res);
}

#endif // SDFG_MemletOps
