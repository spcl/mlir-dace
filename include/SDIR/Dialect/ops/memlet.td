#ifndef SDIR_MemletOps
#define SDIR_MemletOps

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

def SDIR_AllocOp : SDIR_Op<"alloc", [
    ParentOneOf<["SDFGNode", "StateNode"]>
]> {
    let summary = "Array allocation operation";
    let description = [{
        Alloc operation to create arrays and reserve the specified space.
        For example:

        ```mlir
            %A = sdir.alloc() : !sdir.array<i32>
        ```
    }];

    let arguments = (ins 
        Variadic<Index>:$params,
        OptionalAttr<StrAttr>:$name
    );
    let results = (outs SDIR_ArrayType:$res);

    let extraClassDeclaration = [{
        static AllocOp create(PatternRewriter &rewriter, Location loc, Type res);
        static AllocOp create(Location loc, Type res, StringRef name);
        SDFGNode getParentSDFG();
        bool isInState();
        std::string getName();
    }];
}


//===----------------------------------------------------------------------===//
// AllocTransientOp
//===----------------------------------------------------------------------===//

def SDIR_AllocTransientOp : SDIR_Op<"alloc_transient", [
    ParentOneOf<["SDFGNode", "StateNode"]>
]> {
    let summary = "Transient array allocation operation";
    let description = [{
        Alloc operation to create transient arrays and reserve the specified 
        space.
        For example:

        ```mlir
            %A = sdir.alloc_transient() : !sdir.array<i32>
        ```
    }];

    let arguments = (ins Variadic<Index>:$params);
    let results = (outs SDIR_ArrayType:$res);

    let extraClassDeclaration = [{
        SDFGNode getParentSDFG();
        bool isInState();
    }];
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

def SDIR_LoadOp : SDIR_Op<"load", [
    ParentOneOf<["StateNode", "MapNode", "ConsumeNode"]>,
    TypesMatchWith<"result type matches element type of 'memlet'", "arr", "res",
                    "$_self.cast<MemletType>().getElementType()">
]> {
    let summary = "Memlet load operation";
    let description = [{
        Allows a state to load a value from a memlet.
        For example:

        ```mlir
        sdir.state @state_0 {
            ...
            %a = sdir.get_access %A : !sdir.memlet<i32>
            %a_1 = sdir.load %a[0] : !sdir.memlet<i32> -> i32
            ...
        }
        ```
    }];

    let arguments = (ins Variadic<Index>:$indices, SDIR_MemletType:$arr);
    let results = (outs AnyType:$res);

    let extraClassDeclaration = [{
        static LoadOp create(PatternRewriter &rewriter, Location loc, GetAccessOp acc, ValueRange indices);
        static LoadOp create(Location loc, GetAccessOp acc, ValueRange indices);
        static LoadOp create(PatternRewriter &rewriter, Location loc, Type t, Value mem, ValueRange indices);
        static LoadOp create(Location loc, Type t, Value mem, ValueRange indices);
        bool isIndirect();
    }];
}


//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

def SDIR_StoreOp : SDIR_Op<"store", [
    ParentOneOf<["StateNode", "MapNode", "ConsumeNode"]>,
    TypesMatchWith<"value type matches element type of 'memlet'", "arr", "val",
                    "$_self.cast<MemletType>().getElementType()">
]> {
    let summary = "Memlet store operation";
    let description = [{
        Allows a state to store a value in a memlet.
        For example:

        ```mlir
        sdir.state @state_0 {
            ...
            %1 = arith.constant 1 : i32
            %a = sdir.get_access %A : !sdir.memlet<i32>
            sdir.store %1, %a[0] : !sdir.memlet<i32>
            ...
        }
        ```
    }];

    let arguments = (ins 
        Variadic<Index>:$indices, 
        AnyType:$val, 
        SDIR_MemletType:$arr
    );

    let extraClassDeclaration = [{
        static StoreOp create(PatternRewriter &rewriter, Location loc, Value val, Value mem, ValueRange indices);
        static StoreOp create(Location loc, Value val, Value mem, ValueRange indices);
        bool isIndirect();
    }];
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

def SDIR_CopyOp : SDIR_Op<"copy", [
    ParentOneOf<["StateNode", "MapNode", "ConsumeNode"]>,
    SameTypeOperands
]> {
    let summary = "Memlet copy operation";
    let description = [{
        Allows a state to copy the contents from one memlet to another.
        For example:

        ```mlir
        sdir.state @state_0 {
            ...
            sdir.copy %a -> %c : !sdir.memlet<i32>
            ...
        }
        ```
    }];

    let arguments = (ins SDIR_MemletType:$src, SDIR_MemletType:$dest);
}

//===----------------------------------------------------------------------===//
// MemletCastOp
//===----------------------------------------------------------------------===//

def SDIR_MemletCastOp : SDIR_Op<"memlet_cast", [
    ParentOneOf<["StateNode", "MapNode", "ConsumeNode"]>
]> {
    let summary = "memlet casting operation";
    let description = [{
        Casts a memlet from one view to another.
        For example:

        ```mlir
        sdir.state {
            ...
            %B = sdir.memlet_cast %A : !sdir.memlet<2x16xi32> -> !sdir.memlet<32xi32>
            ...
        }
        ```
    }];

    let arguments = (ins SDIR_MemletType:$src);
    let results = (outs SDIR_MemletType:$res);
}

//===----------------------------------------------------------------------===//
// ViewCastOp
//===----------------------------------------------------------------------===//

def SDIR_ViewCastOp : SDIR_Op<"view_cast", [
    ParentOneOf<["StateNode", "MapNode", "ConsumeNode"]>
]> {
    let summary = "view node operation";
    let description = [{
        Represents a view node to cast a memlet from one view to another.
        For example:

        ```mlir
        sdir.state {
            ...
            %B = sdir.view_cast %A : !sdir.memlet<2x16xi32> -> !sdir.memlet<32xi32>
            ...
        }
        ```
    }];

    let arguments = (ins SDIR_MemletType:$src);
    let results = (outs SDIR_MemletType:$res);
}

//===----------------------------------------------------------------------===//
// SubviewOp
//===----------------------------------------------------------------------===//

def SDIR_SubviewOp : SDIR_Op<"subview", [
    ParentOneOf<["StateNode", "MapNode", "ConsumeNode"]>
]> {
    let summary = "Subview operation";
    let description = [{
        Return a subview of a memlet with the provided offsets, sizes and 
        strides.
        For example:

        ```mlir
        sdir.state {
            ...
            %B = sdir.subview %A[3, 4, 2][1, 6, 3][1, 1, 1] : !sdir.memlet<8x16x4xi32> -> !sdir.memlet<6x3xi32>
            ...
        }
        ```
    }];

    let arguments = (ins 
        Variadic<Index>:$ranges,
        SDIR_MemletType:$src,
        ArrayAttr:$offsets,
        ArrayAttr:$sizes,
        ArrayAttr:$strides
    );
    let results = (outs SDIR_MemletType:$res);
}

#endif // SDIR_MemletOps
