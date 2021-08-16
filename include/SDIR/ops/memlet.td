#ifndef SDIR_MemletOps
#define SDIR_MemletOps

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

def SDIR_AllocOp : SDIR_Op<"alloc", [
    ParentOneOf<["SDFGOp", "StateOp"]>
]> {
  let summary = "Array allocation operation";
  let description = [{
    Alloc operation to create arrays and reserve the specified space.
    For example:

    ```mlir
        %A = sdir.alloc() : !sdir.array<i32>
    ```
  }];

    let results = (outs SDIR_ArrayType);
    let assemblyFormat = [{attr-dict `(` `)` `:` type(results)}];
}


//===----------------------------------------------------------------------===//
// AllocTransientOp
//===----------------------------------------------------------------------===//

def SDIR_AllocTransientOp : SDIR_Op<"alloc_transient", [
    ParentOneOf<["SDFGOp", "StateOp"]>
]> {
  let summary = "Transient array allocation operation";
  let description = [{
    Alloc operation to create transient arrays and reserve the specified space.
    For example:

    ```mlir
        %A = sdir.alloc_transient() : !sdir.array<i32>
    ```
  }];

    let results = (outs SDIR_ArrayType);
    let assemblyFormat = [{attr-dict `(` `)` `:` type(results)}];
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

def SDIR_LoadOp : SDIR_Op<"load", [
    HasParent<"StateOp">,
    TypesMatchWith<"result type matches element type of 'memlet'",
                     "arr", "res",
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

    let arguments = (ins SDIR_MemletType:$arr, I64Attr:$idx);
    let results = (outs AnyType:$res);
    let assemblyFormat = [{attr-dict $arr `[` $idx `]` `:` type($arr) `->` type($res)}];
}


//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

def SDIR_StoreOp : SDIR_Op<"store", [
    HasParent<"StateOp">,
    TypesMatchWith<"value type matches element type of 'memlet'",
                  "arr", "val",
                  "$_self.cast<MemletType>().getElementType()">
]> {
  let summary = "Memlet store operation";
  let description = [{
    Allows a state to store a value in a memlet.
    For example:

    ```mlir
    sdir.state @state_0 {
        ...
        %1 = constant 1 : i32
        %a = sdir.get_access %A : !sdir.memlet<i32>
        sdir.store %1, %a[0] : !sdir.memlet<i32>
        ...
    }
    ```
  }];

    let arguments = (ins SDIR_MemletType:$arr, I64Attr:$idx, AnyType:$val);
    let assemblyFormat = [{attr-dict $val `,` $arr `[` $idx `]` `:` type($arr)}];
}


//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

def SDIR_CopyOp : SDIR_Op<"copy", [
    HasParent<"StateOp">,
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
    let assemblyFormat = [{attr-dict $src `->` $dest `:` type($src)}];
}

#endif // SDIR_MemletOps