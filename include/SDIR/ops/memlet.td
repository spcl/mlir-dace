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

    let results = (outs SDIR_ArrayType);
    let assemblyFormat = [{attr-dict `(` `)` `:` type(results)}];
}


//===----------------------------------------------------------------------===//
// AllocTransientOp
//===----------------------------------------------------------------------===//

def SDIR_AllocTransientOp : SDIR_Op<"alloc_transient", [
    ParentOneOf<["SDFGNode", "StateNode"]>
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
    ParentOneOf<["StateNode", "MapNode", "ConsumeNode"]>,
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

  let arguments = (ins SDIR_MemletType:$arr, Variadic<Index>:$indices);
  let results = (outs AnyType:$res);
  let assemblyFormat = [{attr-dict $arr `[` $indices `]` `:` type($arr) `->` type($res)}];

  let verifier = [{
    size_t idx_size = indices().size();
    size_t mem_size = arr().getType().cast<MemletType>().getRank();
    if (idx_size != mem_size)
      return emitOpError("incorrect number of indices for load");
    return success();
  }];
}


//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

def SDIR_StoreOp : SDIR_Op<"store", [
    ParentOneOf<["StateNode", "MapNode", "ConsumeNode"]>,
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

  let arguments = (ins SDIR_MemletType:$arr, Variadic<Index>:$indices, AnyType:$val);
  let assemblyFormat = [{attr-dict $val `,` $arr `[` $indices`]` `:` type($val) `->` type($arr)}];

  let verifier = [{
    size_t idx_size = indices().size();
    size_t mem_size = arr().getType().cast<MemletType>().getRank();
    if (idx_size != mem_size)
      return emitOpError("incorrect number of indices for load");
    return success();
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
    let assemblyFormat = [{attr-dict $src `->` $dest `:` type($src)}];
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
  let assemblyFormat = [{attr-dict $src `:` type($src) `->` type(results)}];

  let verifier = [{
      size_t src_size = src().getType().cast<MemletType>().getRank();
      size_t res_size = res().getType().cast<MemletType>().getRank();
      if (src_size != res_size)
          return emitOpError("incorrect rank for memlet_cast");
      return success();
  }];
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
  let assemblyFormat = [{attr-dict $src `:` type($src) `->` type(results)}];

  let verifier = [{
      size_t src_size = src().getType().cast<MemletType>().getRank();
      size_t res_size = res().getType().cast<MemletType>().getRank();
      if (src_size != res_size)
          return emitOpError("incorrect rank for view_cast");
      return success();
  }];
}

#endif // SDIR_MemletOps
