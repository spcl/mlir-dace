#ifndef SDIR_AllocTransientOp
#define SDIR_AllocTransientOp

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

#endif // SDIR_AllocTransientOp