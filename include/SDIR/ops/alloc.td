#ifndef SDIR_AllocOp
#define SDIR_AllocOp

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

#endif // SDIR_AllocOp